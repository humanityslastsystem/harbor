"""What the exec channel says when it dies.

``_describe_server_exit`` is the only account of why a sandbox vanished mid-run, and it
is read at the one moment nothing else is left: the control channel has dropped, the
rollout is over, and the message goes into the record as the cause.

Two things make the naive reading wrong, and both are pinned here.

The launch process has not been reaped at the instant the exec raises -- the HTTP EOF and
the process death arrive together -- so ``returncode`` is still ``None`` and the
describer reaches for its least useful branch, reporting that the process is still
running in the exact case where it is gone. It has to wait for the status it is about to
describe.

And a payload killed inside the container surfaces through apptainer as a positive
``128+N``, not a negative code. ``rc < 0`` only fires when the host signalled the launch
process itself, so a SIGKILLed server reads as an ordinary non-zero exit and the one
number that names the killer is left undecoded.
"""

import asyncio
import signal
import sys

import httpx
import pytest

from harbor.environments.launch import LaunchSpec
from harbor.environments.singularity.singularity import (
    MemoryLimitExceededError,
    SingularityEnvironment,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="exit statuses are POSIX shaped"
)


class _Proc:
    """A launch process that reports ``returncode`` only once awaited, as asyncio's do."""

    def __init__(self, returncode, *, wait_forever: bool = False):
        self._final = returncode
        self._wait_forever = wait_forever
        self.returncode = None
        self.waited = False

    async def wait(self):
        if self._wait_forever:
            await asyncio.sleep(3600)
        self.waited = True
        self.returncode = self._final
        return self._final


def _env(proc) -> SingularityEnvironment:
    env = SingularityEnvironment.__new__(SingularityEnvironment)
    env._server_process = proc
    return env


def _describe(proc) -> str:
    return asyncio.run(_env(proc)._describe_server_exit())


def test_a_dead_server_is_not_reported_as_running():
    """The launch process is unreaped when the exec raises, so the describer has to wait
    for the status. Reporting "still running" here points the next reader away from the
    death that just happened."""
    proc = _Proc(returncode=1)

    described = _describe(proc)

    assert proc.waited, "described the exit without waiting for it"
    assert "still running" not in described, described
    assert "1" in described


def test_a_signal_inside_the_container_is_decoded_from_its_positive_status():
    """Apptainer surfaces a payload killed in the container as 128+N. Left undecoded it
    reads as an ordinary non-zero exit, and the number that names the killer is lost."""
    described = _describe(_Proc(returncode=128 + int(signal.SIGKILL)))

    assert "137" in described
    assert "SIGKILL" in described


def test_a_signal_to_the_launch_process_itself_still_reads_as_one():
    """The host signalling the launch process gives a negative code. That path keeps its
    own wording."""
    described = _describe(_Proc(returncode=-int(signal.SIGTERM)))

    assert "SIGTERM" in described


def test_an_ordinary_failure_is_not_dressed_up_as_a_signal():
    """Only the 128+N band decodes. A plain non-zero exit says so, without inventing a
    signal for a status that never carried one."""
    described = _describe(_Proc(returncode=2))

    assert "2" in described
    assert "SIG" not in described


def test_a_server_that_will_not_be_reaped_does_not_hang_the_report():
    """The wait is bounded: a launch process that never exits must not stop the rollout
    from recording why it ended. It falls back to the unknown-status wording."""
    proc = _Proc(returncode=0, wait_forever=True)

    described = _describe(proc)

    assert described, "no account at all when the status could not be read"
    assert "still running" in described


def test_no_server_process_at_all():
    assert "no server process" in _describe(None)


def _launched_env(tmp_path, proc) -> SingularityEnvironment:
    """A fully built environment whose launch process is ``proc``."""
    sif = tmp_path / "img.sif"
    sif.write_bytes(b"")
    env = SingularityEnvironment.launch(LaunchSpec(image=str(sif), gpus=0))
    env._server_process = proc
    return env


class _DroppingClient:
    """An HTTP client whose connection dies partway through reading the reply."""

    def __init__(self, before_drop=None):
        self._before_drop = before_drop

    async def post(self, *args, **kwargs):
        if self._before_drop is not None:
            self._before_drop()
        raise httpx.ReadError("")


def test_a_read_error_is_diagnosed_as_a_lost_channel(tmp_path, caplog):
    """A ReadError is the control channel dying mid-reply. It gets the same diagnosis
    as the other channel failures and still propagates, so the rollout ends on it
    rather than handing the agent a failed command."""
    env = _launched_env(tmp_path, _Proc(returncode=128 + int(signal.SIGKILL)))
    env._http_client = _DroppingClient()
    env._control_base = "http://localhost"

    with pytest.raises(httpx.ReadError):
        asyncio.run(env.exec("true"))

    assert "Exec channel lost" in caplog.text
    assert "137" in caplog.text and "SIGKILL" in caplog.text


def test_a_read_error_after_the_memory_watchdog_fired_names_the_limit(tmp_path):
    """If the watchdog killed the server while the request was in flight, the limit is
    the cause to report, not the dropped connection."""
    env = _launched_env(tmp_path, _Proc(returncode=-int(signal.SIGKILL)))
    limit = "Container exceeded memory limit (2MB > 1MB)"

    def watchdog_fires():
        env._memory_limit_exceeded = limit

    env._http_client = _DroppingClient(before_drop=watchdog_fires)
    env._control_base = "http://localhost"

    with pytest.raises(MemoryLimitExceededError, match="exceeded memory limit"):
        asyncio.run(env.exec("true"))


def test_stop_describes_an_earlier_death_as_the_exec_path_does(tmp_path, caplog):
    """An in-container SIGKILL arrives as 137. stop() reads the status the way the exec
    path does: a likely SIGKILL, not a self-exit."""
    proc = _Proc(returncode=128 + int(signal.SIGKILL))
    proc.pid = 4242
    asyncio.run(proc.wait())
    env = _launched_env(tmp_path, proc)

    asyncio.run(env.stop(delete=False))

    assert asyncio.run(env._describe_server_exit()) in caplog.text
    assert "self-exit" not in caplog.text
    assert "OOM killer" in caplog.text
