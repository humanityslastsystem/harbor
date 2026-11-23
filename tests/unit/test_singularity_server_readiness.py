"""How quickly the launch notices that the exec server is up.

Readiness is asked every ``LaunchSpec.ready_poll_sec`` until a fixed budget,
``SERVER_READY_TIMEOUT_SEC``, runs out. The interval only bounds how long a start that
has already succeeded goes unnoticed; the budget alone decides whether a slow start
fails. A caller that opens one sandbox per trajectory pays the interval's rounding on
every start, so it can ask for a short one.

Two properties pull in opposite directions and both are pinned here: readiness is
noticed within a poll, and a slow start still gets the whole budget.
"""

import asyncio
import os
import time

import pytest

from harbor.environments.launch import LaunchSpec
from harbor.environments.singularity import singularity as sg
from harbor.environments.singularity.singularity import SingularityEnvironment


class _Stdout:
    """A server that says nothing and reaches EOF, so the output reader finishes."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _Proc:
    """A launch process that stays alive, as a healthy container's does.

    ``pid`` is this test process's own, so anything reaching for it finds a real and
    harmless target. Teardown is stubbed out below in any case."""

    def __init__(self):
        self.returncode = None
        self.pid = os.getpid()
        self.stdout = _Stdout()

    async def wait(self):
        # Long enough to be "still running", short enough that a bug is reported
        # rather than stalling the suite.
        await asyncio.sleep(5)


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class _Client:
    """A control channel that starts answering only after ``ready_after`` seconds."""

    def __init__(self, ready_after, clock):
        self._ready_at = clock() + ready_after
        self._clock = clock
        self.polls = 0

    async def get(self, url):
        self.polls += 1
        if self._clock() < self._ready_at:
            raise sg.httpx.RequestError("not up yet")
        return _Response(200)

    async def aclose(self):
        pass


def _env(tmp_path, **fields):
    sif = tmp_path / "img.sif"
    sif.write_bytes(b"")
    # Isolated, so the control channel is the UNIX socket and the launch makes exactly
    # one attempt: the measurement is about readiness and not about retries.
    return SingularityEnvironment.launch(
        LaunchSpec(image=str(sif), allow_internet=False, workdir="/app", **fields)
    )


def _patch(monkeypatch, client):
    async def _spawn(*args, **kwargs):
        return _Proc()

    async def _no_teardown(self):
        """No container was started, and the real one signals a process group by pid."""

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(sg.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setattr(
        SingularityEnvironment, "_record_container_init", lambda self: None
    )
    monkeypatch.setattr(SingularityEnvironment, "_terminate_sandbox", _no_teardown)


#: A short interval, as a caller that opens many sandboxes would ask for.
FAST_POLL_SEC = 0.05

#: How late readiness may be noticed, as a multiple of the poll interval: one interval
#: plus scheduling slack.
LATENESS_BUDGET = 3


#: A container that is up almost at once, and one that takes about as long as a real
#: one. A schedule that starts fast and backs off looks fine on the quick case and is
#: at its worst on the slow one.
@pytest.mark.parametrize("ready_after", [0.0, 0.2, 1.2])
async def test_readiness_is_noticed_within_a_poll_of_happening(
    tmp_path, monkeypatch, ready_after
):
    env = _env(tmp_path, ready_poll_sec=FAST_POLL_SEC)
    client = _Client(ready_after, time.monotonic)
    _patch(monkeypatch, client)

    started = time.monotonic()
    await env._start_server()
    late = time.monotonic() - started - ready_after

    assert late >= 0
    assert late < LATENESS_BUDGET * FAST_POLL_SEC, (
        f"ready at {ready_after:.2f}s, noticed {late * 1000:.0f}ms late; "
        f"the poll interval is {FAST_POLL_SEC * 1000:.0f}ms"
    )


async def test_a_slow_start_still_succeeds(tmp_path, monkeypatch):
    env = _env(tmp_path, ready_poll_sec=FAST_POLL_SEC)
    client = _Client(2.0, time.monotonic)
    _patch(monkeypatch, client)
    monkeypatch.setattr(sg, "SERVER_READY_TIMEOUT_SEC", 5.0)

    await env._start_server()
    assert env._http_client is client


@pytest.mark.parametrize("poll", [0.05, 0.005])
async def test_the_deadline_is_the_budget_and_not_a_count_of_polls(
    tmp_path, monkeypatch, poll
):
    # Deriving the budget from the interval (so many polls, so far apart) would
    # shorten it along with the interval. At two intervals an order of magnitude
    # apart, the time to give up must not move.
    env = _env(tmp_path, ready_poll_sec=poll)
    never = _Client(1e9, time.monotonic)
    _patch(monkeypatch, never)
    monkeypatch.setattr(sg, "SERVER_READY_TIMEOUT_SEC", 0.3)

    started = time.monotonic()
    with pytest.raises(RuntimeError):
        await env._start_server()
    took = time.monotonic() - started

    assert 0.3 <= took < 0.3 + 0.25, (
        f"gave up after {took:.2f}s at a {poll}s poll; the budget is 0.3s"
    )


async def test_a_slow_start_is_not_polled_into_the_ground(tmp_path, monkeypatch):
    # The interval is flat, so a long start costs the budget divided by it, not a
    # busy loop.
    env = _env(tmp_path, ready_poll_sec=FAST_POLL_SEC)
    client = _Client(1.0, time.monotonic)
    _patch(monkeypatch, client)

    await env._start_server()
    assert client.polls < 2 / FAST_POLL_SEC, (
        f"polled {client.polls} times in one second"
    )


async def test_the_default_interval_is_one_second(tmp_path, monkeypatch):
    # A launch that does not ask keeps the one-second interval.
    env = _env(tmp_path)
    client = _Client(0.5, time.monotonic)
    _patch(monkeypatch, client)

    started = time.monotonic()
    await env._start_server()
    took = time.monotonic() - started

    assert client.polls == 2
    assert 1.0 <= took < 1.5


def test_the_budget_and_the_default_interval():
    assert sg.SERVER_READY_TIMEOUT_SEC == 60.0
    assert LaunchSpec(image="img.sif").ready_poll_sec == 1.0


@pytest.mark.parametrize("poll", [0.0, -1.0])
def test_a_poll_interval_that_is_not_positive_is_refused(tmp_path, poll):
    with pytest.raises(ValueError, match="ready_poll_sec"):
        _env(tmp_path, ready_poll_sec=poll)
