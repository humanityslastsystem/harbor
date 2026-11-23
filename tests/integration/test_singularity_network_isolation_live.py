"""Whether a real rootless apptainer isolates a container's network.

The unit tests check that the flags are built; this one checks that apptainer accepts
them and that the container has no route out. It needs apptainer (or singularity) and
busybox on PATH and is skipped otherwise. A host whose kernel forbids unprivileged user
namespaces (``kernel.apparmor_restrict_unprivileged_userns = 1``) cannot run rootless
apptainer at all, so run it where that works::

    uv run pytest tests/integration/test_singularity_network_isolation_live.py -v

The probe is two-sided: the same command must reach the network unsealed and be
blocked sealed, in the same image on the same host. A sealed check alone would pass on
a probe that can never succeed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.integration

#: A literal address, so a failure means no route rather than no resolver.
PROBE_HOST = "1.1.1.1"
PROBE = (
    f"(wget -q -T 5 -O /dev/null http://{PROBE_HOST}/ && echo REACHED) || echo BLOCKED"
)
ISOLATE = ("--net", "--network", "none")

apptainer = shutil.which("apptainer") or shutil.which("singularity")
needs_apptainer = pytest.mark.skipif(
    apptainer is None, reason="no apptainer/singularity on PATH"
)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> str:
    """A sandbox directory holding busybox and nothing else.

    A directory rather than a ``.sif`` from ``docker://``, which would need ``proot``
    for unprivileged layer extraction."""
    busybox = shutil.which("busybox")
    if busybox is None:
        pytest.skip("no busybox to build a sandbox from")
    root = tmp_path_factory.mktemp("sandbox")
    for directory in ("bin", "etc", "proc", "sys", "dev", "tmp"):
        (root / directory).mkdir()
    shutil.copy(busybox, root / "bin" / "busybox")
    for applet in ("sh", "wget", "ip", "cat"):
        (root / "bin" / applet).symlink_to("busybox")
    return str(root)


def _run(sandbox: str, *flags: str, command: str) -> str:
    assert apptainer is not None
    result = subprocess.run(
        [apptainer, "exec", "--containall", *flags, sandbox, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (result.stdout + result.stderr).strip()


@needs_apptainer
def test_an_open_container_reaches_the_network(sandbox):
    assert "REACHED" in _run(sandbox, command=PROBE), (
        "the unsealed probe could not reach the network, so this host cannot tell "
        "isolation from a broken probe"
    )


@needs_apptainer
def test_an_isolated_container_has_no_route_out(sandbox):
    output = _run(sandbox, *ISOLATE, command=PROBE)
    assert "BLOCKED" in output, output
    assert "REACHED" not in output, output


@needs_apptainer
def test_an_isolated_container_keeps_its_loopback(sandbox):
    output = _run(sandbox, *ISOLATE, command="ip -o link show")
    assert "lo" in output, output


@needs_apptainer
def test_a_bind_mount_still_crosses_into_an_isolated_container(sandbox, tmp_path):
    # The exec server's control socket reaches the host through a bind of the staging
    # directory.
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "probe.txt").write_text("crossed")
    output = _run(
        sandbox,
        *ISOLATE,
        "-B",
        f"{staging}:/staging",
        command="cat /staging/probe.txt",
    )
    assert "crossed" in output, output
