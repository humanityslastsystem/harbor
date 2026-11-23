"""The bootstrap's choice of interpreter for the exec server.

An image without uvicorn and fastapi makes the bootstrap build a venv, which needs a
network or root. A sealed container has neither, so a launch may name an interpreter it
made available (``LaunchSpec.server_python``), which the bootstrap tries after the
image's own. Only the path that ends in ``exec`` is run here: the fallback after it
would try to install packages on this host.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BOOTSTRAP = (
    Path(__file__).parents[2] / "src/harbor/environments/singularity/bootstrap.sh"
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None,
    reason="the bootstrap is a bash script run inside a Linux container",
)


def _image_python_hosts_the_server() -> bool:
    """Whether an interpreter the bootstrap tries before the named one has the deps."""
    for cand in (
        shutil.which("python3", path="/usr/bin:/bin"),
        "/opt/venv/bin/python3",
    ):
        if cand and os.access(cand, os.X_OK):
            probe = subprocess.run(
                [cand, "-c", "import uvicorn, fastapi"], capture_output=True
            )
            if probe.returncode == 0:
                return True
    return False


def test_a_named_interpreter_hosts_the_server(tmp_path):
    if _image_python_hosts_the_server():
        pytest.skip("this host's own python3 has uvicorn and fastapi")
    fake = tmp_path / "python3"
    record = tmp_path / "served"
    # Passes the bootstrap's import probe, then records how it was started.
    fake.write_text(
        f'#!/bin/bash\nif [ "$1" = "-c" ]; then exit 0; fi\necho "$@" > {record}\n'
    )
    fake.chmod(0o755)
    workdir = tmp_path / "work"

    result = subprocess.run(
        ["bash", str(BOOTSTRAP), str(workdir), "/staging/_hbexec.py", "--uds", "s"],
        env={"PATH": "/usr/bin:/bin", "HARBOR_SERVER_PYTHON": str(fake)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert record.read_text().split() == ["/staging/_hbexec.py", "--uds", "s"]
    assert f"using existing server python: {fake}" in result.stderr
