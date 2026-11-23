"""The working directory the exec server's launch runs in, and the command it spawns.

Apptainer's starter opens its working directory before it starts the container. Under a
user namespace that maps only root (a root launch without CAP_SYS_ADMIN, as in a Docker
container), a directory owned by another uid cannot be opened, and the start dies with
"Failed to open current working directory". ``LaunchSpec.start_in_staging_dir`` runs
the launch from the staging directory, which the launching process created. Without it
the launch inherits the caller's directory.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from harbor.environments.launch import LaunchSpec
from harbor.environments.singularity import singularity as sg
from harbor.environments.singularity.singularity import SingularityEnvironment


class _Stdout:
    """A server that says nothing and reaches EOF, so the output reader finishes."""

    async def readline(self):
        return b""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _Proc:
    """A launch process that stays alive; its pid is this test's own, a harmless target."""

    def __init__(self):
        self.returncode = None
        self.pid = os.getpid()
        self.stdout = _Stdout()

    async def wait(self):
        await asyncio.sleep(5)


class _Ready:
    status_code = 200


class _Client:
    """A control channel that answers at once."""

    async def get(self, url):
        return _Ready()

    async def aclose(self):
        pass


def _sif(tmp_path: Path) -> str:
    sif = tmp_path / "img.sif"
    sif.write_bytes(b"")
    return str(sif)


async def _start(env, monkeypatch) -> tuple[list, dict]:
    """Start ``env`` against a stub launch; return the argv and spawn kwargs."""
    launches: list[tuple[tuple, dict]] = []

    async def _spawn(*args, **kwargs):
        launches.append((args, kwargs))
        return _Proc()

    async def _no_teardown(self):
        """No container was started, and the real one signals a process group by pid."""

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(sg.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(
        SingularityEnvironment, "_record_container_init", lambda self: None
    )
    monkeypatch.setattr(SingularityEnvironment, "_terminate_sandbox", _no_teardown)
    await env.start(force_build=False)
    assert len(launches) == 1
    return list(launches[0][0]), launches[0][1]


@pytest.fixture
def caller_dir(tmp_path, monkeypatch) -> Path:
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    return caller


async def test_the_launch_can_run_from_the_staging_directory(
    tmp_path, monkeypatch, caller_dir
):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), workdir="/app", start_in_staging_dir=True)
    )
    try:
        _, kwargs = await _start(env, monkeypatch)
        cwd = Path(kwargs["cwd"])
        assert cwd == env._staging_dir
        assert cwd != caller_dir
        assert cwd.is_dir() and cwd.stat().st_uid == os.getuid()
    finally:
        if env._staging_dir is not None:
            shutil.rmtree(env._staging_dir, ignore_errors=True)


async def test_by_default_the_launch_inherits_the_callers_directory(
    tmp_path, monkeypatch, caller_dir
):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), workdir="/app")
    )
    try:
        argv, kwargs = await _start(env, monkeypatch)
        # None is subprocess's own default: the child inherits the caller's directory.
        assert kwargs.get("cwd") is None
        # The spawned command is the default one, with the binds the start adds.
        staging = str(env._staging_dir)
        binds = ["-B", f"{staging}:/staging"]
        for netfile in ("/etc/resolv.conf", "/etc/hosts"):
            if os.path.exists(netfile):
                binds += ["-B", f"{netfile}:{netfile}"]
        port = argv[argv.index("--port") + 1]
        assert argv == [
            "singularity",
            "exec",
            "--no-mount",
            "home",
            "--no-mount",
            "tmp",
            "--no-mount",
            "bind-paths",
            "--pwd",
            "/app",
            "--writable-tmpfs",
            "--containall",
            "--pid",
            *binds,
            _sif(tmp_path),
            "bash",
            "-c",
            'exec /staging/bootstrap.sh "$@"',
            "bash",
            "/app",
            "/staging/_hbexec.py",
            "--port",
            port,
            "--workdir",
            "/app",
        ]
    finally:
        if env._staging_dir is not None:
            shutil.rmtree(env._staging_dir, ignore_errors=True)


async def test_an_isolated_launch_listens_on_the_staging_socket(
    tmp_path, monkeypatch, caller_dir
):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), workdir="/app", allow_internet=False)
    )
    try:
        argv, _ = await _start(env, monkeypatch)
        assert argv[argv.index("--uds") + 1] == "/staging/control.sock"
        assert "--port" not in argv
        assert env._control_base == "http://localhost"
    finally:
        if env._staging_dir is not None:
            shutil.rmtree(env._staging_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "field",
    [
        {"image": "img.sif"},
        {"mounts": [{"type": "bind", "source": "data", "target": "/data"}]},
        {"overlay_dir": "overlays/s0"},
        {"lower_overlay_dirs": ("prefixes/a.state",)},
    ],
)
def test_a_relative_host_path_is_refused_when_starting_in_staging(tmp_path, field):
    # From the staging directory a relative path would name a different file.
    spec: dict[str, Any] = {"image": _sif(tmp_path), **field}
    with pytest.raises(ValueError, match="start_in_staging_dir"):
        SingularityEnvironment.launch(LaunchSpec(**spec, start_in_staging_dir=True))
