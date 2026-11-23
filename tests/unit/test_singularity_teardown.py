"""Sandbox teardown: the launch process group and the container's PID namespace.

The launch process leads its own session, so the network seal, the apptainer runtime
parent and the fuse helpers share one process group; the container's own init owns
everything inside it, including processes the payload detached with ``setsid``. These
tests drive both halves with real processes -- the group half with an ordinary shell
tree, the namespace half with a stand-in that calls ``os.unshare`` where the host allows
unprivileged user and PID namespaces. No Apptainer and no container image are involved.
"""

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from harbor.environments.launch import LaunchSpec
from harbor.environments.singularity.singularity import (
    SingularityEnvironment,
    _fd_pid_namespace,
    _proc_cmdline,
    _proc_descendants,
    _proc_pid_namespace,
    _proc_stat,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="teardown reads /proc"
)

# A stand-in for the sandbox: the launch process makes a user and PID namespace and forks
# its init, which adopts a process that has left the launch process group, exactly as the
# agent's setsid'd server does. It maps no uids, so it also runs where AppArmor refuses
# the ``unshare`` tool's uid_map write.
_STAND_IN = """
import os
os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWPID)
pid = os.fork()
if pid == 0:
    os.execvp("bash", ["bash", "-c", "setsid sleep 300; sleep 300"])
os.waitpid(pid, 0)
"""


def _unprivileged_pid_namespaces() -> bool:
    """Whether we may make a user and PID namespace, as the container backend does."""
    if not hasattr(os, "unshare"):
        return False
    probe = (
        "import os\n"
        "os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWPID)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
    )
    try:
        done = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


_HAS_PID_NAMESPACES = _unprivileged_pid_namespaces()

needs_pid_namespaces = pytest.mark.skipif(
    not _HAS_PID_NAMESPACES, reason="host refuses unprivileged user+PID namespaces"
)


def _env(tmp_path: Path) -> SingularityEnvironment:
    sif = tmp_path / "img.sif"
    sif.write_bytes(b"")
    return SingularityEnvironment.launch(LaunchSpec(image=str(sif), gpus=0))


def _group_members(pgid: int) -> list[int]:
    """Live (non-zombie) pids in a process group."""
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        stat = _proc_stat(int(entry))
        if stat is not None and stat[0] != "Z" and stat[1] == pgid:
            members.append(int(entry))
    return members


def _namespace_members(namespace: tuple[int, int]) -> list[int]:
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        stat = _proc_stat(int(entry))
        if stat is not None and stat[0] != "Z":
            if _proc_pid_namespace(int(entry)) == namespace:
                members.append(int(entry))
    return members


async def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


async def _spawn(*argv: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )


async def _spawn_stand_in() -> asyncio.subprocess.Process:
    """A stand-in sandbox in its own session, returned once its init's child runs."""
    proc = await _spawn(sys.executable, "-c", _STAND_IN)
    assert await _wait_until(lambda: len(_proc_descendants(proc.pid)) >= 3)
    return proc


def _stand_in_init(proc: asyncio.subprocess.Process) -> int:
    """The init of the PID namespace a running stand-in made."""
    own = _proc_pid_namespace(os.getpid())
    for pid in _proc_descendants(proc.pid):
        ns = _proc_pid_namespace(pid)
        if ns is not None and ns != own:
            return pid
    raise AssertionError(f"stand-in {proc.pid} has no PID namespace of its own")


def _stand_in_namespace(proc: asyncio.subprocess.Process) -> tuple[int, int]:
    """The PID namespace a running stand-in made."""
    namespace = _proc_pid_namespace(_stand_in_init(proc))
    assert namespace is not None
    return namespace


def _stand_in_user_inode(proc: asyncio.subprocess.Process) -> int:
    """The inode number of the user namespace a running stand-in made."""
    return os.stat(f"/proc/{_stand_in_init(proc)}/ns/user").st_ino


def _hold_namespace(path_or_fd: str | int) -> int:
    """An fd of our own on a namespace a test inspects after teardown.

    Teardown closes its fd, and a freed namespace number can go to a namespace anything
    else on the host creates next, so checking a dead namespace's members by number
    alone could find a stranger's processes.
    """
    if isinstance(path_or_fd, int):
        return os.dup(path_or_fd)
    return os.open(path_or_fd, os.O_RDONLY)


async def _kill_stand_in(proc: asyncio.subprocess.Process) -> None:
    for pid in _proc_descendants(proc.pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await proc.wait()


def test_proc_helpers_describe_this_process():
    state, pgid = _proc_stat(os.getpid())
    assert state in "RS"
    assert pgid == os.getpgid(0)
    namespace = _proc_pid_namespace(os.getpid())
    # The identity carries the inode number the kernel shows in the link, and an fd on
    # the namespace names the same one.
    assert os.readlink("/proc/self/ns/pid") == f"pid:[{namespace[1]}]"
    fd = os.open("/proc/self/ns/pid", os.O_RDONLY)
    try:
        assert _fd_pid_namespace(fd) == namespace
    finally:
        os.close(fd)
    assert _proc_cmdline(os.getpid()) != ""


def test_proc_helpers_tolerate_a_missing_process():
    # /proc/-1 cannot exist, so this stands in for a pid that has already been reaped.
    assert _proc_stat(-1) is None
    assert _proc_pid_namespace(-1) is None
    assert _proc_cmdline(-1) == ""


async def test_proc_descendants_finds_a_child():
    proc = await _spawn("bash", "-c", "sleep 30 & wait")
    try:
        assert await _wait_until(lambda: len(_proc_descendants(proc.pid)) >= 2)
        assert _proc_descendants(proc.pid)[0] == proc.pid
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()


async def test_terminate_sandbox_clears_the_launch_process_group(tmp_path):
    env = _env(tmp_path)
    proc = await _spawn("bash", "-c", "sleep 300 & wait")
    env._server_process = proc
    env._sandbox_pgid = proc.pid
    assert await _wait_until(lambda: len(_group_members(proc.pid)) >= 2)

    await env._terminate_sandbox()

    assert _group_members(proc.pid) == []
    assert env._sandbox_pgid is None


async def test_terminate_sandbox_kills_a_group_the_launch_process_left_behind(tmp_path):
    # The launch process is gone, its children are not, and they do not answer SIGTERM.
    # Only the group SIGKILL reaches them.
    env = _env(tmp_path)
    proc = await _spawn("bash", "-c", "trap '' TERM; sleep 300 & exit 0")
    env._server_process = proc
    env._sandbox_pgid = proc.pid
    await proc.wait()
    assert await _wait_until(lambda: len(_group_members(proc.pid)) == 1)

    await env._terminate_sandbox()

    assert _group_members(proc.pid) == []


async def test_terminate_sandbox_spares_a_reused_container_init_pid(tmp_path):
    # Pids wrap around during a long build, so the recorded init pid is only signalled
    # while it still leads the namespace it was recorded for. Any namespace the
    # bystander is not in can stand for the container's; our network namespace is one
    # we can always open.
    env = _env(tmp_path)
    bystander = await _spawn("sleep", "300")
    proc = await _spawn("bash", "-c", "sleep 300 & wait")
    env._server_process = proc
    env._sandbox_pgid = proc.pid
    env._container_init_pid = bystander.pid
    env._container_pid_ns_fd = os.open("/proc/self/ns/net", os.O_RDONLY)
    try:
        assert await _wait_until(lambda: len(_group_members(proc.pid)) >= 2)

        await env._terminate_sandbox()

        assert _group_members(proc.pid) == []
        assert _proc_stat(bystander.pid) is not None
    finally:
        bystander.kill()
        await bystander.wait()


async def test_terminate_sandbox_releases_the_namespace_fd_when_it_raises(tmp_path):
    env = _env(tmp_path)
    proc = await _spawn("bash", "-c", "sleep 300 & wait")
    env._server_process = proc
    env._sandbox_pgid = proc.pid
    fd = os.open("/proc/self/ns/net", os.O_RDONLY)
    env._container_init_pid = proc.pid
    env._container_pid_ns_fd = fd

    async def broken_sweep() -> list[int]:
        raise RuntimeError("sweep failed")

    env._sweep_sandbox = broken_sweep
    try:
        with pytest.raises(RuntimeError, match="sweep failed"):
            await env._terminate_sandbox()

        assert env._container_pid_ns_fd is None
        assert env._container_init_pid is None
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


async def test_record_container_init_warns_without_a_namespace(tmp_path, caplog):
    env = _env(tmp_path)
    proc = await _spawn("bash", "-c", "sleep 300 & wait")
    env._server_process = proc
    env._sandbox_pgid = proc.pid
    try:
        assert await _wait_until(lambda: len(_proc_descendants(proc.pid)) >= 2)
        env._record_container_init()
        assert env._container_init_pid is None
        assert env._container_pid_ns_fd is None
        assert "PID-namespace init" in caplog.text
    finally:
        await env._terminate_sandbox()


@needs_pid_namespaces
async def test_terminate_sandbox_kills_the_container_pid_namespace(tmp_path):
    env = _env(tmp_path)
    proc = await _spawn_stand_in()
    env._server_process = proc
    env._sandbox_pgid = proc.pid

    env._record_container_init()
    assert env._container_init_pid is not None
    assert env._container_pid_ns_fd is not None
    held = _hold_namespace(env._container_pid_ns_fd)
    try:
        namespace = _fd_pid_namespace(held)
        assert namespace == _stand_in_namespace(proc)
        assert namespace != _proc_pid_namespace(os.getpid())
        assert len(_namespace_members(namespace)) >= 2

        await env._terminate_sandbox()

        assert _namespace_members(namespace) == []
        assert _group_members(proc.pid) == []
        assert env._container_pid_ns_fd is None
    finally:
        os.close(held)


@needs_pid_namespaces
async def test_a_dead_sandbox_does_not_lend_its_namespace_to_the_next(tmp_path):
    # A sandbox that dies on its own frees its PID namespace, and the kernel can give
    # the freed inode number to a later namespace, often the very next one. Without the
    # fd pin, a later sandbox in this test takes the dead one's number about half the
    # time, and the dead one's teardown then kills it.
    #
    # User and PID namespaces draw inode numbers from one pool, and each stand-in makes
    # both, so the freed number can go to a stand-in's user namespace instead. That one
    # is no danger to the sweep, which matches only PID namespaces, but it uses the
    # number up, and the PID-namespace check below then sees nothing. A user namespace
    # holding the number is therefore also a failure: it shows the pin was not holding
    # the number, and it is what makes this test catch a lost pin every time.
    # test_the_namespace_fd_is_still_open_when_the_sweep_runs checks the pin directly.
    env = _env(tmp_path)
    dead = await _spawn_stand_in()
    env._server_process = dead
    env._sandbox_pgid = dead.pid
    env._record_container_init()
    assert env._container_pid_ns_fd is not None
    namespace = _fd_pid_namespace(env._container_pid_ns_fd)

    os.kill(env._container_init_pid, signal.SIGKILL)
    await dead.wait()
    assert await _wait_until(lambda: _namespace_members(namespace) == [])

    later: list[asyncio.subprocess.Process] = []
    try:
        for _ in range(30):
            later.append(await _spawn_stand_in())
            assert _stand_in_namespace(later[-1]) != namespace
            assert _stand_in_user_inode(later[-1]) != namespace[1]
        members = [pid for proc in later for pid in _proc_descendants(proc.pid)]

        await env._terminate_sandbox()

        assert all(_proc_stat(pid) is not None for pid in members)
    finally:
        for proc in later:
            await _kill_stand_in(proc)


@needs_pid_namespaces
async def test_the_namespace_fd_is_still_open_when_the_sweep_runs(tmp_path):
    # The pin only protects the sweep while it is open and still names the sandbox's
    # namespace. This checks that directly, so it does not depend on the kernel handing
    # a freed number to a later namespace.
    env = _env(tmp_path)
    proc = await _spawn_stand_in()
    env._server_process = proc
    env._sandbox_pgid = proc.pid
    try:
        env._record_container_init()
        assert env._container_init_pid is not None
        assert env._container_pid_ns_fd is not None
        recorded = _proc_pid_namespace(env._container_init_pid)
        assert recorded is not None and recorded == _stand_in_namespace(proc)

        real_sweep = env._sweep_sandbox
        seen: list[tuple[int, int]] = []

        async def sweep() -> list[int]:
            assert env._container_pid_ns_fd is not None
            st = os.fstat(env._container_pid_ns_fd)
            seen.append((st.st_dev, st.st_ino))
            return await real_sweep()

        env._sweep_sandbox = sweep

        await env._terminate_sandbox()

        assert seen == [recorded]
    finally:
        await _kill_stand_in(proc)


@needs_pid_namespaces
async def test_sweep_matches_the_held_namespace_not_its_number(tmp_path, monkeypatch):
    # The shape a reused number takes: another sandbox's namespace link reads as the
    # same pid:[N] as ours. Only processes in the namespace we hold are ours.
    env = _env(tmp_path)
    ours = await _spawn_stand_in()
    other = await _spawn_stand_in()
    held = None
    try:
        env._server_process = ours
        env._sandbox_pgid = ours.pid
        env._record_container_init()
        assert env._container_pid_ns_fd is not None
        held = _hold_namespace(env._container_pid_ns_fd)
        namespace = _fd_pid_namespace(held)
        theirs = [pid for pid in _proc_descendants(other.pid) if pid != other.pid]
        assert theirs and _stand_in_namespace(other) != namespace

        real_readlink = os.readlink
        reused = {f"/proc/{pid}/ns/pid" for pid in theirs}

        def readlink(path, *args, **kwargs):
            if str(path) in reused:
                return f"pid:[{namespace[1]}]"
            return real_readlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "readlink", readlink)

        await env._terminate_sandbox()

        assert _namespace_members(namespace) == []
        assert all(_proc_stat(pid) is not None for pid in theirs)
    finally:
        monkeypatch.undo()
        if held is not None:
            os.close(held)
        await _kill_stand_in(other)
        await _kill_stand_in(ours)


@needs_pid_namespaces
async def test_an_unpinnable_namespace_is_left_out_of_the_sweep(
    tmp_path, caplog, monkeypatch
):
    # If the init is gone before its namespace can be opened, nothing names that
    # namespace any more, so teardown falls back to the launch process group.
    env = _env(tmp_path)
    proc = await _spawn_stand_in()
    env._server_process = proc
    env._sandbox_pgid = proc.pid
    held = _hold_namespace(f"/proc/{_stand_in_init(proc)}/ns/pid")
    try:
        namespace = _fd_pid_namespace(held)
        real_open = os.open

        def open_(path, *args, **kwargs):
            if str(path).endswith("/ns/pid"):
                raise FileNotFoundError(path)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_)
        env._record_container_init()
        monkeypatch.undo()

        assert env._container_init_pid is None
        assert env._container_pid_ns_fd is None
        assert "already gone" in caplog.text

        await env._terminate_sandbox()

        # The init is in the launch group, so the group kill still takes the namespace
        # down.
        assert _group_members(proc.pid) == []
        assert _namespace_members(namespace) == []
    finally:
        os.close(held)


async def test_stop_is_safe_before_the_sandbox_starts(tmp_path):
    await _env(tmp_path).stop(delete=False)


def test_launch_command_keeps_the_pid_namespace(tmp_path):
    # Teardown's strongest handle is the container's own PID namespace, which only
    # exists because the launch command asks for one.
    env = _env(tmp_path)
    cmd = env._build_singularity_command(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    assert "--pid" in cmd
