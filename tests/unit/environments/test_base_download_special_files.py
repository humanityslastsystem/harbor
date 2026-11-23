"""Workspace collection through ``download_dir_with_exclusions`` when the tree holds
special files.

The collector extracts the transfer archive with a filter that behaves like tarfile's
``data`` filter but skips FIFOs and device nodes instead of raising on them. These tests
drive the real ``download_dir_with_exclusions`` against a stub environment whose container
side is a local directory, so the archive is produced by the same ``tar`` command the
collector issues and unpacked by the same extraction call.
"""

from __future__ import annotations

import io
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from harbor.environments.base import (
    _ENV_TRANSFER_TAR_PATH,
    BaseEnvironment,
    ExecResult,
    _drop_special_files,
)
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig, TaskOS
from harbor.models.trial.paths import TrialPaths


class _LocalEnvironment(BaseEnvironment):
    """An environment whose container filesystem is the host filesystem.

    ``exec`` runs the collector's tar command with the local ``tar`` binary, with the
    in-container archive path rewritten to ``host_tar``. When ``archive`` is given the
    command is not run and those bytes become the archive instead, so a test can feed the
    collector a hand-built tar. ``download_file`` copies the archive to the requested path.
    """

    def __init__(self, *args, host_tar: Path, archive: bytes | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_tar = host_tar
        self.archive = archive

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    def _validate_definition(self):
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool):
        pass

    async def upload_file(self, source_path, target_path):
        pass

    async def upload_dir(self, source_dir, target_dir):
        pass

    async def download_file(self, source_path, target_path):
        assert source_path == _ENV_TRANSFER_TAR_PATH
        shutil.copyfile(self.host_tar, target_path)

    async def download_dir(self, source_dir, target_dir):
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if self.archive is not None:
            self.host_tar.write_bytes(self.archive)
            return ExecResult(return_code=0, stdout="", stderr="")
        argv = [
            str(self.host_tar) if tok == _ENV_TRANSFER_TAR_PATH else tok
            for tok in shlex.split(command)
        ]
        assert argv[0] == "tar"
        done = subprocess.run(argv, capture_output=True, text=True)
        return ExecResult(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )


def _environment(tmp_path: Path, archive: bytes | None = None) -> _LocalEnvironment:
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    return _LocalEnvironment(
        environment_dir=tmp_path,
        environment_name="test",
        session_id="session",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(os=TaskOS.LINUX),
        host_tar=tmp_path / "transfer.tar.gz",
        archive=archive,
    )


def _archive(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    """A gzipped tar of hand-built members, for shapes ``tar`` cannot produce from a tree."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def _regular(
    name: str, data: bytes = b"x", mode: int = 0o644
) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    return info, data


def _special(name: str, kind: bytes) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = kind
    if kind in (tarfile.CHRTYPE, tarfile.BLKTYPE):
        info.devmajor, info.devminor = 1, 3
    return info, None


def _symlink(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


def _relative_files(root: Path) -> list[str]:
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    )


@pytest.mark.skipif(shutil.which("tar") is None, reason="needs a local tar binary")
async def test_collection_keeps_the_tree_and_drops_a_fifo(tmp_path: Path) -> None:
    """A named pipe in a benchmark directory. Every regular file,
    directory and in-tree symlink must arrive intact, the pipe must not, and the
    exclude patterns must still apply."""
    ws = tmp_path / "ws"
    (ws / "bench" / "results").mkdir(parents=True)
    (ws / "server.py").write_text("print('hi')\n")
    (ws / "bench" / "load_test.py").write_text("x = 1\n")
    (ws / "bench" / "results" / "run1.json").write_text("{}\n")
    (ws / "scratch.pyc").write_bytes(b"\x00")
    os.mkfifo(ws / "bench" / "fifo")
    os.symlink("server.py", ws / "latest")
    os.symlink("../server.py", ws / "bench" / "link_up")

    env = _environment(tmp_path)
    out = tmp_path / "out"
    await env.download_dir_with_exclusions(
        source_dir=str(ws), target_dir=out, exclude=["*.pyc"]
    )

    assert _relative_files(out) == [
        "bench/load_test.py",
        "bench/results/run1.json",
        "server.py",
    ]
    assert (out / "server.py").read_text() == "print('hi')\n"
    assert (out / "bench" / "load_test.py").read_text() == "x = 1\n"
    assert (out / "bench" / "results").is_dir()
    assert os.readlink(out / "latest") == "server.py"
    assert os.readlink(out / "bench" / "link_up") == "../server.py"
    assert (out / "bench" / "link_up").read_text() == "print('hi')\n"
    assert not (out / "bench" / "fifo").exists()
    assert not (out / "scratch.pyc").exists()

    # The same archive is refused outright by the stock filter.
    with pytest.raises(tarfile.SpecialFileError):
        with tarfile.open(env.host_tar) as tf:
            tf.extractall(path=tmp_path / "strict", filter="data")


@pytest.mark.parametrize(
    "kind",
    [tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE],
    ids=["fifo", "char-device", "block-device"],
)
async def test_every_special_file_type_is_dropped(tmp_path: Path, kind: bytes) -> None:
    """Device nodes are refused by the ``data`` filter the same way a pipe is, and the
    collector must drop each of them rather than lose the archive."""
    members = [
        _regular("./a.txt", b"a"),
        _special("./dev/node", kind),
        _regular("./b/c.txt", b"c"),
    ]
    env = _environment(tmp_path, archive=_archive(members))
    out = tmp_path / "out"
    await env.download_dir_with_exclusions(source_dir="/ws", target_dir=out, exclude=[])

    assert _relative_files(out) == ["a.txt", "b/c.txt"]
    assert not (out / "dev" / "node").exists()
    with pytest.raises(tarfile.SpecialFileError):
        with tarfile.open(fileobj=io.BytesIO(env.archive)) as tf:
            tf.extractall(path=tmp_path / "strict", filter="data")


@pytest.mark.parametrize(
    ("member", "error"),
    [
        (_regular("../escaped.txt"), tarfile.OutsideDestinationError),
        (_symlink("./ln", "/etc/passwd"), tarfile.AbsoluteLinkError),
        (_symlink("./ln", "../outside"), tarfile.LinkOutsideDestinationError),
    ],
    ids=["parent-traversal", "absolute-link", "link-outside"],
)
async def test_unsafe_members_are_still_refused(tmp_path: Path, member, error) -> None:
    """Skipping special files must not loosen the rest of the ``data`` filter: a member
    that would land or point outside the destination is refused, and by the same error
    the stock filter raises."""
    archive = _archive([_regular("./a.txt"), member])
    env = _environment(tmp_path, archive=archive)
    with pytest.raises(error):
        await env.download_dir_with_exclusions(
            source_dir="/ws", target_dir=tmp_path / "out", exclude=[]
        )
    with pytest.raises(error):
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(path=tmp_path / "strict", filter="data")


async def test_path_and_mode_sanitising_matches_the_data_filter(tmp_path: Path) -> None:
    """The ``data`` filter does not refuse an absolute member name; it strips the leading
    slash so the file lands inside the destination. It also clears setuid and setgid bits.
    Both must survive the wrapping filter unchanged."""
    archive = _archive(
        [_regular("/abs/evil.txt", b"e"), _regular("./suid", b"s", mode=0o4755)]
    )
    for label, flt in (("data", "data"), ("collector", _drop_special_files)):
        dest = tmp_path / label
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(path=dest, filter=flt)
        assert _relative_files(dest) == ["abs/evil.txt", "suid"]
        assert not Path("/abs/evil.txt").exists()
        assert (dest / "suid").stat().st_mode & 0o7777 == 0o755

    env = _environment(tmp_path, archive=archive)
    out = tmp_path / "out"
    await env.download_dir_with_exclusions(source_dir="/ws", target_dir=out, exclude=[])
    assert _relative_files(out) == ["abs/evil.txt", "suid"]
