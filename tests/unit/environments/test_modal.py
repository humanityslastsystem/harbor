"""Unit tests for ModalEnvironment resource configuration."""

import asyncio
import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("modal")

import harbor.environments.modal as modal_mod
from harbor.environments.base import ExecResult
from harbor.environments.modal import (
    _DIR_LIST_TIMEOUT_SECS,
    _MODAL_DEFAULT_CPU_REQUEST_CORES,
    _MODAL_DEFAULT_MEMORY_REQUEST_MB,
    ModalEnvironment,
    _ModalDinD,
)
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode, ServiceVolumeConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths


def _make_env(
    temp_dir: Path,
    *,
    compose: bool = False,
    cpus: int | None = 2,
    memory_mb: int | None = 4096,
    cpu_mode: ResourceMode = ResourceMode.AUTO,
    memory_mode: ResourceMode = ResourceMode.AUTO,
    gpus: int = 0,
    gpu_types: list[str] | None = None,
    task_env: dict[str, str] | None = None,
    persistent_env: dict[str, str] | None = None,
    mounts: list[ServiceVolumeConfig] | None = None,
    extra_docker_compose: list[Path] | None = None,
) -> ModalEnvironment:
    env_dir = temp_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    if compose:
        (env_dir / "docker-compose.yaml").write_text(
            "services:\n  main:\n    environment:\n      - CPUS=${CPUS:-2}\n"
        )
    else:
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    trial_dir = temp_dir / "trial"
    trial_dir.mkdir(exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    extra: dict = {}
    if persistent_env is not None:
        extra["persistent_env"] = persistent_env
    if mounts is not None:
        extra["mounts"] = mounts
    if extra_docker_compose is not None:
        extra["extra_docker_compose"] = extra_docker_compose

    return ModalEnvironment(
        environment_dir=env_dir,
        environment_name="test-task",
        session_id="Test.Session.123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            cpus=cpus,
            memory_mb=memory_mb,
            gpus=gpus,
            gpu_types=gpu_types or [],
            env=task_env or {},
        ),
        cpu_enforcement_policy=cpu_mode,
        memory_enforcement_policy=memory_mode,
        **extra,
    )


class TestCapabilities:
    def test_modal_supports_limits_and_requests(self, temp_dir):
        caps = type(_make_env(temp_dir)).resource_capabilities()
        assert caps is not None
        assert caps.cpu_limit is True
        assert caps.cpu_request is True
        assert caps.memory_limit is True
        assert caps.memory_request is True


class TestCpuConfig:
    def test_returns_tuple_with_equal_request_and_limit(self, temp_dir):
        env = _make_env(temp_dir, cpus=4)
        assert env._cpu_config() == (4, 4)

    def test_default_single_cpu(self, temp_dir):
        env = _make_env(temp_dir, cpus=1)
        assert env._cpu_config() == (1, 1)

    def test_omitted_cpu_uses_modal_default(self, temp_dir):
        env = _make_env(temp_dir, cpus=None)
        assert env._cpu_config() is None

    def test_request_mode_returns_scalar(self, temp_dir):
        env = _make_env(temp_dir, cpus=4, cpu_mode=ResourceMode.REQUEST)
        assert env._cpu_config() == 4

    def test_limit_mode_sets_minimum_request_and_limit(self, temp_dir):
        env = _make_env(temp_dir, cpus=4, cpu_mode=ResourceMode.LIMIT)
        assert env._cpu_config() == (_MODAL_DEFAULT_CPU_REQUEST_CORES, 4)


class TestMemoryConfig:
    def test_auto_mode_returns_scalar_request(self, temp_dir):
        env = _make_env(temp_dir, memory_mb=4096)
        assert env._memory_config() == 4096

    def test_omitted_memory_uses_modal_default(self, temp_dir):
        env = _make_env(temp_dir, memory_mb=None)
        assert env._memory_config() is None

    def test_limit_mode_sets_minimum_request_and_limit(self, temp_dir):
        env = _make_env(temp_dir, memory_mb=4096, memory_mode=ResourceMode.LIMIT)
        assert env._memory_config() == (_MODAL_DEFAULT_MEMORY_REQUEST_MB, 4096)

    def test_guarantee_mode_sets_equal_request_and_limit(self, temp_dir):
        env = _make_env(temp_dir, memory_mb=4096, memory_mode=ResourceMode.GUARANTEE)
        assert env._memory_config() == (4096, 4096)


class TestGpuConfig:
    def test_no_gpus_returns_none(self, temp_dir):
        env = _make_env(temp_dir, gpus=0)
        assert env._gpu_config() is None

    def test_any_type(self, temp_dir):
        env = _make_env(temp_dir, gpus=1, gpu_types=None)
        assert env._gpu_config() == "any:1"

    def test_specific_type(self, temp_dir):
        env = _make_env(temp_dir, gpus=1, gpu_types=["H100"])
        assert env._gpu_config() == "H100:1"

    def test_multi_gpu_count_is_preserved(self, temp_dir):
        env = _make_env(temp_dir, gpus=4, gpu_types=["A100-80GB"])
        assert env._gpu_config() == "A100-80GB:4"

    def test_first_type_wins_when_multiple_specified(self, temp_dir):
        env = _make_env(temp_dir, gpus=1, gpu_types=["H100", "A100"])
        assert env._gpu_config() == "H100:1"


class TestComposeDetection:
    def test_extra_compose_enables_compose_mode(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        env = _make_env(temp_dir, compose=False, extra_docker_compose=[extra])
        assert env._compose_mode is True
        assert isinstance(env._strategy, _ModalDinD)


def _dind(env: ModalEnvironment) -> _ModalDinD:
    strategy = env._strategy
    assert isinstance(strategy, _ModalDinD)
    return strategy


class TestDinDComposeEnvVars:
    def test_contains_required_keys(self, temp_dir):
        dind = _dind(_make_env(temp_dir, compose=True))
        env_vars = dind._compose_env_vars()
        required = {
            "CONTEXT_DIR",
            "MAIN_IMAGE_NAME",
            "CPUS",
            "MEMORY",
        }
        assert required <= set(env_vars.keys())

    def test_legacy_path_keys_are_self_bound(self, temp_dir):
        dind = _dind(
            _make_env(
                temp_dir,
                compose=True,
                mounts=[
                    {
                        "type": "bind",
                        "source": "/host/verifier",
                        "target": str(EnvironmentPaths.verifier_dir),
                    },
                    {
                        "type": "bind",
                        "source": "/host/agent",
                        "target": str(EnvironmentPaths.agent_dir),
                    },
                    {
                        "type": "bind",
                        "source": "/host/artifacts",
                        "target": str(EnvironmentPaths.artifacts_dir),
                    },
                ],
            )
        )
        env_vars = dind._compose_env_vars()
        assert env_vars["HOST_VERIFIER_LOGS_PATH"] == str(EnvironmentPaths.verifier_dir)
        assert env_vars["ENV_VERIFIER_LOGS_PATH"] == str(EnvironmentPaths.verifier_dir)
        assert env_vars["HOST_AGENT_LOGS_PATH"] == str(EnvironmentPaths.agent_dir)
        assert env_vars["ENV_AGENT_LOGS_PATH"] == str(EnvironmentPaths.agent_dir)
        assert env_vars["HOST_ARTIFACTS_PATH"] == str(EnvironmentPaths.artifacts_dir)
        assert env_vars["ENV_ARTIFACTS_PATH"] == str(EnvironmentPaths.artifacts_dir)

    def test_infra_vars_win_over_referenced_task_and_persistent_env(
        self, temp_dir, monkeypatch, caplog
    ):
        monkeypatch.setenv("CPUS", "999")
        env = _make_env(
            temp_dir,
            compose=True,
            task_env={"MEMORY": "1G", "CONTEXT_DIR": "/wrong"},
            persistent_env={"MAIN_IMAGE_NAME": "wrong-image"},
        )
        dind = _dind(env)

        with caplog.at_level(logging.WARNING):
            env_vars = dind._compose_env_vars()

        assert env_vars["CPUS"] == "2"
        assert env_vars["MEMORY"] == "4096M"
        assert env_vars["CONTEXT_DIR"] == "/harbor/environment"
        assert env_vars["MAIN_IMAGE_NAME"] == "hb__test-task"
        assert any("CPUS" in rec.message for rec in caplog.records)


class TestDinDComposeMounts:
    def test_host_network_overlay_preserves_build_from_base_compose(self, temp_dir):
        env_dir = temp_dir / "environment"
        env_dir.mkdir()
        (env_dir / "docker-compose.yaml").write_text(
            "services:\n"
            "  sidecar:\n"
            "    build: ./sidecar\n"
            "  redis:\n"
            "    image: redis:7\n"
        )
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    environment:\n      FOO: bar\n")

        overlay = yaml.safe_load(
            _ModalDinD._build_host_network_overlay(env_dir, extra_compose_paths=[extra])
        )

        assert overlay["services"]["sidecar"]["build"]["network"] == "host"
        assert "build" not in overlay["services"]["redis"]

    def test_mounts_compose_file_included(self, temp_dir):
        dind = _dind(_make_env(temp_dir, compose=True))
        flags = dind._compose_file_flags()
        paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        assert any(path.endswith("docker-compose-mounts.json") for path in paths)

    def test_extra_compose_positioned_after_task_compose(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        dind = _dind(_make_env(temp_dir, compose=True, extra_docker_compose=[extra]))
        flags = dind._compose_file_flags()
        paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        env_idx = next(
            i
            for i, path in enumerate(paths)
            if path.endswith("/harbor/environment/docker-compose.yaml")
        )
        extra_idx = next(
            i
            for i, path in enumerate(paths)
            if path.endswith("docker-compose-extra-0.yaml")
        )
        mounts_idx = next(
            i
            for i, path in enumerate(paths)
            if path.endswith("docker-compose-mounts.json")
        )
        assert mounts_idx < env_idx < extra_idx

    def test_extra_compose_positioned_after_mounts_without_task_compose(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        dind = _dind(_make_env(temp_dir, compose=False, extra_docker_compose=[extra]))
        flags = dind._compose_file_flags()
        paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        extra_idx = next(
            i
            for i, path in enumerate(paths)
            if path.endswith("docker-compose-extra-0.yaml")
        )
        mounts_idx = next(
            i
            for i, path in enumerate(paths)
            if path.endswith("docker-compose-mounts.json")
        )
        assert mounts_idx < extra_idx

    async def test_writes_json_locally_and_uploads_to_vm(self, temp_dir):
        mounts: list[ServiceVolumeConfig] = [
            {
                "type": "bind",
                "source": "/discarded",
                "target": str(EnvironmentPaths.verifier_dir),
            }
        ]
        env = _make_env(temp_dir, compose=True, mounts=mounts)
        dind = _dind(env)
        uploaded: list[tuple[str, str, dict]] = []

        async def _fake_upload(source, target):
            source = Path(source)
            assert source.name == "docker-compose-mounts.json"
            assert source.parent != env.trial_paths.trial_dir
            uploaded.append((str(source), target, json.loads(source.read_text())))

        env._sdk_upload_file = _fake_upload  # type: ignore[method-assign]

        volumes = dind._resolve_volumes()
        await dind._stage_mounts_compose_file(volumes)

        source, target, body = uploaded[0]
        assert not Path(source).exists()
        assert not list(env.trial_paths.trial_dir.glob("*docker-compose-mounts.json"))
        assert body["services"]["main"]["volumes"] == cast(list, volumes)
        assert target == "/harbor/compose/docker-compose-mounts.json"


def test_modal_image_digest_reads_object_id():
    self = Mock()
    self._image = Mock(object_id="im-xyz")
    assert asyncio.run(ModalEnvironment.image_digest(self)) == "im-xyz"


def test_modal_image_digest_none_when_image_unset():
    self = Mock()
    self._image = None
    assert asyncio.run(ModalEnvironment.image_digest(self)) is None


def test_modal_image_digest_none_when_not_hydrated():
    class _Unhydrated:
        @property
        def object_id(self):
            raise RuntimeError("not hydrated")

    self = Mock()
    self._image = _Unhydrated()
    assert asyncio.run(ModalEnvironment.image_digest(self)) is None


# --- Fake Modal sandbox for _sdk_upload_dir tests -------------------------
# Mimics the ``.aio(...)`` async-call convention of Modal's SDK: methods and
# process streams expose an ``aio`` coroutine.  Just enough surface for
# _sdk_exec (sandbox.exec + process.stdout/stderr/wait) and _sdk_upload_file
# (sandbox.filesystem.copy_from_local).


class _FakeMethod:
    """Wraps a sync fn so it is callable via ``.aio(...)`` like Modal's SDK."""

    def __init__(self, fn):
        self._fn = fn

    async def _acall(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    @property
    def aio(self):
        return self._acall


class _FakeStream:
    def __init__(self, data: str):
        self.read = _FakeMethod(lambda: data)


class _FakeProcess:
    def __init__(self, *, stdout: str = "", stderr: str = "", return_code: int = 0):
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.wait = _FakeMethod(lambda: return_code)


class _FakeFilesystem:
    def __init__(self):
        # (source, target, bytes); bytes are stashed at call time because the
        # caller deletes the host tar in its finally block.
        self.copies: list[tuple[str, str, bytes]] = []
        self.copy_from_local = _FakeMethod(self._copy)

    def _copy(self, source, target):
        self.copies.append((str(source), str(target), Path(source).read_bytes()))


class _FakeSandbox:
    def __init__(self, *, extract_rc: int = 0, extract_stderr: str = ""):
        self.filesystem = _FakeFilesystem()
        self.commands: list[str] = []
        self._extract_rc = extract_rc
        self._extract_stderr = extract_stderr
        self.exec = _FakeMethod(self._exec)

    def _exec(self, shell, flag, command, **kwargs):
        self.commands.append(command)
        if command.startswith("tar "):
            return _FakeProcess(
                stderr=self._extract_stderr, return_code=self._extract_rc
            )
        return _FakeProcess()


def _upload_env(temp_dir: Path, **sandbox_kwargs) -> ModalEnvironment:
    env = _make_env(temp_dir)
    env._sandbox = _FakeSandbox(**sandbox_kwargs)  # type: ignore[assignment]
    return env


def _tree_snapshot(root: Path) -> dict[str, object]:
    """Map each entry (relative to root) to a comparable representation."""
    snap: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            snap[rel] = ("symlink", os.readlink(path))
        elif path.is_dir():
            snap[rel] = ("dir", None)
        else:
            snap[rel] = ("file", path.read_bytes(), path.stat().st_mode & 0o777)
    return snap


class TestSdkUploadDir:
    async def test_single_copy_and_single_extract_regardless_of_count(self, temp_dir):
        src = temp_dir / "src"
        (src / "a" / "b").mkdir(parents=True)
        for i in range(25):
            (src / f"top_{i}.txt").write_text(f"top{i}")
        for i in range(25):
            (src / "a" / f"mid_{i}.txt").write_text(f"mid{i}")
        (src / "a" / "b" / "deep.txt").write_text("deep")

        env = _upload_env(temp_dir)
        sandbox = env._sandbox  # type: ignore[assignment]

        await env._sdk_upload_dir(src, "/harbor/dest")

        # Exactly one upload, and exactly one extract exec, plus the mkdir.
        assert len(sandbox.filesystem.copies) == 1
        assert len(sandbox.commands) == 2
        assert sandbox.commands[0] == "mkdir -p /harbor/dest"
        tar_cmds = [c for c in sandbox.commands if c.startswith("tar ")]
        assert len(tar_cmds) == 1
        assert " -C /harbor/dest " in tar_cmds[0]

        # Tarball uploaded to a unique /tmp path; the extract exec targets it.
        remote_tar = sandbox.filesystem.copies[0][1]
        assert remote_tar.startswith("/tmp/harbor-upload-")
        assert remote_tar.endswith(".tar")
        assert remote_tar in tar_cmds[0]

        # The host temp tar is removed in the finally block.
        assert not Path(sandbox.filesystem.copies[0][0]).exists()

    async def test_nested_tree_round_trips_through_real_tar(self, temp_dir):
        src = temp_dir / "src"
        (src / "pkg" / "sub").mkdir(parents=True)
        (src / "empty_dir").mkdir()
        (src / "root.txt").write_text("root file")
        (src / "pkg" / "mod.py").write_text("print('hi')\n")
        (src / "pkg" / "sub" / "data.bin").write_bytes(b"\x00\x01\x02\x03")
        script = src / "run.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o755)
        (src / "link.txt").symlink_to("root.txt")

        env = _upload_env(temp_dir)
        await env._sdk_upload_dir(src, "/harbor/dest")

        tar_bytes = env._sandbox.filesystem.copies[0][2]  # type: ignore[attr-defined]
        dest = temp_dir / "extracted"
        dest.mkdir()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            tar.extractall(dest, filter="fully_trusted")

        assert _tree_snapshot(dest) == _tree_snapshot(src)

    async def test_empty_dir_creates_target_and_succeeds(self, temp_dir):
        src = temp_dir / "empty_src"
        src.mkdir()

        env = _upload_env(temp_dir)
        sandbox = env._sandbox  # type: ignore[assignment]

        await env._sdk_upload_dir(src, "/harbor/dest")

        # Target is still created (mkdir) and an empty tar round-trips cleanly.
        assert sandbox.commands[0] == "mkdir -p /harbor/dest"
        assert len(sandbox.filesystem.copies) == 1
        dest = temp_dir / "extracted"
        dest.mkdir()
        tar_bytes = sandbox.filesystem.copies[0][2]
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            assert tar.getmembers() == []
            tar.extractall(dest, filter="fully_trusted")
        assert list(dest.iterdir()) == []

    async def test_missing_source_raises_file_not_found(self, temp_dir):
        env = _upload_env(temp_dir)
        with pytest.raises(FileNotFoundError):
            await env._sdk_upload_dir(temp_dir / "does_not_exist", "/harbor/dest")

    async def test_raises_when_sandbox_not_started(self, temp_dir):
        src = temp_dir / "src"
        src.mkdir()
        env = _make_env(temp_dir)
        env._sandbox = None  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="Sandbox not found"):
            await env._sdk_upload_dir(src, "/harbor/dest")

    async def test_extract_failure_raises_loudly(self, temp_dir):
        src = temp_dir / "src"
        src.mkdir()
        (src / "f.txt").write_text("x")

        env = _upload_env(temp_dir, extract_rc=2, extract_stderr="tar: corrupt archive")
        sandbox = env._sandbox  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="tar: corrupt archive"):
            await env._sdk_upload_dir(src, "/harbor/dest")

        # Even on failure the host temp tar is cleaned up.
        assert not Path(sandbox.filesystem.copies[0][0]).exists()


# --- Fakes for client-side download / exec timeout tests ------------------
# A stalled gRPC stream returns no data and never raises; these fakes model
# that with ``.aio(...)`` coroutines that never resolve, so the client-side
# deadline is the only thing that ends the await.


class _HangingMethod:
    """A ``.aio(...)`` coroutine that never resolves (models a stalled stream)."""

    async def _acall(self, *args, **kwargs):
        await asyncio.Event().wait()

    @property
    def aio(self):
        return self._acall


class _HangingStream:
    def __init__(self):
        self.read = _HangingMethod()


class _StalledReadProcess:
    """Process whose stdout read hangs forever; stderr/wait would succeed."""

    def __init__(self):
        self.stdout = _HangingStream()
        self.stderr = _FakeStream("")
        self.wait = _FakeMethod(lambda: 0)


class _StalledExecSandbox:
    def __init__(self):
        self.exec = _FakeMethod(lambda *a, **k: _StalledReadProcess())


class _RecordingSandbox:
    """Sandbox whose exec returns a fixed process; records the remote timeout."""

    def __init__(self, *, stdout: str = "OUT", stderr: str = "ERR", rc: int = 7):
        self._out, self._err, self._rc = stdout, stderr, rc
        self.exec_timeouts: list = []
        self.exec = _FakeMethod(self._exec)

    def _exec(self, shell, flag, command, timeout=None, **kwargs):
        self.exec_timeouts.append(timeout)
        return _FakeProcess(stdout=self._out, stderr=self._err, return_code=self._rc)


class _CopyToLocal:
    """Fake copy_to_local: hangs for the first ``hang_times`` calls, else records."""

    def __init__(self, hang_times: int, sink: list):
        self.hang_times = hang_times
        self.calls = 0
        self._sink = sink

    async def _acall(self, source, target):
        self.calls += 1
        if self.calls <= self.hang_times:
            await asyncio.Event().wait()
        self._sink.append((str(source), str(target)))

    @property
    def aio(self):
        return self._acall


class _DownloadFilesystem:
    def __init__(self, hang_times: int):
        self.written: list[tuple[str, str]] = []
        self.copy_to_local = _CopyToLocal(hang_times, self.written)


class _DownloadSandbox:
    def __init__(self, hang_times: int):
        self.filesystem = _DownloadFilesystem(hang_times)


class TestSdkExecClientTimeout:
    async def test_none_client_timeout_is_byte_identical(self, temp_dir):
        env = _make_env(temp_dir)
        env._sandbox = _RecordingSandbox(stdout="OUT", stderr="ERR", rc=7)  # type: ignore[assignment]

        # Default (no arg) and explicit None set no deadline and return the result as is.
        default = await env._sdk_exec("echo hi")
        explicit = await env._sdk_exec("echo hi", client_timeout_secs=None)

        for result in (default, explicit):
            assert result.stdout == "OUT"
            assert result.stderr == "ERR"
            assert result.return_code == 7

    async def test_client_timeout_raises_on_stalled_read(self, temp_dir):
        env = _make_env(temp_dir)
        env._sandbox = _StalledExecSandbox()  # type: ignore[assignment]

        with pytest.raises(TimeoutError):
            await env._sdk_exec("echo hi", client_timeout_secs=0.05)


class TestSdkDownloadFileTimeout:
    async def test_hung_copy_raises_timeout_after_retry(self, temp_dir, monkeypatch):
        monkeypatch.setattr(modal_mod, "_FILE_COPY_TIMEOUT_SECS", 0.05)
        env = _make_env(temp_dir)
        env._sandbox = _DownloadSandbox(hang_times=999)  # type: ignore[assignment]

        with pytest.raises(TimeoutError):
            await env._sdk_download_file("/remote/f", temp_dir / "local_f")

        # @retry engaged: two attempts, both timed out (a hang would never
        # reach the second call).
        assert env._sandbox.filesystem.copy_to_local.calls == 2

    async def test_retry_succeeds_when_second_attempt_completes(
        self, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(modal_mod, "_FILE_COPY_TIMEOUT_SECS", 0.05)
        env = _make_env(temp_dir)
        env._sandbox = _DownloadSandbox(hang_times=1)  # type: ignore[assignment]

        await env._sdk_download_file("/remote/f", "/local/f")

        fs = env._sandbox.filesystem
        assert fs.copy_to_local.calls == 2
        assert fs.written == [("/remote/f", "/local/f")]


class TestSdkDownloadDirTimeout:
    async def test_surfaces_file_that_times_out_twice(self, temp_dir, monkeypatch):
        monkeypatch.setattr(modal_mod, "_FILE_COPY_TIMEOUT_SECS", 0.05)
        env = _make_env(temp_dir)
        env._sandbox = _DownloadSandbox(hang_times=999)  # type: ignore[assignment]

        captured: dict = {}

        async def _fake_exec_on_vm(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return ExecResult(stdout="/remote/dir/file.txt\n", stderr="", return_code=0)

        env._strategy.exec_on_vm = _fake_exec_on_vm  # type: ignore[method-assign]

        with pytest.raises(BaseExceptionGroup) as excinfo:
            await env._sdk_download_dir("/remote/dir", temp_dir / "out")

        # The directory-listing exec is bounded by the client-side deadline...
        assert captured["kwargs"]["client_timeout_secs"] == _DIR_LIST_TIMEOUT_SECS
        # ...and a file that times out on both attempts is surfaced, not swallowed.
        matched, _rest = excinfo.value.split(TimeoutError)
        assert matched is not None
        assert env._sandbox.filesystem.copy_to_local.calls == 2
