"""Unit tests for the Modal serve-path (launch / start / endpoint /
commit_checkpoint + capabilities). The Modal SDK is mocked throughout —
these tests never touch the network."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("modal")

from harbor.environments.base import BaseEnvironment
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.launch import LaunchSpec
from harbor.environments.modal import ModalEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _make_serve_env(
    temp_dir: Path,
    *,
    serve_command: list[str] | None = None,
    service_port: int | None = 8000,
    serve_image: object = "img:latest",
    gpus: int = 0,
    allow_internet: bool = False,
    region: str | None = None,
    env: dict[str, str] | None = None,
) -> ModalEnvironment:
    """Construct a serve-mode ModalEnvironment directly (no SDK calls)."""
    env_dir = temp_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    trial_dir = temp_dir / "trial"
    trial_dir.mkdir(exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()
    return ModalEnvironment(
        environment_dir=env_dir,
        environment_name="serve-task",
        session_id="Serve.Session.1",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image=serve_image if isinstance(serve_image, str) else None,
            gpus=gpus,
            allow_internet=allow_internet,
            env=env or {},
        ),
        _serve_mode=True,
        _service_command=serve_command or ["python", "server.py"],
        _service_port=service_port,
        _serve_image=serve_image,
        _serve_region=region,
    )


class _CommitEnv(BaseEnvironment):
    """Minimal env that can commit and returns an opaque (non-str) handle."""

    _SENTINEL = object()

    @staticmethod
    def type() -> str:
        return "commit-dummy"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(can_commit=True)

    def _validate_definition(self): ...
    async def start(self, force_build): ...
    async def stop(self, delete): ...
    async def upload_file(self, source_path, target_path): ...
    async def upload_dir(self, source_dir, target_dir): ...
    async def download_file(self, source_path, target_path): ...
    async def download_dir(self, source_dir, target_dir): ...
    async def exec(self, *args, **kwargs): ...

    def commit_checkpoint(
        self, tag: str, entrypoint: list[str] | None = None
    ) -> object:
        return self._SENTINEL


class TestCommitCheckpointReturnsObject:
    def test_base_signature_allows_opaque_handle(self):
        # Construct without running BaseEnvironment.__init__ (it does heavy
        # validation); we only exercise the overridden commit_checkpoint.
        env = _CommitEnv.__new__(_CommitEnv)
        assert env.commit_checkpoint("ignored-tag") is _CommitEnv._SENTINEL


def _patch_modal_sdk(
    monkeypatch,
    *,
    sandbox=None,
    image=None,
):
    """Patch the Modal SDK symbols imported into harbor.environments.modal.

    Returns a SimpleNamespace exposing the ``Sandbox.create.aio``,
    ``App.lookup.aio`` and ``Image.from_registry`` mocks plus the fake sandbox
    so tests can assert on call args without touching the network.
    """
    from harbor.environments import modal as modal_mod

    fake_sandbox = sandbox or MagicMock(name="sandbox")
    fake_image = image if image is not None else MagicMock(name="image")
    fake_app = MagicMock(name="app")

    create = AsyncMock(return_value=fake_sandbox)
    lookup = AsyncMock(return_value=fake_app)
    from_registry = MagicMock(return_value=fake_image)
    # Spy on Secret.from_dict so tests can assert env vars are injected as a
    # Modal Secret (from_dict is local — no network); returns a sentinel we can
    # locate in the Sandbox.create secrets list, and records the dict passed in.
    from_dict = MagicMock(return_value=MagicMock(name="env-secret"))

    monkeypatch.setattr(modal_mod.Sandbox, "create", MagicMock(aio=create))
    monkeypatch.setattr(modal_mod.App, "lookup", MagicMock(aio=lookup))
    monkeypatch.setattr(modal_mod.Image, "from_registry", from_registry)
    monkeypatch.setattr(modal_mod.Secret, "from_dict", from_dict)

    return SimpleNamespace(
        create=create,
        lookup=lookup,
        from_registry=from_registry,
        from_dict=from_dict,
        sandbox=fake_sandbox,
        image=fake_image,
        app=fake_app,
    )


class TestServeCapabilities:
    def test_serve_mode_reports_serve_caps(self, temp_dir):
        env = _make_serve_env(temp_dir)
        caps = env.capabilities
        assert caps.serves is True
        assert caps.can_commit is True
        assert caps.gpu_pinning is False
        assert caps.gpus is True
        assert caps.disable_internet is True

    def test_non_serve_mode_keeps_compose_caps(self, temp_dir):
        env_dir = temp_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        trial_dir = temp_dir / "trial"
        trial_dir.mkdir(exist_ok=True)
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()
        env = ModalEnvironment(
            environment_dir=env_dir,
            environment_name="t",
            session_id="S.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(),
        )
        assert env.capabilities.serves is False
        assert env.capabilities.can_commit is False


class TestLaunch:
    def test_launch_builds_prebuilt_serve_env(self):
        env = ModalEnvironment.launch(
            LaunchSpec(
                image="img:latest",
                serve_command=["python", "s.py"],
                service_port=8000,
                gpus=1,
                allow_internet=False,
            )
        )
        assert env.is_prebuilt is True
        assert env._serve_mode is True
        assert env._service_port == 8000
        assert env._service_command == ["python", "s.py"]
        assert env._serve_image == "img:latest"
        assert env.capabilities.serves is True
        # launch must not have created a sandbox (no network touched)
        assert env._sandbox is None

    def test_launch_stores_region(self):
        env = ModalEnvironment.launch(
            LaunchSpec(
                image="img:latest",
                serve_command=["python", "s.py"],
                service_port=8000,
                region="us-east",
            )
        )
        assert env._serve_region == "us-east"

    def test_launch_region_defaults_none(self):
        env = ModalEnvironment.launch(
            LaunchSpec(image="img:latest", serve_command=["x"], service_port=8000)
        )
        assert env._serve_region is None

    def test_launch_threads_env_into_persistent_env(self):
        env = ModalEnvironment.launch(
            LaunchSpec(
                image="img:latest",
                serve_command=["python", "s.py"],
                service_port=8000,
                env={"HF_HUB_OFFLINE": "1", "MODEL": "x"},
            )
        )
        # spec.env must reach the env-injection source (the same path that
        # _secrets_config turns into a Modal Secret at start()).
        assert env.task_env_config.env == {"HF_HUB_OFFLINE": "1", "MODEL": "x"}
        assert env._persistent_env["HF_HUB_OFFLINE"] == "1"
        assert env._persistent_env["MODEL"] == "x"

    def test_launch_empty_env_default(self):
        env = ModalEnvironment.launch(
            LaunchSpec(image="img:latest", serve_command=["x"], service_port=8000)
        )
        assert env.task_env_config.env == {}

    def test_launch_accepts_image_handle(self):
        sentinel = object()
        env = ModalEnvironment.launch(
            LaunchSpec(
                image=sentinel,
                serve_command=["python", "s.py"],
                service_port=8000,
            )
        )
        assert env._serve_image is sentinel

    def test_launch_rejects_gpu_device_ids(self):
        with pytest.raises(ValueError, match="gpu_pinning"):
            ModalEnvironment.launch(
                LaunchSpec(
                    image="img:latest",
                    serve_command=["x"],
                    service_port=8000,
                    gpu_device_ids=["0"],
                )
            )


class TestServeStart:
    async def test_start_creates_sandbox_with_ports_and_gpu(
        self, temp_dir, monkeypatch
    ):
        env = _make_serve_env(
            temp_dir,
            serve_command=["python", "s.py"],
            service_port=8000,
            gpus=1,
            serve_image="img:latest",
            allow_internet=False,
        )
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        sdk.create.assert_awaited_once()
        args, kwargs = sdk.create.call_args
        assert list(args) == ["python", "s.py"]
        assert kwargs["encrypted_ports"] == [8000]
        assert kwargs["gpu"] == "any:1"
        assert kwargs["image"] is sdk.image
        sdk.from_registry.assert_called_once_with("img:latest", secret=None)
        assert env._sandbox is sdk.sandbox
        # Sealed serve: Modal forbids block_network with open ports, so sealing
        # is realized as inbound-open + outbound-denied (empty egress allowlist).
        assert kwargs["block_network"] is False
        assert kwargs["outbound_cidr_allowlist"] == []

    async def test_start_pins_cpu_request_equals_limit_from_launch_spec(
        self, monkeypatch
    ):
        """A LaunchSpec.cpus flows through launch() (default AUTO policy) to
        Sandbox.create as cpu=(cpus, cpus): request == limit, so the CPU envelope is
        pinned rather than platform-decided. This is the contract the hls Modal
        substrate depends on for every sandbox it launches."""
        env = ModalEnvironment.launch(
            LaunchSpec(
                image="img:latest",
                serve_command=["python", "s.py"],
                service_port=8000,
                cpus=16,
            )
        )
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        _, kwargs = sdk.create.call_args
        assert kwargs["cpu"] == (16, 16)

    async def test_start_injects_env_as_secret(self, temp_dir, monkeypatch):
        env = _make_serve_env(temp_dir, env={"HF_HUB_OFFLINE": "1", "MODEL": "x"})
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        # Env vars must be injected into the sandbox via Secret.from_dict, and
        # the resulting secret handed to Sandbox.create(secrets=...).
        sdk.from_dict.assert_called_once()
        (passed_dict,), _ = sdk.from_dict.call_args
        assert passed_dict["HF_HUB_OFFLINE"] == "1"
        assert passed_dict["MODEL"] == "x"
        _, kwargs = sdk.create.call_args
        assert sdk.from_dict.return_value in kwargs["secrets"]

    async def test_start_no_secret_without_env(self, temp_dir, monkeypatch):
        env = _make_serve_env(temp_dir)  # no env
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        sdk.from_dict.assert_not_called()
        _, kwargs = sdk.create.call_args
        assert kwargs["secrets"] == []

    async def test_open_serve_has_no_outbound_restriction(self, temp_dir, monkeypatch):
        # allow_internet=True (e.g. BUILD): the served sandbox keeps full egress.
        env = _make_serve_env(
            temp_dir,
            serve_command=["python", "s.py"],
            service_port=8000,
            allow_internet=True,
        )
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        _, kwargs = sdk.create.call_args
        assert kwargs["encrypted_ports"] == [8000]
        assert kwargs["block_network"] is False
        assert "outbound_cidr_allowlist" not in kwargs

    async def test_start_pins_region_when_set(self, temp_dir, monkeypatch):
        env = _make_serve_env(temp_dir, region="us-east")
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        _, kwargs = sdk.create.call_args
        assert kwargs["region"] == "us-east"

    async def test_start_omits_region_by_default(self, temp_dir, monkeypatch):
        env = _make_serve_env(temp_dir)  # no region
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        _, kwargs = sdk.create.call_args
        assert "region" not in kwargs

    async def test_start_uses_image_handle_directly(self, temp_dir, monkeypatch):
        handle = MagicMock(name="snapshot-image")
        env = _make_serve_env(
            temp_dir,
            serve_command=["python", "s.py"],
            service_port=9000,
            serve_image=handle,
        )
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        # A pre-built image handle is used directly; from_registry is not called.
        sdk.from_registry.assert_not_called()
        _, kwargs = sdk.create.call_args
        assert kwargs["image"] is handle


class TestEndpoint:
    def _env_with_tunnels(self, temp_dir, tunnels):
        env = _make_serve_env(temp_dir, service_port=8000)
        aio = AsyncMock(return_value=tunnels)
        env._sandbox = MagicMock(tunnels=MagicMock(aio=aio))
        return env, aio

    def test_endpoint_returns_tunnel_url(self, temp_dir):
        env, _ = self._env_with_tunnels(
            temp_dir, {8000: SimpleNamespace(url="https://x.modal.host")}
        )
        assert env.endpoint(8000) == "https://x.modal.host"

    def test_endpoint_caches_tunnel_dict(self, temp_dir):
        env, aio = self._env_with_tunnels(
            temp_dir, {8000: SimpleNamespace(url="https://x.modal.host")}
        )
        env.endpoint(8000)
        env.endpoint(8000)
        aio.assert_awaited_once()

    def test_endpoint_missing_port_raises(self, temp_dir):
        env, _ = self._env_with_tunnels(
            temp_dir, {8000: SimpleNamespace(url="https://x.modal.host")}
        )
        with pytest.raises(RuntimeError, match="9999"):
            env.endpoint(9999)

    def test_endpoint_without_sandbox_raises(self, temp_dir):
        env = _make_serve_env(temp_dir, service_port=8000)
        with pytest.raises(RuntimeError):
            env.endpoint(8000)


class TestCommitCheckpoint:
    def test_commit_returns_snapshot_image(self, temp_dir):
        env = _make_serve_env(temp_dir)
        sentinel = MagicMock(name="snapshot-image")
        env._sandbox = MagicMock(
            snapshot_filesystem=MagicMock(aio=AsyncMock(return_value=sentinel))
        )
        result = env.commit_checkpoint("ignored-tag")
        assert result is sentinel

    def test_commit_snapshot_relaunchable_as_image(self, temp_dir):
        """The returned handle must be accepted back by launch() as the image."""
        env = _make_serve_env(temp_dir)
        sentinel = MagicMock(name="snapshot-image")
        env._sandbox = MagicMock(
            snapshot_filesystem=MagicMock(aio=AsyncMock(return_value=sentinel))
        )
        image = env.commit_checkpoint("ignored-tag")
        relaunched = ModalEnvironment.launch(
            LaunchSpec(image=image, serve_command=["python", "s.py"], service_port=8000)
        )
        assert relaunched._serve_image is sentinel

    def test_commit_without_sandbox_raises(self, temp_dir):
        env = _make_serve_env(temp_dir)
        with pytest.raises(RuntimeError):
            env.commit_checkpoint("ignored-tag")

    def test_commit_gated_on_can_commit(self, temp_dir):
        # A non-serve (agent-eval) env has can_commit=False and must refuse.
        env_dir = temp_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        trial_dir = temp_dir / "trial"
        trial_dir.mkdir(exist_ok=True)
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()
        env = ModalEnvironment(
            environment_dir=env_dir,
            environment_name="t",
            session_id="S.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(),
        )
        assert env.capabilities.can_commit is False
        with pytest.raises(ValueError, match="can_commit"):
            env.commit_checkpoint("tag")


class TestKeepaliveServe:
    """``launch(serve_command=None)`` yields a region-pinned *keepalive* sandbox
    with no main service process -- a co-located load driver execs /
    uploads / downloads against it. This is the same Modal keepalive the
    agent-eval ``_ModalDirect`` path relies on (``Sandbox.create`` with no
    positional command args), reached here through ``_ModalServe`` with an empty
    command. No service port -> no inbound tunnel; region is pinned for
    co-location with the artifact under test.
    """

    async def test_keepalive_creates_sandbox_with_empty_command(self, monkeypatch):
        env = ModalEnvironment.launch(
            LaunchSpec(
                image="img:latest",
                serve_command=None,  # keepalive: no main process
                gpus=0,
                region="us-east",
                allow_internet=True,
            )
        )
        assert env._service_command is None
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        sdk.create.assert_awaited_once()
        args, kwargs = sdk.create.call_args
        # No positional command args => Modal keepalive sandbox: it stays alive
        # with no main process (exactly like the agent-eval _ModalDirect path),
        # so the driver can exec/upload/download into it.
        assert list(args) == []
        # No service_port => no inbound encrypted-port tunnel is opened.
        assert "encrypted_ports" not in kwargs
        # Region is pinned so the driver co-locates with the artifact.
        assert kwargs["region"] == "us-east"
        # allow_internet=True => egress open, no outbound restriction.
        assert kwargs["block_network"] is False
        assert "outbound_cidr_allowlist" not in kwargs
        assert env._sandbox is sdk.sandbox

    async def test_keepalive_omits_region_by_default(self, monkeypatch):
        env = ModalEnvironment.launch(
            LaunchSpec(image="img:latest", serve_command=None, gpus=0)
        )
        sdk = _patch_modal_sdk(monkeypatch)

        await env.start(force_build=False)

        _, kwargs = sdk.create.call_args
        assert "region" not in kwargs

    async def test_keepalive_exec_upload_download_route_to_sdk(self, monkeypatch):
        env = ModalEnvironment.launch(
            LaunchSpec(image="img:latest", serve_command=None, gpus=0)
        )
        _patch_modal_sdk(monkeypatch)  # mock SDK so start() touches no network
        await env.start(force_build=False)

        # exec / upload_file / download_file must delegate through the
        # _ModalServe strategy (inherited from _ModalDirect / _ModalStrategy) to
        # the SDK-facing helpers. Spy on those helpers to prove the routing.
        exec_spy = AsyncMock(return_value="ok")
        up_spy = AsyncMock()
        down_spy = AsyncMock()
        monkeypatch.setattr(env, "_sdk_exec", exec_spy)
        monkeypatch.setattr(env, "_sdk_upload_file", up_spy)
        monkeypatch.setattr(env, "_sdk_download_file", down_spy)

        await env.exec("echo hi")
        await env.upload_file("/local/f", "/remote/f")
        await env.download_file("/remote/f", "/local/f")

        exec_spy.assert_awaited_once()
        assert exec_spy.call_args.args[0] == "echo hi"
        up_spy.assert_awaited_once_with("/local/f", "/remote/f")
        down_spy.assert_awaited_once_with("/remote/f", "/local/f")


class TestLaunchSpecServicePort:
    def test_service_port_field(self):
        spec = LaunchSpec(image="x", service_port=8000)
        assert spec.service_port == 8000

    def test_service_port_defaults_none(self):
        assert LaunchSpec(image="x").service_port is None

    def test_region_field(self):
        spec = LaunchSpec(image="x", region="us-east")
        assert spec.region == "us-east"

    def test_region_defaults_none(self):
        assert LaunchSpec(image="x").region is None
