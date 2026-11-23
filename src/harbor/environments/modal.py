from __future__ import annotations

import asyncio
import os
import re
import shlex
import tarfile
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from tenacity import retry, stop_after_attempt, wait_exponential

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.docker import (
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    RESOURCES_COMPOSE_NAME,
    self_bind_mount,
    write_mounts_compose_file,
    write_resources_compose_file,
)
from harbor.environments.docker.compose_env import (
    ComposeInfraEnvVars,
    legacy_log_mount_env_vars,
    merge_compose_env,
)
from harbor.environments.docker.docker import _sanitize_docker_image_name
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.config import ServiceVolumeConfig
from harbor.models.trial.paths import TrialPaths
from harbor.utils.env import resolve_env_vars
from harbor.utils.optional_import import MissingExtraError

if TYPE_CHECKING:
    from harbor.environments.launch import LaunchSpec

try:
    from modal import App, Image, Sandbox, Secret, Volume

    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False

_MODAL_DEFAULT_CPU_REQUEST_CORES = 0.125
_MODAL_DEFAULT_MEMORY_REQUEST_MB = 128

# Default deadline (seconds) for resolving a sandbox's tunnel map in endpoint().
# GPU sandboxes (e.g. oracle/metric) may take longer to boot and expose their
# encrypted-port tunnel. 300s gives headroom for GPU allocation (which can take
# 2-5 minutes for GPU types like A100/H100 under demand) plus container boot time.
_MODAL_TUNNEL_TIMEOUT_SEC = 300

# Client-side deadline (seconds) per file copy. The SDK copy call exposes no
# deadline parameter and a stalled stream never raises, so bound each copy here.
_FILE_COPY_TIMEOUT_SECS = 120

# Client-side deadline (seconds) for the directory-listing exec that precedes a
# recursive download. Bounds the local SDK awaits so a stalled stream cannot
# wedge the pull before any file copy begins.
_DIR_LIST_TIMEOUT_SECS = 60


def _run_blocking(make_coro: Any) -> Any:
    """Run an async SDK call from a synchronous method.

    ``endpoint()`` matches the synchronous ``BaseEnvironment.endpoint``
    signature but Modal's tunnel lookup is async. ``make_coro`` is a zero-arg
    factory that builds the coroutine; building it lazily lets us create it on
    whichever event loop will actually run it. When no loop is running we use
    ``asyncio.run``; when called from within a running loop (e.g. an async
    deploy path) we run it to completion on a dedicated worker thread.
    """
    import concurrent.futures

    async def _run() -> Any:
        return await make_coro()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run())).result()


class _ModalStrategy:
    """Base class for Modal execution strategies.

    A direct strategy runs everything in a single sandbox container,
    while a compose (DinD) strategy runs Docker inside the sandbox and
    orchestrates multiple containers via docker-compose.

    Methods that simply delegate to the Modal SDK (upload, download,
    is_dir, is_file) have concrete defaults here so that only strategies
    with genuinely different behaviour need to override them.
    """

    def __init__(self, env: "ModalEnvironment"):
        self._env = env

    @abstractmethod
    async def start(self, force_build: bool) -> None:
        """Start the environment."""

    async def stop(self, delete: bool) -> None:
        """Stop the environment and optionally delete resources."""
        await self._teardown_sandbox()

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Execute a command in the environment's main container."""

    @abstractmethod
    async def attach(self) -> None:
        """Attach an interactive shell to the environment."""

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._sdk_upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._env._sdk_upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._sdk_download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._env._sdk_download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a directory (uses sandbox.ls)."""
        if not self._env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        try:
            await self._env._sandbox.ls.aio(path)
            return True
        except (NotADirectoryError, FileNotFoundError):
            return False

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a file (uses sandbox.ls)."""
        if not self._env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        try:
            await self._env._sandbox.ls.aio(path)
            return False
        except NotADirectoryError:
            return True
        except FileNotFoundError:
            return False

    async def _teardown_sandbox(self) -> None:
        """Terminate the sandbox and reset references."""
        env = self._env
        if not env._sandbox:
            return
        try:
            await env._terminate_sandbox()
            await env._sandbox.wait.aio(raise_on_termination=False)
        except Exception as e:
            env.logger.warning(f"Error terminating Modal sandbox: {e}")
        finally:
            env._sandbox = None
            env._app = None
            env._image = None

    async def container_logs(self, service: str = "main") -> str:
        """Capture the sandbox's stdout+stderr and return them as a string.

        Terminates the sandbox first so the streams reach EOF, then reads them
        via the Modal SDK.  The sandbox is stopped as a side-effect; a
        subsequent ``stop()`` call is safe (idempotent: no sandbox to tear
        down).  Subclasses that run Docker-in-Docker override this to read
        logs via ``docker compose logs`` before stopping.
        """
        env = self._env
        saved_sandbox = env._sandbox
        await self._teardown_sandbox()  # terminates → EOF on streams
        if saved_sandbox is None:
            return ""
        stdout = await saved_sandbox.stdout.read.aio()
        stderr = await saved_sandbox.stderr.read.aio()
        return (stdout or "") + (stderr or "")

    async def exec_on_vm(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash",
        client_timeout_secs: float | None = None,
    ) -> ExecResult:
        """Run a command directly on the sandbox VM (bypasses compose)."""
        return await self._env._sdk_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            shell=shell,
            client_timeout_secs=client_timeout_secs,
        )


class _ModalDirect(_ModalStrategy):
    """Single-container sandbox — runs agent and verifier directly in the
    Modal sandbox.

    Inherits upload/download/is_dir/is_file from `_ModalStrategy` since
    the default SDK implementations are sufficient for a single container.
    """

    async def start(self, force_build: bool) -> None:
        env = self._env

        docker_image = env.task_env_config.docker_image
        if docker_image:
            registry_secret = (
                Secret.from_name(env._registry_secret) if env._registry_secret else None
            )
            if ".dkr.ecr." in docker_image:
                env._image = Image.from_aws_ecr(
                    docker_image,
                    secret=registry_secret,
                )
            else:
                env._image = Image.from_registry(
                    docker_image,
                    secret=registry_secret,
                )
        else:
            env._image = Image.from_dockerfile(
                env._environment_definition_path,
                context_dir=env.environment_dir,
            )

        env._app = await App.lookup.aio(
            name=env._app_name,
            create_if_missing=True,
        )

        env._sandbox = await env._create_sandbox()

        # Create log directories and make them world-writable so non-root
        # agent/verifier users can write to them.
        await env.ensure_dirs(env._mount_targets(writable_only=True))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._env._sdk_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, login=True
        )

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        os.execvp(
            "modal",
            [
                "modal",
                "shell",
                env._sandbox.object_id,
            ],
        )


class _ModalServe(_ModalDirect):
    """Serve-mode strategy: run a prebuilt image's service command as the
    sandbox's main process, with the service port tunnelled.

    Created via ``ModalEnvironment.launch(LaunchSpec(...))``. Unlike
    ``_ModalDirect`` (agent-eval: a keepalive sandbox the agent execs into),
    this boots ``serve_command`` directly and exposes ``service_port`` through a
    Modal encrypted-port tunnel resolved by ``endpoint()``. exec/attach/upload/
    download are inherited from ``_ModalDirect`` for debugging the live service.
    """

    async def start(self, force_build: bool) -> None:
        env = self._env

        serve_image = env._serve_image
        if isinstance(serve_image, str):
            env._image = Image.from_registry(serve_image, secret=None)
        else:
            # A modal.Image handle produced by a prior commit_checkpoint
            # (filesystem snapshot) — run it directly, no registry pull.
            env._image = serve_image

        env._app = await App.lookup.aio(
            name=env._app_name,
            create_if_missing=True,
        )

        encrypted_ports = [env._service_port] if env._service_port is not None else []
        env._sandbox = await env._create_sandbox(
            command_args=env._service_command or [],
            encrypted_ports=encrypted_ports,
            region=env._serve_region,
        )


class _ModalDinD(_ModalStrategy):
    """Docker-in-Docker compose strategy for multi-container tasks.

    Uses Modal's ``experimental_options={"enable_docker": True}`` to run
    a Docker daemon inside the sandbox.

    Topology:
        Local machine (harbor CLI)
          └── Modal Sandbox (DinD, enable_docker=True)
                ├── dockerd (Docker daemon, managed by Modal)
                └── docker compose
                      ├── main        ← agent runs here, exec/upload/download target
                      ├── sidecar     ← additional services
                      └── ...
    """

    # Max iterations when polling for Docker daemon readiness.
    # Each iteration sleeps 2s, so worst-case wall-clock time is ~60s.
    _DOCKER_DAEMON_POLL_LIMIT = 30
    _COMPOSE_DIR = "/harbor/compose"
    _ENVIRONMENT_DIR = "/harbor/environment"
    _MOUNTS_COMPOSE_NAME = "docker-compose-mounts.json"

    def __init__(self, env: "ModalEnvironment"):
        super().__init__(env)
        self._use_prebuilt = False

        self._resolved_task_env: dict[str, str] = {}
        if self._env.task_env_config.env:
            self._resolved_task_env = resolve_env_vars(self._env.task_env_config.env)

    @staticmethod
    def _build_host_network_overlay(
        environment_dir: Path,
        *,
        use_prebuilt: bool = False,
        extra_compose_paths: list[Path] | None = None,
    ) -> str:
        """Generate a compose overlay that sets host networking on all services.

        Parses service names from the task's docker-compose.yaml and extra
        compose overlays so the overlay covers all services regardless of naming
        conventions.
        Only adds ``build.network: host`` for services that have a build
        context (not pure image-based services like redis).

        Each service gets ``extra_hosts`` entries mapping every *other*
        service name to ``127.0.0.1`` so that Docker DNS hostnames
        (e.g. ``redis``, ``agent1``) resolve correctly under host networking.
        """
        import yaml

        compose_path = environment_dir / "docker-compose.yaml"
        services: dict[str, bool] = {}  # name -> has_build
        compose_paths = [compose_path, *(extra_compose_paths or [])]
        for path in compose_paths:
            if not path.exists():
                continue
            doc = yaml.safe_load(path.read_text())
            if doc and "services" in doc:
                for name, cfg in doc["services"].items():
                    has_build = isinstance(cfg, dict) and "build" in cfg
                    services[name] = services.get(name, False) or has_build

        # Fallback if parsing fails
        if not services:
            services = {"main": True, "sidecar": True, "redis": False}

        # main always needs host networking.  In build mode it also needs
        # build.network: host; in prebuilt mode only network_mode: host.
        if use_prebuilt:
            services.setdefault("main", False)
        else:
            services["main"] = True

        service_names = list(services.keys())
        lines = ["services:"]
        for svc, has_build in services.items():
            lines.append(f"  {svc}:")
            if has_build:
                lines.append("    build:")
                lines.append("      network: host")
            lines.append("    network_mode: host")
            # Map all other service names to localhost so Docker DNS
            # hostnames work under host networking.
            others = [s for s in service_names if s != svc]
            if others:
                lines.append("    extra_hosts:")
                for other in others:
                    lines.append(f'      - "{other}:127.0.0.1"')
            # NOTE: Do NOT add environment: here — it replaces (not merges)
            # the service's entire environment block from the base compose
            # file, wiping out AGENT_ID, API keys, etc.
        return "\n".join(lines)

    async def _vm_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "sh",
    ) -> ExecResult:
        """Run a command on the DinD sandbox VM (defaults to sh for Alpine)."""
        return await self._env._sdk_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, shell=shell
        )

    def _compose_referenced_env_vars(self) -> dict[str, str]:
        """Extract env vars referenced in task and extra docker compose files.

        Parses ``${VAR_NAME}`` and ``${VAR_NAME:-default}`` patterns from the
        compose files and returns values from os.environ for any that are set.
        """
        compose_paths = [
            self._env.environment_dir / "docker-compose.yaml",
            *self._env.extra_docker_compose_paths,
        ]

        content = "\n".join(path.read_text() for path in compose_paths if path.exists())
        # Match ${VAR}, ${VAR:-default}, and bare $VAR references
        matches = re.findall(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)\b",
            content,
        )
        var_names = {g1 or g2 for g1, g2 in matches}

        env_vars: dict[str, str] = {}
        for name in var_names:
            value = os.environ.get(name)
            if value is not None:
                env_vars[name] = value
        return env_vars

    def _infra_env_vars(self) -> dict[str, str]:
        """Harbor infrastructure vars required by the compose templates."""
        env_vars = ComposeInfraEnvVars(
            main_image_name=_sanitize_docker_image_name(
                f"hb__{self._env.environment_name}"
            ),
            context_dir=self._ENVIRONMENT_DIR,
            prebuilt_image_name=(
                self._env.task_env_config.docker_image if self._use_prebuilt else None
            ),
            cpus=self._env._effective_cpus,
            memory=f"{memory_mb}M"
            if (memory_mb := self._env._effective_memory_mb)
            else None,
        ).to_env_dict()
        env_vars.update(
            legacy_log_mount_env_vars(self._resolve_volumes(), host_value="target")
        )
        return env_vars

    def _compose_env_vars(self) -> dict[str, str]:
        """All environment variables for docker compose commands."""
        user_env: dict[str, str] = self._compose_referenced_env_vars()
        if self._resolved_task_env:
            user_env.update(self._resolved_task_env)
        if self._env._persistent_env:
            user_env.update(self._env._persistent_env)
        return merge_compose_env(
            user_env=user_env,
            infra_env=self._infra_env_vars(),
            logger=self._env.logger,
            collision_label="Referenced/task/persistent env vars",
        )

    def _compose_file_flags(self) -> list[str]:
        """Return -f flag pairs for all compose files as a flat list."""
        build_or_prebuilt = (
            "docker-compose-prebuilt.yaml"
            if self._use_prebuilt
            else "docker-compose-build.yaml"
        )
        files = [
            f"{self._COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}",
            f"{self._COMPOSE_DIR}/{build_or_prebuilt}",
            f"{self._COMPOSE_DIR}/{self._MOUNTS_COMPOSE_NAME}",
        ]
        if (self._env.environment_dir / "docker-compose.yaml").exists():
            files.append(f"{self._ENVIRONMENT_DIR}/docker-compose.yaml")
        files.extend(self._extra_compose_target_paths())
        if not self._env.task_env_config.allow_internet:
            files.append(f"{self._COMPOSE_DIR}/docker-compose-no-network.yaml")

        # Modal sandboxes lack netlink permissions for creating veth pairs,
        # so all services must use the host network namespace.
        files.append(f"{self._COMPOSE_DIR}/docker-compose-host-network.yaml")

        flags: list[str] = []
        for f in files:
            flags.extend(["-f", f])
        return flags

    def _extra_compose_target_paths(self) -> list[str]:
        return [
            f"{self._COMPOSE_DIR}/docker-compose-extra-{index}.yaml"
            for index, _ in enumerate(self._env.extra_docker_compose_paths)
        ]

    async def _stage_extra_compose_files(self) -> None:
        for source, target in zip(
            self._env.extra_docker_compose_paths,
            self._extra_compose_target_paths(),
            strict=True,
        ):
            await self._env._sdk_upload_file(source, target)

    def _resolve_volumes(self) -> list[ServiceVolumeConfig]:
        """Materialize Trial's mount intent for the VM filesystem (self-bind).

        See ``_DaytonaDinD._resolve_volumes`` for the full rationale.
        """
        return [
            self_bind_mount(m) if m.get("type") == "bind" else m
            for m in self._env._mounts
        ]

    async def _stage_mounts_compose_file(
        self, volumes: list[ServiceVolumeConfig]
    ) -> None:
        """Write the mounts compose override locally and upload it to the VM."""
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / self._MOUNTS_COMPOSE_NAME
            write_mounts_compose_file(local_path, volumes)
            await self._env._sdk_upload_file(
                local_path,
                f"{self._COMPOSE_DIR}/{self._MOUNTS_COMPOSE_NAME}",
            )

    async def _stage_resources_compose_file(self) -> None:
        """Write the resource policy compose override locally and upload it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / RESOURCES_COMPOSE_NAME
            write_resources_compose_file(
                local_path,
                cpu_request=self._env._resource_request_value(
                    "cpu", auto_mode=ResourceMode.LIMIT
                ),
                cpu_limit=self._env._resource_limit_value(
                    "cpu", auto_mode=ResourceMode.LIMIT
                ),
                memory_request_mb=self._env._resource_request_value(
                    "memory", auto_mode=ResourceMode.LIMIT
                ),
                memory_limit_mb=self._env._resource_limit_value(
                    "memory", auto_mode=ResourceMode.LIMIT
                ),
            )
            await self._env._sdk_upload_file(
                local_path,
                f"{self._COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}",
            )

    @property
    def _project_name(self) -> str:
        return self._env.session_id.lower().replace(".", "-")

    def _compose_cmd(self, subcommand: list[str]) -> str:
        """Build a fully shell-escaped docker compose command string."""
        parts = [
            "docker",
            "compose",
            "-p",
            self._project_name,
            "--project-directory",
            self._ENVIRONMENT_DIR,
            *self._compose_file_flags(),
            *subcommand,
        ]
        return shlex.join(parts)

    async def _compose_exec(
        self,
        subcommand: list[str],
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Run a docker compose subcommand on the sandbox."""
        return await self._vm_exec(
            self._compose_cmd(subcommand),
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )

    async def _wait_for_docker_daemon(self) -> None:
        """Poll until the Docker daemon inside the sandbox is responsive."""
        self._env.logger.debug("Waiting for Docker daemon inside DinD sandbox...")
        last_output = ""
        for _ in range(self._DOCKER_DAEMON_POLL_LIMIT):
            result = await self._vm_exec("docker info", timeout_sec=10)
            if result.return_code == 0:
                self._env.logger.debug("Docker daemon is ready")
                return
            last_output = (result.stdout or "") + (result.stderr or "")
            await asyncio.sleep(2)
        raise RuntimeError(
            f"Docker daemon not ready after {self._DOCKER_DAEMON_POLL_LIMIT} "
            f"poll attempts. Last output: {last_output}"
        )

    async def _wait_for_main_container(self, timeout_sec: int = 60) -> None:
        """Poll until the 'main' compose service is running."""
        self._env.logger.debug("Waiting for main container to be running...")
        for _ in range(timeout_sec // 2):
            result = await self._compose_exec(
                ["exec", "-T", "main", "true"], timeout_sec=10
            )
            if result.return_code == 0:
                self._env.logger.debug("Main container is running")
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"Main container not running after {timeout_sec}s")

    async def start(self, force_build: bool) -> None:
        env = self._env

        dind_image: str = env._kwargs.get("dind_image", "docker:28.3.3-dind")
        # Pre-configure dockerd for Modal sandboxes which lack iptables kernel
        # modules and netlink permissions for creating veth pairs.  Disabling
        # iptables and the default bridge avoids both issues.  All compose
        # services must use network_mode: host (handled by a compose overlay).
        env._image = Image.from_registry(dind_image).dockerfile_commands(
            "RUN mkdir -p /etc/docker "
            '&& echo \'{"iptables": false, "bridge": "none"}\' '
            "> /etc/docker/daemon.json"
        )

        env._app = await App.lookup.aio(
            name=env._app_name,
            create_if_missing=True,
        )

        # DinD sandbox needs network for Docker daemon and image pulls
        env._sandbox = await env._create_sandbox(
            block_network=False,
            experimental_options={"enable_docker": True},
        )

        # Wait for Docker daemon to be ready inside the sandbox
        await self._wait_for_docker_daemon()

        env.logger.debug(
            "DinD mode uses host networking: no port isolation between "
            "services, no Docker DNS service discovery (extra_hosts entries "
            "map service names to 127.0.0.1 instead), and no network "
            "namespace isolation."
        )

        # Upload Harbor compose files to the sandbox
        for path in (
            COMPOSE_BUILD_PATH,
            COMPOSE_PREBUILT_PATH,
            COMPOSE_NO_NETWORK_PATH,
        ):
            await env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{path.name}")
        await self._stage_resources_compose_file()

        # Upload task environment directory (Dockerfiles, compose file, etc.)
        await env._sdk_upload_dir(env.environment_dir, self._ENVIRONMENT_DIR)

        await self._stage_extra_compose_files()

        # Materialize Trial's mount intent for the VM (self-bind), write the
        # compose override locally, and upload it alongside the shared files.
        volumes = self._resolve_volumes()
        await self._stage_mounts_compose_file(volumes)

        # Ensure each bind-mount source dir exists on the VM and is writable
        # by non-root agent/verifier users.
        bind_sources = [v["source"] for v in volumes if v.get("type") == "bind"]
        if bind_sources:
            quoted = " ".join(shlex.quote(s) for s in bind_sources)
            await self._vm_exec(f"mkdir -p {quoted} && chmod 777 {quoted}")

        # Build and start compose services
        self._use_prebuilt = not force_build and bool(env.task_env_config.docker_image)

        overlay = self._build_host_network_overlay(
            env.environment_dir,
            use_prebuilt=self._use_prebuilt,
            extra_compose_paths=env.extra_docker_compose_paths,
        )
        await self._vm_exec(
            f"cat > /harbor/compose/docker-compose-host-network.yaml << 'YAML'\n"
            f"{overlay}\n"
            f"YAML",
            timeout_sec=10,
        )

        env.logger.debug("Building compose services inside DinD sandbox...")
        result = await self._compose_exec(
            ["build"],
            timeout_sec=round(env.task_env_config.build_timeout_sec),
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose build failed: {result.stdout} {result.stderr}"
            )

        env.logger.debug("Starting compose services inside DinD sandbox...")
        result = await self._compose_exec(["up", "-d"], timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose up failed: {result.stdout} {result.stderr}"
            )

        await self._wait_for_main_container()

    async def container_logs(self, service: str = "main") -> str:
        """Capture logs from a docker compose service running inside the DinD sandbox.

        Reads via ``docker compose logs <service>`` while the sandbox is still
        alive (the compose stack has not been torn down yet).  Overrides the
        base strategy which reads the sandbox's own stdout/stderr (not relevant
        for DinD where the compose services produce the meaningful output).
        """
        result = await self._compose_exec(["logs", service])
        return (result.stdout or "") + (result.stderr or "")

    async def stop(self, delete: bool) -> None:
        if self._env._sandbox:
            try:
                await self._compose_exec(["down", "--remove-orphans"], timeout_sec=30)
            except Exception as e:
                self._env.logger.warning(f"docker compose down failed: {e}")

        await self._teardown_sandbox()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute command inside the main compose container."""
        parts: list[str] = ["exec", "-T"]
        if cwd:
            parts.extend(["-w", cwd])
        if env:
            for k, v in env.items():
                parts.extend(["-e", f"{k}={v}"])
        if user is not None:
            parts.extend(["-u", str(user)])
        parts.extend(["main", "bash", "-lc", command])

        return await self._compose_exec(parts, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Two-hop upload: SDK → sandbox temp, docker compose cp → main."""
        temp = f"/tmp/harbor_{uuid4().hex}"
        try:
            await self._env._sdk_upload_file(source_path, temp)
            result = await self._compose_exec(
                ["cp", temp, f"main:{target_path}"], timeout_sec=60
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Two-hop upload: SDK → sandbox temp dir, docker compose cp → main."""
        temp = f"/tmp/harbor_{uuid4().hex}"
        try:
            await self._env._sdk_upload_dir(source_dir, temp)
            result = await self._compose_exec(
                ["cp", f"{temp}/.", f"main:{target_dir}"], timeout_sec=120
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    def _sandbox_log_path(self, container_path: str) -> str | None:
        """Return *container_path* when it's under a self-bound log dir.

        Under the self-bind convention, the VM filesystem path equals the
        container path, so paths under ``/logs/{verifier,agent,artifacts}``
        can be transferred via the SDK directly without ``docker compose cp``.
        Returns ``None`` for paths outside the bound dirs so callers fall
        back to the compose-cp slow path.
        """
        prefixes = tuple(self._env._mount_targets())
        if any(
            container_path == p or container_path.startswith(p + "/") for p in prefixes
        ):
            return container_path
        return None

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the main container.

        Fast path: if the file is under a volume-mounted log dir, download
        directly from the sandbox. Slow path: docker compose cp to sandbox
        temp, then SDK download.
        """
        sandbox_path = self._sandbox_log_path(source_path)
        if sandbox_path:
            await self._env._sdk_download_file(sandbox_path, target_path)
            return

        temp = f"/tmp/harbor_{uuid4().hex}"
        try:
            result = await self._compose_exec(
                ["cp", f"main:{source_path}", temp], timeout_sec=60
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_file(temp, target_path)
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the main container.

        Fast path: if under a volume-mounted log dir, download directly from
        the sandbox. Slow path: docker compose cp to sandbox temp, then SDK
        download.
        """
        sandbox_path = self._sandbox_log_path(source_dir)
        if sandbox_path:
            await self._env._sdk_download_dir(sandbox_path, target_dir)
            return

        temp = f"/tmp/harbor_{uuid4().hex}"
        try:
            await self._vm_exec(f"mkdir -p {shlex.quote(temp)}", timeout_sec=10)
            result = await self._compose_exec(
                ["cp", f"main:{source_dir}/.", temp], timeout_sec=120
            )
            if result.return_code != 0:
                self._env.logger.error(
                    f"download_dir: docker compose cp failed: "
                    f"{result.stdout} {result.stderr}"
                )
                raise RuntimeError(
                    f"download_dir: docker compose cp failed: "
                    f"{result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_dir(temp, target_dir)
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -f {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Drop into the main compose container, not the DinD sandbox VM
        compose_exec_cmd = self._compose_cmd(["exec", "main", "bash"])
        os.execvp(
            "modal",
            ["modal", "shell", env._sandbox.object_id, "--cmd", compose_exec_cmd],
        )


class ModalEnvironment(BaseEnvironment):
    environment_dir: Path
    environment_name: str
    session_id: str
    trial_paths: TrialPaths
    config: EnvironmentConfig

    @classmethod
    def preflight(cls) -> None:
        import os
        from pathlib import Path

        modal_config = Path.home() / ".modal.toml"
        has_env_token = os.environ.get("MODAL_TOKEN_ID") and os.environ.get(
            "MODAL_TOKEN_SECRET"
        )
        if not modal_config.exists() and not has_env_token:
            raise SystemExit(
                "Modal requires authentication. Run 'modal token new' to set up "
                "credentials, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                "environment variables."
            )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.MODAL

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(
            cpu_limit=True,
            cpu_request=True,
            memory_limit=True,
            memory_request=True,
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return self._capabilities

    @property
    def _uses_compose(self) -> bool:
        return self._compose_mode

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def is_prebuilt(self) -> bool:
        """True when constructed from a LaunchSpec (serve mode): a prebuilt
        image (registry tag or snapshot handle), no on-disk build definition."""
        return self._serve_mode

    async def image_digest(self) -> str | None:
        """Modal's content-addressed built-image id (``im-…``), available once the
        image is hydrated (after the sandbox is created). ``None`` when no image is
        set or it has not hydrated yet. Never raises."""
        img = self._image
        if img is None:
            return None
        try:
            return img.object_id
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def launch(cls, spec: "LaunchSpec") -> "ModalEnvironment":
        """Construct a prebuilt serve-mode ModalEnvironment from a LaunchSpec.

        Mirrors ``DockerEnvironment.launch``: no on-disk Dockerfile/compose and
        no trial scaffolding — the image (a registry tag, or a ``modal.Image``
        handle returned by a prior ``commit_checkpoint``) is the definition.
        The sandbox is NOT created here; that happens in ``start()``. The spec
        is validated against this backend's capabilities (fail loud) before
        returning — notably a ``gpu_device_ids`` request raises because Modal
        has no device pinning (gpu_pinning=False).
        """
        import uuid

        # docker_image is a str field; a non-str image handle (modal.Image from a
        # snapshot) is carried separately in _serve_image and consumed by start().
        # gpu_kind from the spec maps to EnvironmentConfig.gpu_types so _gpu_config()
        # can request the correct GPU model (e.g. "H100:1" instead of "any:1").
        task_env_config = EnvironmentConfig(
            docker_image=spec.image if isinstance(spec.image, str) else None,
            gpus=spec.gpus or None,
            gpu_types=[spec.gpu_kind] if spec.gpu_kind else None,
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
            allow_internet=spec.allow_internet,
            workdir=spec.workdir,
            # Runtime env vars (HF_HUB_OFFLINE, model config, secrets) are
            # injected into the sandbox as a Modal Secret at start() — they flow
            # task_env_config.env -> _persistent_env -> _secrets_config().
            env=dict(spec.env),
        )

        environment_dir = Path(tempfile.mkdtemp(prefix="hb-modal-launch-"))
        trial_paths = TrialPaths(trial_dir=environment_dir)

        short_id = uuid.uuid4().hex[:8]
        env = cls(
            environment_dir=environment_dir,
            environment_name=f"launch-{short_id}",
            session_id=f"launch-{short_id}",
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            mounts=list(spec.mounts) if spec.mounts else None,
            _serve_mode=True,
            _service_command=list(spec.serve_command) if spec.serve_command else None,
            _service_port=spec.service_port,
            _serve_image=spec.image,
            _serve_region=spec.region,
            _launch_outbound_cidr_allowlist=spec.outbound_cidr_allowlist,
            _launch_outbound_domain_allowlist=spec.outbound_domain_allowlist,
        )
        env._validate_launch_capabilities(spec)
        return env

    def _validate_definition(self):
        # Serve mode runs a prebuilt image (registry tag or snapshot handle);
        # there is no on-disk Dockerfile/compose to validate.
        if self._serve_mode:
            return
        if self.task_env_config.docker_image:
            return
        if self._compose_mode:
            path = self.environment_dir / "docker-compose.yaml"
            if not path.exists() and self.extra_docker_compose_paths:
                return
        else:
            path = self._environment_definition_path
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Please ensure the file exists.")

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        secrets: list[str] | None = None,
        registry_secret: str | None = None,
        volumes: dict[str, str] | None = None,
        app_name: str = "__harbor__",
        sandbox_timeout_secs: int = 60 * 60 * 24,
        sandbox_idle_timeout_secs: int | None = None,
        extra_docker_compose: list[Path] | None = None,
        *args,
        _serve_mode: bool = False,
        _service_command: list[str] | None = None,
        _service_port: int | None = None,
        _serve_image: str | object | None = None,
        _serve_region: str | None = None,
        _launch_outbound_cidr_allowlist: list[str] | None = None,
        _launch_outbound_domain_allowlist: list[str] | None = None,
        **kwargs,
    ):
        """
        Initialize a ModalEnvironment instance.

        Args:
            environment_dir: The directory containing the environment definition files.
            environment_name: The name identifier for this environment instance.
            session_id: Unique session identifier for this environment instance.
            trial_paths: Path configuration object containing trial-related directory
                paths.
            task_env_config: Environment configuration specifying resources (CPU,
                memory), GPU types, and network access.
            secrets: Optional list of Modal secret names to mount into the sandbox.
            registry_secret: Optional Modal secret name for authenticating with
                a private container registry (e.g. AWS ECR). When set, the
                Dockerfile's FROM image is pulled using Image.from_aws_ecr()
                instead of Image.from_dockerfile().
            volumes: Optional mapping of mount paths to Modal volume names.
            app_name: Name of the Modal App to use. All sandboxes created
                with the same app name share a single Modal App. Default
                is "__harbor__".
            sandbox_timeout_secs: Maximum lifetime of the sandbox in seconds.
                The sandbox will be terminated after this duration regardless of
                activity. Default is 86400 (24 hours). See Modal sandbox docs:
                https://modal.com/docs/reference/modal.Sandbox#create
            sandbox_idle_timeout_secs: Seconds of inactivity after which the
                sandbox will be automatically terminated. None means no idle
                timeout (default). See Modal sandbox docs:
                https://modal.com/docs/reference/modal.Sandbox#create
        """
        # Serve-mode markers. Set *before* super().__init__ because
        # _validate_definition / capability validators read them.
        self._serve_mode = _serve_mode
        self._service_command = list(_service_command) if _service_command else None
        self._service_port = _service_port
        # Opaque runnable image handle for serve mode: a registry tag (str) or a
        # modal.Image from a prior snapshot. modal.Image is only importable when
        # the optional extra is installed, so this stays Any at the type level.
        self._serve_image: Any = _serve_image
        # Region to place the serve sandbox in (co-locates a load driver with
        # the artifact it measures); None lets Modal choose.
        self._serve_region = _serve_region
        # Outbound allowlists sealing the sandbox to specific destinations (the
        # egress proxy / model gateway), by CIDR and/or by domain name. None lets
        # egress follow the normal block_network policy. Set by ``launch`` from
        # LaunchSpec; threaded into Sandbox.create by _create_sandbox.
        self._launch_outbound_cidr_allowlist = _launch_outbound_cidr_allowlist
        self._launch_outbound_domain_allowlist = _launch_outbound_domain_allowlist
        # Resolved tunnel map, cached after the first endpoint() lookup.
        self._tunnels: dict[int, Any] | None = None

        # Detect compose mode *before* super().__init__ which calls
        # _validate_definition
        self._compose_mode = (environment_dir / "docker-compose.yaml").exists() or bool(
            extra_docker_compose
        )
        if self._serve_mode:
            # Serve path: a long-lived service on a tunnelled port, snapshot-able.
            # Modal pins GPU count+type only (no device indices), so gpu_pinning
            # stays False — a LaunchSpec with gpu_device_ids set is rejected by
            # the inherited _validate_launch_capabilities.
            self._capabilities = EnvironmentCapabilities(
                gpus=True,
                disable_internet=True,
                serves=True,
                can_commit=True,
                gpu_pinning=False,
                egress_allowlist=True,  # Sandbox.create(outbound_cidr_allowlist=...)
            )
        else:
            # DinD mode requires host networking — cannot enforce network isolation.
            self._capabilities = EnvironmentCapabilities(
                gpus=True,
                disable_internet=not self._compose_mode,
                docker_compose=True,
            )
        self._kwargs = kwargs
        if not _HAS_MODAL:
            raise MissingExtraError(package="modal", extra="modal")

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            extra_docker_compose=extra_docker_compose,
            **kwargs,
        )
        self._image: Image | None = None
        self._app: App | None = None
        self._sandbox: Sandbox | None = None
        self._secrets = secrets or []
        self._registry_secret = registry_secret
        self._volumes = volumes or {}
        self._app_name = app_name
        self._sandbox_timeout = sandbox_timeout_secs
        self._sandbox_idle_timeout = sandbox_idle_timeout_secs

        # Select strategy: serve mode takes precedence (a prebuilt service),
        # then DinD compose, then the single-container direct strategy.
        self._strategy: _ModalStrategy
        if self._serve_mode:
            self._strategy = _ModalServe(self)
        elif self._compose_mode:
            self._strategy = _ModalDinD(self)
        else:
            self._strategy = _ModalDirect(self)
        self.logger.debug(f"Selected strategy: {self._strategy.__class__.__name__}")

    @property
    def _default_shell(self) -> str:
        """Shell available on the sandbox VM.

        Alpine-based DinD images only have ``sh``; standard images have ``bash``.
        """
        return "sh" if self._compose_mode else "bash"

    def _cpu_config(self) -> int | float | tuple[int | float, int] | None:
        """Resolve CPU configuration for sandbox creation.

        Modal's scalar form is a request-only value with a soft limit that
        lets containers burst up to +16 cores. The tuple form sets separate
        request and limit values for stricter modes.
        """
        cpus = self._effective_cpus
        if cpus is None:
            return None
        if self._cpu_resource_mode == ResourceMode.REQUEST:
            return cpus
        if self._cpu_resource_mode == ResourceMode.LIMIT:
            return (min(_MODAL_DEFAULT_CPU_REQUEST_CORES, cpus), cpus)
        return (cpus, cpus)

    def _memory_config(self) -> int | tuple[int, int] | None:
        memory_mb = self._effective_memory_mb
        if memory_mb is None:
            return None
        if self._memory_resource_mode in (ResourceMode.AUTO, ResourceMode.REQUEST):
            return memory_mb
        if self._memory_resource_mode == ResourceMode.LIMIT:
            return (min(_MODAL_DEFAULT_MEMORY_REQUEST_MB, memory_mb), memory_mb)
        return (memory_mb, memory_mb)

    def _gpu_config(self) -> str | None:
        """Resolve GPU configuration string for sandbox creation."""
        if self._effective_gpus <= 0:
            return None
        gpu_type = "any"
        if self.task_env_config.gpu_types:
            if len(self.task_env_config.gpu_types) > 1:
                self.logger.debug(
                    "Multiple GPU types specified but Modal only supports one "
                    "GPU type. Using the first GPU type."
                )
            gpu_type = self.task_env_config.gpu_types[0]
        return f"{gpu_type}:{self._effective_gpus}"

    def _secrets_config(self) -> list:
        secrets = [Secret.from_name(secret) for secret in self._secrets]
        # Inject resolved [environment.env] from task.toml into the sandbox
        if self._persistent_env:
            secrets.append(
                Secret.from_dict(dict[str, str | None](self._persistent_env))
            )
        return secrets

    def _volumes_config(self) -> dict[str, Volume]:
        return {
            mount_path: Volume.from_name(volume_name)
            for mount_path, volume_name in self._volumes.items()
        }

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(
        self,
        *,
        block_network: bool | None = None,
        experimental_options: dict[str, Any] | None = None,
        command_args: list[str] | None = None,
        encrypted_ports: list[int] | None = None,
        region: str | None = None,
    ) -> Sandbox:
        """Create a sandbox with retry logic for transient failures.

        ``command_args`` are passed as the sandbox's main process (positional
        args to ``Sandbox.create``); empty means a keepalive sandbox. Serve mode
        uses ``encrypted_ports`` to open a public HTTPS tunnel on the service
        port (resolved later by ``endpoint()``) and optionally pins ``region``.
        """
        if block_network is None:
            block_network = not self.task_env_config.allow_internet

        kwargs: dict[str, Any] = {}
        if experimental_options:
            kwargs["experimental_options"] = experimental_options
        if (cpu := self._cpu_config()) is not None:
            kwargs["cpu"] = cpu
        if (memory := self._memory_config()) is not None:
            kwargs["memory"] = memory
        if (gpu := self._gpu_config()) is not None:
            kwargs["gpu"] = gpu
        if encrypted_ports:
            kwargs["encrypted_ports"] = encrypted_ports
            # Modal rejects ``block_network=True`` together with open ports: a
            # served sandbox must keep its inbound encrypted-port tunnel
            # reachable (that URL is how the load driver reaches it). Realize a
            # "sealed" served sandbox as inbound-open + outbound-denied instead:
            # an empty outbound allowlist permits no egress CIDRs (deny all),
            # preserving the trust boundary (the artifact can't call out /
            # exfiltrate) while remaining reachable.
            if block_network:
                block_network = False
                kwargs["outbound_cidr_allowlist"] = []
        if region is not None:
            kwargs["region"] = region
        # Seal the sandbox to an explicit outbound allowlist (e.g. only the egress
        # proxy / model gateway), by CIDR and/or by domain name. Modal forbids
        # ``block_network=True`` with an allowlist, so any allowlist forces
        # block_network False -- egress is restricted to these destinations rather
        # than blocked entirely. This overrides the empty deny-all set above for
        # served sandboxes: a sealed producer must still reach its proxy/gateway.
        if self._launch_outbound_cidr_allowlist:
            block_network = False
            kwargs["outbound_cidr_allowlist"] = list(
                self._launch_outbound_cidr_allowlist
            )
        if self._launch_outbound_domain_allowlist:
            block_network = False
            kwargs["outbound_domain_allowlist"] = list(
                self._launch_outbound_domain_allowlist
            )

        return await Sandbox.create.aio(
            *(command_args or []),
            app=self._app,
            image=self._image,
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            name=self.session_id,
            block_network=block_network,
            secrets=self._secrets_config(),
            volumes=self._volumes_config(),  # type: ignore[arg-type]
            **kwargs,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _terminate_sandbox(self) -> None:
        """Terminate sandbox with retry logic."""
        if self._sandbox:
            await self._sandbox.terminate.aio()

    async def _sdk_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash",
        login: bool = False,
        client_timeout_secs: float | None = None,
    ) -> ExecResult:
        """Execute a command directly on the Modal sandbox VM.

        This is the low-level exec that talks to the Modal SDK.  Strategies
        should use this (or the public ``exec``) rather than calling
        ``sandbox.exec.aio`` directly.

        Args:
            shell: Shell to use (``"bash"`` for standard images,
                ``"sh"`` for Alpine-based images like docker:dind).
            login: If True, use a login shell (``-lc``) so that
                ``.bashrc``, ``.profile``, etc. are sourced.
            client_timeout_secs: Optional client-side deadline bounding the
                local SDK await sequence (exec + stdout/stderr read + wait).
                This is distinct from ``timeout_sec``: Modal's exec ``timeout=``
                bounds the REMOTE process, whereas this bounds the LOCAL SDK
                awaits so a stalled gRPC stream cannot hang forever. ``None``
                (the default) sets no client-side deadline.
        """
        # Merge persistent env vars (--ae flags) with per-exec env vars
        env = self._merge_env(env)

        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        async def _run() -> ExecResult:
            process = await self._sandbox.exec.aio(
                shell,
                "-lc" if login else "-c",
                command,
                workdir=cwd,
                secrets=[Secret.from_dict(env)] if env else [],  # type: ignore
                timeout=timeout_sec,
            )

            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            return_code = await process.wait.aio()

            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
            )

        if client_timeout_secs is None:
            return await _run()
        return await asyncio.wait_for(_run(), timeout=client_timeout_secs)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_upload_file(self, source_path: Path | str, target_path: str) -> None:
        """
        Adds a local file to the environment.

        Args:
            source_path: The path to the source local file.
            target_path: The path to which to copy the file.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Creates parent directories on the remote side if needed.
        await self._sandbox.filesystem.copy_from_local.aio(source_path, target_path)

    async def _sdk_upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """
        Adds a local directory to the environment.

        The Modal filesystem API has no directory primitive, so instead of one
        round-trip per file this packs the directory's contents into a single
        uncompressed tar on the host, uploads that one tarball, and extracts it
        on the sandbox: two Modal round-trips (upload + extract exec) plus the
        target ``mkdir``, regardless of file count.  tar preserves modes and
        symlinks.

        Args:
            source_dir: The path to the source local directory.
            target_dir: The path to which to copy the directory.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")

        shell = self._default_shell
        await self._sdk_exec(f"mkdir -p {shlex.quote(target_dir)}", shell=shell)

        # Pack contents with arcnames relative to the source root so that
        # extraction with ``-C target_dir`` lands files directly in the target.
        # An empty source dir yields an empty tar and the target (already
        # created above) is left empty -- a valid no-op.
        fd, host_tar = tempfile.mkstemp(suffix=".tar")
        os.close(fd)
        try:
            with tarfile.open(host_tar, "w") as tar:
                for entry in sorted(source_path.iterdir()):
                    tar.add(entry, arcname=entry.name)

            remote_tar = f"/tmp/harbor-upload-{uuid4().hex}.tar"
            await self._sdk_upload_file(host_tar, remote_tar)

            # _sdk_exec returns a nonzero return_code rather than raising, so
            # check it explicitly and fail loud with the captured stderr.
            quoted_tar = shlex.quote(remote_tar)
            result = await self._sdk_exec(
                f"tar -xf {quoted_tar} -C {shlex.quote(target_dir)} "
                f"&& rm -f {quoted_tar}",
                shell=shell,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"Failed to extract {remote_tar} into {target_dir} "
                    f"(rc={result.return_code}): {result.stderr}"
                )
        finally:
            os.remove(host_tar)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        """
        Downloads a file from the environment to the local machine.

        Args:
            source_path: The path to the source file in the environment.
            target_path: The local path to which to copy the file.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Creates parent directories locally if needed. The SDK call has no
        # deadline parameter and a stalled stream never raises, so bound it here;
        # the raised TimeoutError then flows through @retry.
        await asyncio.wait_for(
            self._sandbox.filesystem.copy_to_local.aio(source_path, target_path),
            timeout=_FILE_COPY_TIMEOUT_SECS,
        )

    async def _sdk_download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """
        Downloads a directory from the environment to the local machine. This overwrites
        existing files in the target directory.

        Args:
            source_dir: The path to the source directory in the environment.
            target_dir: The local path to which to copy the directory.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Run find on the sandbox VM directly via exec_on_vm, not through
        # the strategy's exec (which in DinD mode routes to the compose
        # container, not the sandbox filesystem).
        result = await self._strategy.exec_on_vm(
            f"find {shlex.quote(source_dir)} -type f",
            shell="sh",
            client_timeout_secs=_DIR_LIST_TIMEOUT_SECS,
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to list files in {source_dir}: {result.stderr}")

        if not result.stdout or not result.stdout.strip():
            return

        file_paths = [p for p in result.stdout.strip().split("\n") if p.strip()]

        sem = asyncio.Semaphore(5)

        async def _download_one(remote_path: str) -> None:
            async with sem:
                rel = Path(remote_path).relative_to(Path(source_dir))
                local_path = target_dir / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                await self._sdk_download_file(remote_path, local_path)

        async with asyncio.TaskGroup() as tg:
            for p in file_paths:
                tg.create_task(_download_one(p))

    async def start(self, force_build: bool) -> None:
        return await self._strategy.start(force_build)

    async def stop(self, delete: bool):
        return await self._strategy.stop(delete)

    async def container_logs(self, service: str = "main") -> str:
        """Capture and return stdout+stderr of the named service.

        For serve/direct mode (single Modal sandbox): terminates the sandbox so
        its output streams reach EOF, then reads and returns them.  The sandbox
        is stopped as a side-effect — a subsequent ``stop()`` call is a safe
        no-op.

        For DinD/compose mode: reads ``docker compose logs <service>`` while
        the compose stack is still running; the caller must call ``stop()``
        separately to tear down the sandbox.
        """
        return await self._strategy.container_logs(service)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        env = self._merge_env(env)

        if user is not None:
            # Modal doesn't support user= on exec; wrap with su.
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(user))
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"

        effective_cwd = cwd or self.task_env_config.workdir
        return await self._strategy.exec(
            command, cwd=effective_cwd, env=env, timeout_sec=timeout_sec
        )

    async def upload_file(self, source_path: Path | str, target_path: str):
        return await self._strategy.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        return await self._strategy.upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str):
        return await self._strategy.download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        return await self._strategy.download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return await self._strategy.is_dir(path, user=self._resolve_user(user))

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return await self._strategy.is_file(path, user=self._resolve_user(user))

    async def attach(self) -> None:
        return await self._strategy.attach()

    def endpoint(self, container_port: int) -> str:
        """Return the public HTTPS tunnel URL for an encrypted service port.

        Serve mode opens ``encrypted_ports=[service_port]`` at sandbox-create
        time; Modal exposes each as a public tunnel. Resolves (and caches) the
        sandbox's tunnel map and returns ``tunnels[container_port].url``. Raises
        if the sandbox is not started or the port has no tunnel.
        """
        if not self._sandbox:
            raise RuntimeError(
                "Cannot resolve endpoint: the Modal sandbox is not started. "
                "Call start() first."
            )
        if self._tunnels is None:
            sandbox = self._sandbox
            self._tunnels = _run_blocking(
                lambda: sandbox.tunnels.aio(_MODAL_TUNNEL_TIMEOUT_SEC)
            )

        tunnel = self._tunnels.get(container_port)
        if tunnel is None:
            available = sorted(self._tunnels)
            raise RuntimeError(
                f"No Modal tunnel for port {container_port} "
                f"(available tunnelled ports: {available}). The port must be "
                "declared via LaunchSpec.service_port before start()."
            )
        return tunnel.url

    def commit_checkpoint(
        self, tag: str, entrypoint: list[str] | None = None
    ) -> object:
        """Snapshot the sandbox filesystem into a reusable ``modal.Image``.

        Captures the entire filesystem (agent-written files + installed deps),
        mirroring ``docker commit`` on the local backend. Returns the
        ``modal.Image`` handle, which ``ModalEnvironment.launch`` accepts back as
        ``LaunchSpec.image`` for a fresh serve sandbox.

        ``tag`` and ``entrypoint`` are ignored on Modal: the snapshot does not
        bake an ENTRYPOINT — the serve command is supplied at the next launch
        (HLS carries ``serve_command`` separately). Gated by ``can_commit``.
        """
        if not self.capabilities.can_commit:
            raise ValueError(
                f"{self.type()} environment cannot commit a checkpoint "
                "(can_commit=False)."
            )
        if not self._sandbox:
            raise RuntimeError(
                "Cannot commit checkpoint: the Modal sandbox is not started. "
                "Call start() first."
            )
        sandbox = self._sandbox
        return _run_blocking(lambda: sandbox.snapshot_filesystem.aio())
