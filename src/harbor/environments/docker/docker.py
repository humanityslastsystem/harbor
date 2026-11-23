import asyncio
import asyncio.subprocess
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.environments.base import BaseEnvironment, ExecResult

if TYPE_CHECKING:
    from harbor.environments.launch import EgressGatewaySpec, LaunchSpec
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.docker import (
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    COMPOSE_WINDOWS_KEEPALIVE_PATH,
    RESOURCES_COMPOSE_NAME,
    write_command_compose_file,
    write_egress_gateway_compose_file,
    write_env_compose_file,
    write_mounts_compose_file,
    write_network_compose_file,
    write_resources_compose_file,
)
from harbor.environments.docker.compose_env import (
    ComposeInfraEnvVars,
    legacy_log_mount_env_vars,
    merge_compose_env,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig, TaskOS
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import TrialPaths
from harbor.utils.env import resolve_env_vars


def _sanitize_docker_image_name(name: str) -> str:
    """
    Sanitize a name to be a valid Docker image name.

    See: https://github.com/opencontainers/distribution-spec/blob/5e57cc0a07ea002e507a65d4757e823f133fcb52/spec.md#pulling-manifests
    """
    # Convert to lowercase
    name = name.lower()
    # If the first character is not alphanumeric, prepend '0'
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    # Replace any character that is not a-z, 0-9, ., _, - with -
    # Note: / is not allowed here because we want only one directory hierarchy.
    name = re.sub(r"[^a-z0-9._-]", "-", name)
    return name


def _sanitize_docker_compose_project_name(name: str) -> str:
    """
    Sanitize a name to be a valid Docker Compose project name.

    See: https://docs.docker.com/compose/how-tos/project-name/
    """
    # Convert to lowercase
    name = name.lower()
    # If the first character is not alphanumeric, prepend '0'
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    # Replace any character that is not a-z, 0-9, -, or _ with -
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    return name


class DockerEnvironment(BaseEnvironment):
    _DOCKER_COMPOSE_BUILD_PATH = COMPOSE_BUILD_PATH
    _DOCKER_COMPOSE_PREBUILT_PATH = COMPOSE_PREBUILT_PATH
    _DOCKER_COMPOSE_NO_NETWORK_PATH = COMPOSE_NO_NETWORK_PATH

    _DOCKER_COMPOSE_WINDOWS_KEEPALIVE_PATH = COMPOSE_WINDOWS_KEEPALIVE_PATH

    # Class-level lock per image name to prevent parallel builds of the same image.
    _image_build_locks: dict[str, asyncio.Lock] = {}

    # Marks the lightweight, trial-free construction path (see ``launch``). Only
    # that path has no on-disk Dockerfile/compose, so only that path may skip
    # definition validation. Defaults False so the normal trial-driven
    # ``__init__`` path always validates -- set as a class attribute so
    # it exists when ``_validate_definition`` runs during ``super().__init__``
    # even before the instance attribute is assigned. Exposed via ``is_prebuilt``.
    _prebuilt: bool = False

    # Selects the sealed-egress override used when ``allow_internet`` is False.
    # The trial-driven ``__init__`` path keeps ``network_mode: none``
    # (no network namespace at all -- unreachable by anyone), so this
    # defaults False. Only the lightweight ``launch`` path sets it True, opting
    # into the ``hls_internal`` internal-network override that blocks outbound
    # internet while staying host-reachable (needed by EVAL servers). Declared as
    # a class attribute so it exists if read during construction before the
    # instance attribute is assigned.
    _internal_network_when_sealed: bool = False

    # Service-mode (EVAL) command override. When set to a non-None list the env
    # writes a compose override that REPLACES the base prebuilt/build keepalive
    # command (``sh -c sleep infinity``) with these args, so the image's own
    # server ENTRYPOINT runs as the long-lived process. Defaults None so the
    # trial-driven path and idle prebuilt envs keep the keepalive unchanged.
    # Declared as a class attribute so it exists if read during construction
    # before the instance attribute is assigned.
    _service_command: list[str] | None = None

    # Service-container environment override. When set to a non-None dict the
    # env writes a compose override declaring ``services.main.environment`` so
    # these vars reach the long-lived service process (the base prebuilt/build
    # compose has no ``environment:`` block, and per-exec env doesn't apply to
    # the service process). Defaults None so the trial-driven path and plain
    # prebuilt envs inject nothing. Declared as a class attribute so it exists
    # if read during construction before the instance attribute is assigned.
    _service_env: dict[str, str] | None = None

    # Specific GPU device IDs to pin (e.g. ["1", "2", "3"]) from the allocator.
    # When set, the compose resource override uses ``device_ids`` instead of
    # ``count``, guaranteeing the container lands on exactly those devices.
    # Defaults None so the gpu_count path applies.
    _gpu_device_ids: list[str] | None = None

    # Egress-gateway sidecar (see ``launch``). When set, the env appends the gateway
    # compose override: ``main`` joins the gateway's netns (``network_mode: service``)
    # and the gateway transparently intercepts/filters all egress. Defaults None so the
    # trial-driven path and plain prebuilt envs add no sidecar. Declared as a class
    # attribute so it exists if read during construction before the instance attribute
    # is set.
    _egress_gateway: "EgressGatewaySpec | None" = None

    @staticmethod
    def _detect_daemon_os() -> str | None:
        """Return the Docker daemon's OSType (e.g. 'linux' or 'windows'), or None on error."""
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.OSType}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            value = result.stdout.strip().lower()
            return value or None
        except Exception:
            return None

    @staticmethod
    def _detect_windows_containers() -> bool:
        """Detect if Docker is running in Windows container mode.

        Retained for back-compat with existing test fixtures.  New code should
        rely on :attr:`os` derived from ``task.toml``'s ``[environment].os``
        field; this helper is now used only for daemon-mode validation.
        """
        if sys.platform != "win32":
            return False
        return DockerEnvironment._detect_daemon_os() == "windows"

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which("docker"):
            raise SystemExit(
                "Docker is not installed or not on PATH. "
                "Please install Docker and try again."
            )
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise SystemExit(
                "Docker daemon is not running. Please start Docker and try again."
            )

    @classmethod
    def launch(cls, spec: "LaunchSpec") -> "DockerEnvironment":
        """Construct a DockerEnvironment that runs a prebuilt image directly.

        This is the lightweight, non-trial construction path: no on-disk
        Dockerfile/compose and no trial scaffolding are required. The image
        itself is the environment definition. The trial-driven ``__init__``
        path is unaffected. The spec is validated against this backend's
        capabilities (fail loud on unsupported features) before returning.
        """
        import uuid

        if not isinstance(spec.image, str):
            raise ValueError(
                "DockerEnvironment.launch requires a registry image tag (str); "
                f"got {type(spec.image).__name__}. Image handles produced by another "
                "backend's commit_checkpoint cannot be launched on Docker."
            )

        task_env_config = EnvironmentConfig(
            docker_image=spec.image,
            gpus=spec.gpus or None,
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
            allow_internet=spec.allow_internet,
            workdir=spec.workdir,
        )

        # The trial-driven path roots paths/dirs in the trial output directory.
        # With no trial, root everything in a throwaway temp directory so the
        # env is self-contained and never touches an on-disk task definition.
        # The directory is intentionally not cleaned up here: callers may rely
        # on it for the env's lifetime, and the OS reclaims temp space.
        environment_dir = Path(tempfile.mkdtemp(prefix="hb-launch-"))
        trial_paths = TrialPaths(trial_dir=environment_dir)

        short_id = uuid.uuid4().hex[:8]
        env = cls(
            environment_dir=environment_dir,
            environment_name=f"launch-{short_id}",
            session_id=f"launch-{short_id}",
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            mounts=list(spec.mounts) if spec.mounts else None,
            _prebuilt=True,
            _internal_network_when_sealed=True,
            _service_command=list(spec.serve_command) if spec.serve_command else None,
            _service_env=dict(spec.env) if spec.env else None,
            _gpu_device_ids=list(spec.gpu_device_ids) if spec.gpu_device_ids else None,
        )
        env._validate_launch_capabilities(spec)
        # Store the shared eval network name so start() can connect the container.
        env._shared_network = getattr(spec, "shared_network", None)
        # Store the egress-gateway sidecar so start()/_docker_compose_paths can append
        # the netns-sharing override that seals ``main`` behind it.
        env._egress_gateway = spec.egress_gateway
        return env

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        keep_containers: bool = False,
        *args,
        _prebuilt: bool = False,
        _internal_network_when_sealed: bool = False,
        _service_command: list[str] | None = None,
        _service_env: dict[str, str] | None = None,
        _gpu_device_ids: list[str] | None = None,
        **kwargs,
    ):
        # Set before ``super().__init__`` because the base ``__init__`` calls
        # ``_validate_definition`` (which reads this flag) before returning.
        self._prebuilt = _prebuilt
        # Set before any compose-path computation reads it. Only the prebuilt
        # path passes True; the trial path keeps the False class default so its
        # sealed containers continue to use ``network_mode: none``.
        self._internal_network_when_sealed = _internal_network_when_sealed
        # Service-mode command override (None = keep keepalive). Read by
        # ``start``/``_docker_compose_paths``; set before ``super().__init__``
        # for consistency with the other construction markers.
        self._service_command = list(_service_command) if _service_command else None
        # Service-container env override (None = inject nothing). Read by
        # ``start``/``_docker_compose_paths``; set before ``super().__init__``
        # for consistency with the other construction markers.
        self._service_env = dict(_service_env) if _service_env else None
        # Specific GPU device IDs from the allocator. When set, the compose
        # resource override uses ``device_ids`` instead of ``count``.
        self._gpu_device_ids = list(_gpu_device_ids) if _gpu_device_ids else None
        # Shared eval network name. ``launch`` sets it post-construction when the
        # container must join an existing network; the trial path keeps None.
        # Declared here so it is part of the instance type, not set only dynamically.
        self._shared_network: str | None = None
        # Egress-gateway sidecar. ``launch`` sets it post-construction; the trial
        # path keeps None. Declared here so it is part of the instance type.
        self._egress_gateway: "EgressGatewaySpec | None" = None
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._keep_containers = keep_containers
        self._is_windows_container = task_env_config.os == TaskOS.WINDOWS
        self._mounts_compose_temp_dir: tempfile.TemporaryDirectory | None = None
        self._mounts_compose_path: Path | None = None
        self._resources_compose_temp_dir: tempfile.TemporaryDirectory | None = None
        self._resources_compose_path: Path | None = None
        self._network_compose_temp_dir: tempfile.TemporaryDirectory | None = None
        self._network_compose_path: Path | None = None
        self._command_compose_temp_dir: tempfile.TemporaryDirectory | None = None
        self._command_compose_path: Path | None = None
        self._env_compose_temp_dir: tempfile.TemporaryDirectory | None = None
        self._env_compose_path: Path | None = None
        self._egress_compose_temp_dir: tempfile.TemporaryDirectory | None = None
        self._egress_compose_path: Path | None = None

        # Select the platform-specific file-transfer and exec helpers.
        if self._is_windows_container:
            import uuid

            from harbor.environments.docker.docker_windows import WindowsOps

            self._windows_container_name = f"harbor-{uuid.uuid4().hex[:12]}"
            self._platform = WindowsOps(self, self._windows_container_name)
        else:
            from harbor.environments.docker.docker_unix import UnixOps

            self._windows_container_name: str | None = None
            self._platform = UnixOps(self)

        self._env_vars = ComposeInfraEnvVars(
            main_image_name=_sanitize_docker_image_name(f"hb__{environment_name}"),
            context_dir=str(self.environment_dir.resolve().absolute()),
            prebuilt_image_name=task_env_config.docker_image,
            cpus=self._effective_cpus,
            memory=f"{memory_mb}M"
            if (memory_mb := self._effective_memory_mb)
            else None,
        )
        self._use_prebuilt = False

        self._compose_task_env: dict[str, str] = {}
        if task_env_config.env and self._uses_compose:
            self._compose_task_env = resolve_env_vars(task_env_config.env)

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def _uses_compose(self) -> bool:
        return self._environment_docker_compose_path.exists() or bool(
            self.extra_docker_compose_paths
        )

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            gpus=True,  # GPU via Compose device reservation; requires NVIDIA Container Toolkit
            disable_internet=True,
            windows=True,
            mounted=True,
            docker_compose=True,
            serves=True,  # publishes a container port; endpoint() resolves it
            can_commit=True,  # docker commit snapshots the running container
            gpu_pinning=True,  # device_ids compose override pins specific GPUs
            egress_allowlist=True,  # transparent egress gateway via a netns-sharing sidecar
        )

    @property
    def is_prebuilt(self) -> bool:
        return self._prebuilt

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    @property
    def _docker_compose_paths(self) -> list[Path]:
        """
        Returns the docker-compose file(s) to use.

        Two options for task authors:

        Option 1: Simple task (just Dockerfile)
        - No docker-compose needed
        - Uses: base + build/prebuilt

        Option 2: Task with extra services (docker-compose.yaml)
        - Create docker-compose.yaml with additional services or overrides
        - Uses: base + build/prebuilt + docker-compose.yaml
        - Task file is last so it can override scalars from build/prebuilt
        - Relative paths (e.g. build context) resolve relative to the file
          where they are defined, regardless of -f order

        For Windows-container tasks, a keepalive override is inserted between
        build/prebuilt and the task's own docker-compose.yaml. This lets the
        keepalive override the Linux `tail -f /dev/null` baked into
        build/prebuilt, while still allowing a Windows task's own compose
        file to override the keepalive command if it needs a different
        long-running process.

        When allow_internet is False the sealed-egress override is appended last.
        Which override depends on the construction path:

        - Lightweight ``launch`` envs (``_internal_network_when_sealed``
          True) get the ``hls_internal`` internal-network override, which blocks
          outbound internet while keeping the container reachable from the host
          (needed for EVAL environments that serve HTTP).
        - The trial-driven ``__init__`` path gets the
          ``network_mode: none`` override, which removes the network namespace
          entirely so the sealed container is unreachable by anyone.
        """
        build_or_prebuilt = (
            self._DOCKER_COMPOSE_PREBUILT_PATH
            if self._use_prebuilt
            else self._DOCKER_COMPOSE_BUILD_PATH
        )

        paths = []
        if self._resources_compose_path:
            paths.append(self._resources_compose_path)
        paths.append(build_or_prebuilt)

        if self._is_windows_container:
            paths.append(self._DOCKER_COMPOSE_WINDOWS_KEEPALIVE_PATH)

        if self._environment_docker_compose_path.exists():
            paths.append(self._environment_docker_compose_path)

        paths.extend(self.extra_docker_compose_paths)

        if self._mounts_compose_path:
            paths.append(self._mounts_compose_path)

        # Service-mode command override. Appended after the base prebuilt/build
        # compose (and after the windows keepalive / task compose) so Compose's
        # last-wins merge replaces the keepalive command with the server args.
        if self._command_compose_path:
            paths.append(self._command_compose_path)

        # Service-container environment override. Appended after the base
        # prebuilt/build compose so its ``services.main.environment`` block is
        # merged onto the running service process.
        if self._env_compose_path:
            paths.append(self._env_compose_path)

        # The egress gateway, when set, IS the network seal: ``main`` joins the
        # gateway sidecar's netns (``network_mode: service``), which is mutually
        # exclusive with the native-block overrides below. So the seal field is
        # AUTHORITATIVE over ``allow_internet`` -- apply the native block only when
        # there is no gateway, rather than relying on the caller to also set
        # ``allow_internet=True`` (a mis-set bool would otherwise double-seal and
        # produce a conflicting compose merge).
        if self._egress_gateway is None and not self.task_env_config.allow_internet:
            if self._internal_network_when_sealed:
                if self._network_compose_path:
                    paths.append(self._network_compose_path)
            else:
                paths.append(self._DOCKER_COMPOSE_NO_NETWORK_PATH)

        # Egress-gateway override (appended last): puts ``main`` in the gateway
        # sidecar's network namespace so the gateway transparently intercepts/filters
        # all egress. The seal is the gateway's redirect rules; ``allow_internet`` is
        # irrelevant here (see above).
        if self._egress_compose_path:
            paths.append(self._egress_compose_path)

        return paths

    def _write_mounts_compose_file(self) -> Path:
        """Write the trial mounts compose override."""
        self._cleanup_mounts_compose_file()
        self._mounts_compose_temp_dir = tempfile.TemporaryDirectory()
        path = Path(self._mounts_compose_temp_dir.name) / "docker-compose-mounts.json"
        return write_mounts_compose_file(path, list(self._mounts))

    def _write_resources_compose_file(self) -> Path | None:
        """Write the trial resource policy compose override."""
        self._cleanup_resources_compose_file()
        self._resources_compose_temp_dir = tempfile.TemporaryDirectory()
        path = (
            Path(self._resources_compose_temp_dir.name)
            / f"{self.session_id}-{RESOURCES_COMPOSE_NAME}"
        )
        return write_resources_compose_file(
            path,
            cpu_request=self._resource_request_value(
                "cpu", auto_mode=ResourceMode.LIMIT
            ),
            cpu_limit=self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT),
            memory_request_mb=self._resource_request_value(
                "memory", auto_mode=ResourceMode.LIMIT
            ),
            memory_limit_mb=self._resource_limit_value(
                "memory", auto_mode=ResourceMode.LIMIT
            ),
            gpu_count=self._effective_gpus or None,
            gpu_device_ids=self._gpu_device_ids,
        )

    def _write_network_compose_file(self) -> Path:
        """Write the internal-network compose override (sealed-egress envs).

        Only meaningful when ``allow_internet`` is False; the override attaches
        the main service to a Docker ``internal`` network so the host can still
        reach it while outbound internet is blocked.
        """
        self._cleanup_network_compose_file()
        self._network_compose_temp_dir = tempfile.TemporaryDirectory()
        path = Path(self._network_compose_temp_dir.name) / "docker-compose-network.json"
        return write_network_compose_file(path, internal=True)

    def _write_command_compose_file(self) -> Path:
        """Write the service-mode command override (replaces the keepalive).

        Only called when ``_service_command`` is set; the override declares
        ``services.main.command`` with the server args and must be ordered after
        the base prebuilt/build compose so it replaces the keepalive command.
        """
        self._cleanup_command_compose_file()
        self._command_compose_temp_dir = tempfile.TemporaryDirectory()
        path = Path(self._command_compose_temp_dir.name) / "docker-compose-command.json"
        return write_command_compose_file(path, list(self._service_command or []))

    def _write_env_compose_file(self) -> Path:
        """Write the service-container environment override.

        Called when ``_service_env`` is set; the override declares
        ``services.main.environment`` and must be ordered after the base
        prebuilt/build compose so its env reaches the long-lived service process.
        """
        self._cleanup_env_compose_file()
        self._env_compose_temp_dir = tempfile.TemporaryDirectory()
        path = Path(self._env_compose_temp_dir.name) / "docker-compose-env.json"
        return write_env_compose_file(path, dict(self._service_env or {}))

    def _write_egress_gateway_compose_file(self) -> Path:
        """Write the egress-gateway sidecar override.

        Only called when ``_egress_gateway`` is set; the override defines the gateway
        service and puts ``main`` in its network namespace (``network_mode: service``)
        so the gateway transparently intercepts and filters all of ``main``'s egress.
        """
        self._cleanup_egress_compose_file()
        self._egress_compose_temp_dir = tempfile.TemporaryDirectory()
        path = Path(self._egress_compose_temp_dir.name) / "docker-compose-egress.json"
        if self._egress_gateway is None:
            raise RuntimeError(
                "_write_egress_gateway_compose_file called without an egress gateway."
            )
        return write_egress_gateway_compose_file(path, gateway=self._egress_gateway)

    def _cleanup_mounts_compose_file(self) -> None:
        if self._mounts_compose_temp_dir is None:
            return

        try:
            self._mounts_compose_temp_dir.cleanup()
        except OSError as e:
            self.logger.debug(f"Failed to remove mounts compose file: {e}")
        finally:
            self._mounts_compose_temp_dir = None
            self._mounts_compose_path = None

    def _cleanup_resources_compose_file(self) -> None:
        if self._resources_compose_temp_dir is None:
            return

        try:
            self._resources_compose_temp_dir.cleanup()
        except OSError as e:
            self.logger.debug(f"Failed to remove resources compose file: {e}")
        finally:
            self._resources_compose_temp_dir = None
            self._resources_compose_path = None

    def _cleanup_network_compose_file(self) -> None:
        if self._network_compose_temp_dir is None:
            return

        try:
            self._network_compose_temp_dir.cleanup()
        except OSError as e:
            self.logger.debug(f"Failed to remove network compose file: {e}")
        finally:
            self._network_compose_temp_dir = None
            self._network_compose_path = None

    def _cleanup_command_compose_file(self) -> None:
        if self._command_compose_temp_dir is None:
            return

        try:
            self._command_compose_temp_dir.cleanup()
        except OSError as e:
            self.logger.debug(f"Failed to remove command compose file: {e}")
        finally:
            self._command_compose_temp_dir = None
            self._command_compose_path = None

    def _cleanup_env_compose_file(self) -> None:
        if self._env_compose_temp_dir is None:
            return

        try:
            self._env_compose_temp_dir.cleanup()
        except OSError as e:
            self.logger.debug(f"Failed to remove env compose file: {e}")
        finally:
            self._env_compose_temp_dir = None
            self._env_compose_path = None

    def _cleanup_egress_compose_file(self) -> None:
        if self._egress_compose_temp_dir is None:
            return

        try:
            self._egress_compose_temp_dir.cleanup()
        except OSError as e:
            self.logger.debug(f"Failed to remove egress compose file: {e}")
        finally:
            self._egress_compose_temp_dir = None
            self._egress_compose_path = None

    @property
    def _main_image_name(self) -> str:
        return self._env_vars.main_image_name

    def _compose_infra_env_vars(self) -> dict[str, str]:
        env_vars = self._env_vars.to_env_dict(include_os_env=False)
        if not self._use_prebuilt:
            env_vars.pop("PREBUILT_IMAGE_NAME", None)
        env_vars.update(legacy_log_mount_env_vars(self._mounts, host_value="source"))
        return env_vars

    def _compose_env_vars(self, include_os_env: bool = True) -> dict[str, str]:
        user_env: dict[str, str] = {}
        if self._compose_task_env:
            user_env.update(self._compose_task_env)
        if self._persistent_env:
            user_env.update(self._persistent_env)

        infra = self._compose_infra_env_vars()
        env_vars = merge_compose_env(
            base_env=os.environ if include_os_env else None,
            user_env=user_env,
            infra_env=infra,
            logger=self.logger,
        )
        # Inject after user env so it cannot be accidentally overridden.
        if self._windows_container_name:
            env_vars["HARBOR_CONTAINER_NAME"] = self._windows_container_name
        return env_vars

    def _validate_definition(self):
        # Only the lightweight `launch` construction path is prebuilt: the image
        # itself is the complete environment, so there is no on-disk
        # Dockerfile/compose to validate. Note that a truthy
        # `task_env_config.docker_image` does NOT imply prebuilt -- a trial task
        # may set both `[environment].docker_image` and `force_build=True`, in
        # which case `start()` sets `_use_prebuilt=False` and runs
        # `docker compose build`, which still requires an on-disk definition. So
        # the trial path must keep failing fast here.
        if self.is_prebuilt:
            return
        if (
            not self._dockerfile_path.exists()
            and not self._environment_docker_compose_path.exists()
            and not self.extra_docker_compose_paths
        ):
            raise FileNotFoundError(
                f"{self._dockerfile_path} and {self._environment_docker_compose_path} "
                "not found, and no extra Docker Compose files were provided. "
                "Please ensure at least one environment definition exists."
            )

    async def _run_docker_compose_command(
        self, command: list[str], check: bool = True, timeout_sec: int | None = None
    ) -> ExecResult:
        """Run a docker compose command and return the result."""
        full_command = [
            "docker",
            "compose",
            "--project-name",
            _sanitize_docker_compose_project_name(self.session_id),
            "--project-directory",
            str(self.environment_dir.resolve().absolute()),
        ]
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(command)

        env = self._compose_env_vars(include_os_env=True)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None

        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )

        if check and result.return_code != 0:
            raise RuntimeError(
                f"Docker compose command failed for environment {self.environment_name}. "
                f"Command: {' '.join(full_command)}. "
                f"Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. "
                f"Stderr: {result.stderr}. "
            )

        return result

    def _validate_daemon_mode(self) -> None:
        """Verify the Docker daemon mode matches the task's declared OS.

        Raises ``RuntimeError`` with remediation guidance when the task
        targets Windows but Docker Desktop is in Linux container mode (or
        vice versa), or when a Windows task is launched on a non-Windows
        host.
        """
        if self._is_windows_container and sys.platform != "win32":
            raise RuntimeError(
                "Task declares [environment].os = 'windows' but the host is "
                f"not Windows ({sys.platform!r}). Windows containers require "
                "a Windows host with Docker Desktop in Windows container mode."
            )

        daemon_os = self._detect_daemon_os()
        if daemon_os is None:
            # Could not query the daemon; defer to docker compose to error.
            return

        expected = "windows" if self._is_windows_container else "linux"
        if daemon_os != expected:
            switch_to = "Windows" if expected == "windows" else "Linux"
            raise RuntimeError(
                f"Task declares [environment].os = {expected!r} but the Docker "
                f"daemon is running in {daemon_os!r} container mode. "
                f"Switch Docker Desktop to {switch_to} containers "
                "(right-click the system tray icon -> 'Switch to "
                f"{switch_to} containers...') and try again."
            )

    async def _validate_image_os(self, image_name: str) -> None:
        """Verify the Docker image's OS matches the task's declared OS.

        Runs ``docker inspect --format "{{.Os}}" <image>`` and raises
        ``RuntimeError`` on mismatch.  Silently skipped when the image cannot
        be inspected (e.g. not yet pulled in unusual edge cases).
        """
        try:
            result = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "--format",
                "{{.Os}}",
                image_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
        except Exception as e:
            self.logger.debug(f"Skipping image OS validation for {image_name}: {e}")
            return

        if result.returncode != 0:
            self.logger.debug(
                f"Skipping image OS validation for {image_name}: "
                f"docker inspect returned {result.returncode}"
            )
            return

        image_os = stdout.decode("utf-8", errors="replace").strip().lower()
        expected = "windows" if self._is_windows_container else "linux"
        if image_os and image_os != expected:
            raise RuntimeError(
                f"Task declares [environment].os = {expected!r} but Docker image "
                f"{image_name!r} reports OS {image_os!r}. Use a "
                f"{expected}-compatible base image, or update [environment].os "
                "in task.toml to match the image."
            )

    async def start(self, force_build: bool):
        # Volume declarations always come from the runtime override now --
        # the static base compose declares none. Write before any compose
        # command runs.
        self._mounts_compose_path = self._write_mounts_compose_file()
        self._resources_compose_path = self._write_resources_compose_file()
        if (
            not self.task_env_config.allow_internet
            and self._internal_network_when_sealed
        ):
            self._network_compose_path = self._write_network_compose_file()
        if self._service_command is not None:
            self._command_compose_path = self._write_command_compose_file()
        if self._service_env is not None:
            self._env_compose_path = self._write_env_compose_file()
        if self._egress_gateway is not None:
            self._egress_compose_path = self._write_egress_gateway_compose_file()

        self._use_prebuilt = not force_build and self.task_env_config.docker_image

        # Fail fast if the daemon mode disagrees with the task's declared OS.
        self._validate_daemon_mode()

        if not self._use_prebuilt:
            # Serialize image builds: if multiple environments with the same image name
            # start concurrently, only one builds while others wait for the cached image.
            lock = self._image_build_locks.setdefault(
                self.environment_name, asyncio.Lock()
            )
            async with lock:
                await self._run_docker_compose_command(["build"])

        # Validate image OS after build/pull but before container start.
        image_to_check = (
            self.task_env_config.docker_image
            if self._use_prebuilt
            else self._main_image_name
        )
        if image_to_check:
            await self._validate_image_os(image_to_check)

        # Remove any stale containers from previous runs with the same session ID.
        try:
            await self._run_docker_compose_command(["down", "--remove-orphans"])
        except RuntimeError:
            pass

        await self._run_docker_compose_command(["up", "--detach", "--wait"])

        # Connect the netns holder to the shared eval network (if set) so that
        # containers from different compose projects can communicate, such as a
        # driver container reaching the artifact container's service port within
        # the same eval run.  When sealed behind the
        # egress gateway, ``main`` shares the gateway's netns and has no NIC of its
        # own, so the GATEWAY is connected (it holds the served port); a container in
        # ``network_mode: service`` cannot itself be attached to a network.
        shared_net = getattr(self, "_shared_network", None)
        if shared_net:
            container_id = self._compose_ps_quiet(self._network_holder())
            if container_id:
                try:
                    subprocess.run(
                        ["docker", "network", "connect", shared_net, container_id],
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    pass  # Non-fatal: the container may already be on this network.

        # Auto-create + chmod each writable mount target inside the container.  Bind
        # mounts auto-create the target as part of the mount, so mkdir is
        # generally a no-op, but chmod makes the dir writable for non-root
        # agent/verifier users.  Skipped for Windows containers which use
        # ACLs rather than POSIX permissions.
        if not self._is_windows_container:
            await self.ensure_dirs(self._mount_targets(writable_only=True))

    async def prepare_logs_for_host(self) -> None:
        """Chown the bind-mounted logs directory to the host user.

        On Linux, files created inside the container are owned by the agent
        UID.  The host process (which may run as a different UID) cannot read
        them until ownership is corrected.  This is a no-op on macOS/Windows
        where Docker Desktop's VM layer handles ownership transparently.
        """
        try:
            for target in self._mount_targets(writable_only=True):
                await self._chown_to_host_user(target, recursive=True)
        except Exception as e:
            self.logger.warning(f"Failed to chown logs directory: {e}")

    async def stop(self, delete: bool):
        try:
            # Best-effort: fix ownership of bind-mounted directories so the host
            # user can read/write/delete them after the container is gone.
            await self.prepare_logs_for_host()

            if self._keep_containers and delete:
                self.logger.warning(
                    "Both `keep_containers` and `--delete` option are set. "
                    "keep_containers takes precedence."
                )
            if self._keep_containers:
                try:
                    await self._run_docker_compose_command(["stop"])
                except Exception as e:
                    self.logger.warning(f"Docker compose stop failed: {e}")
            elif delete:
                try:
                    await self._run_docker_compose_command(
                        ["down", "--rmi", "all", "--volumes", "--remove-orphans"]
                    )
                except Exception as e:
                    self.logger.warning(f"Docker compose down failed: {e}")
            else:
                try:
                    await self._run_docker_compose_command(["down"])
                except Exception as e:
                    self.logger.warning(f"Docker compose down failed: {e}")
        finally:
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
            self._cleanup_network_compose_file()
            self._cleanup_command_compose_file()
            self._cleanup_env_compose_file()
            self._cleanup_egress_compose_file()

    def _compose_ps_quiet(self, service: str) -> str:
        """Return the container id of *service* via ``docker compose ps -q``.

        Uses the same project name and compose -f list as
        ``_run_docker_compose_command`` so it resolves the running container
        for this session. Returns an empty string when the service has no
        container.
        """
        full_command = [
            "docker",
            "compose",
            "--project-name",
            _sanitize_docker_compose_project_name(self.session_id),
            "--project-directory",
            str(self.environment_dir.resolve().absolute()),
        ]
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(["ps", "-q", service])

        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            env=self._compose_env_vars(include_os_env=True),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to resolve container id for service {service!r} in "
                f"environment {self.environment_name}: {result.stderr.strip()}"
            )
        return (
            result.stdout.strip().splitlines()[0].strip()
            if result.stdout.strip()
            else ""
        )

    def _network_holder(self) -> str:
        """The compose service that owns ``main``'s network namespace, identity, and
        served port. Normally ``main`` itself; when sealed behind the egress gateway
        ``main`` joins the gateway's netns (``network_mode: service``) with no NIC of its
        own, so the GATEWAY holds the eval-network attachment + the port -- siblings reach
        the served artifact through the gateway's address, and main's egress stays
        transparently filtered by the gateway's redirect rules (inbound is untouched)."""
        return "egress-gateway" if self._egress_gateway is not None else "main"

    def endpoint(self, container_port: int) -> str:
        """Return a host-reachable ``http://<ip>:<port>`` URL for the served container.

        Resolves the netns holder's IP on its attached Docker network and builds a URL
        the host can reach. For sealed EVAL environments the container sits on an
        ``internal`` network, which still allows host->container traffic. Raises if the
        container is not running or has no IP address.

        When a ``shared_network`` was set (intra-eval communication), prefers the IP on
        that specific network so sibling containers (e.g. the driver) can reach the
        artifact via its shared-network address. When sealed behind the egress gateway,
        the holder is the gateway (which owns ``main``'s netns + served port).
        """
        holder = self._network_holder()
        container_id = self._compose_ps_quiet(holder)
        if not container_id:
            raise RuntimeError(
                f"Cannot resolve endpoint: the '{holder}' container for environment "
                f"{self.environment_name} is not running."
            )

        # When a shared eval network was set, get the IP on THAT network
        # specifically so driver containers can reach it (they're on the same
        # shared network but may not be on hls_internal).
        shared_net = getattr(self, "_shared_network", None)
        if shared_net:
            inspect_fmt = (
                '{{(index .NetworkSettings.Networks "' + shared_net + '").IPAddress}}'
            )
            net_result = subprocess.run(
                ["docker", "inspect", "-f", inspect_fmt, container_id],
                capture_output=True,
                text=True,
            )
            if net_result.returncode == 0:
                shared_ip = net_result.stdout.strip()
                if shared_ip and shared_ip != "<no value>":
                    return f"http://{shared_ip}:{container_port}"

        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
                container_id,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker inspect failed for container {container_id}: "
                f"{result.stderr.strip()}"
            )

        ip = next(
            (part for part in result.stdout.split() if part.strip()),
            None,
        )
        if not ip:
            raise RuntimeError(
                f"The 'main' container ({container_id}) for environment "
                f"{self.environment_name} has no IP address on any network."
            )

        return f"http://{ip}:{container_port}"

    def container_logs(self, service: str = "main") -> str:
        """Return the combined stdout+stderr logs of a running compose service.

        Used by HLS observability to capture reference/artifact container output
        before teardown. Reads logs via 'docker logs <container_id>'.
        """
        container_id = self._compose_ps_quiet(service)
        if not container_id:
            raise RuntimeError(f"cannot read logs: service '{service}' is not running")
        result = subprocess.run(
            ["docker", "logs", container_id],
            capture_output=True,
            text=True,
        )
        return (result.stdout or "") + (result.stderr or "")

    def commit_checkpoint(self, tag: str, entrypoint: list[str] | None = None) -> str:
        """Snapshot the running ``main`` container into a new image ``tag``.

        Runs ``docker commit`` on the ``main`` container (resolved via the same
        compose project as the rest of this env). When *entrypoint* is given it
        is baked into the committed image as an exec-form ENTRYPOINT via
        ``--change``. Returns *tag*. Raises if the container is not running.
        """
        container_id = self._compose_ps_quiet("main")
        if not container_id:
            raise RuntimeError(
                f"Cannot commit image: the 'main' container for environment "
                f"{self.environment_name} is not running."
            )

        command = ["docker", "commit"]
        if entrypoint is not None:
            command.extend(["--change", f"ENTRYPOINT {json.dumps(entrypoint)}"])
        command.extend([container_id, tag])

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker commit failed for container {container_id} "
                f"(environment {self.environment_name}): {result.stderr.strip()}"
            )

        return tag

    async def image_digest(self) -> str | None:
        """Resolve the deployed image to its content id via ``docker inspect``.
        Best-effort: ``None`` if the tag is unknown or docker is unavailable."""
        image_name = self.task_env_config.docker_image
        if not image_name:
            return None
        try:
            result = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "--format",
                "{{.Id}}",
                image_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"image_digest: cannot inspect {image_name}: {e}")
            return None
        if result.returncode != 0:
            # Mirror _validate_image_os: a non-zero inspect (e.g. image not yet
            # pulled) is non-fatal, but log it so an absent digest is diagnosable.
            self.logger.debug(
                f"image_digest: docker inspect returned {result.returncode} "
                f"for {image_name}"
            )
            return None
        return stdout.decode("utf-8", errors="replace").strip() or None

    async def upload_file(self, source_path: Path | str, target_path: str):
        await self._platform.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        await self._platform.upload_dir(source_dir, target_dir)

    async def _chown_to_host_user(self, path: str, recursive: bool = False) -> None:
        """Best-effort chown of a container path to the host user's UID:GID.

        No-op on Windows (where os.getuid/os.getgid are unavailable).
        """
        if not hasattr(os, "getuid"):
            return
        flag = "-R " if recursive else ""
        await self.exec(
            f"chown {flag}{os.getuid()}:{os.getgid()} {shlex.quote(path)}", user="root"
        )

    async def download_file(self, source_path: str, target_path: Path | str):
        await self._platform.download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        await self._platform.download_dir(source_dir, target_dir)

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

        exec_command = ["exec"]

        effective_cwd = cwd or self.task_env_config.workdir
        if effective_cwd:
            exec_command.extend(["-w", effective_cwd])

        if env:
            for key, value in env.items():
                exec_command.extend(["-e", f"{key}={value}"])

        if user is not None:
            exec_command.extend(["-u", str(user)])

        exec_command.append("main")
        exec_command.extend(self._platform.exec_shell_args(command))

        return await self._run_docker_compose_command(
            exec_command, check=False, timeout_sec=timeout_sec
        )

    async def attach(self) -> None:
        if self._is_windows_container:
            raise NotImplementedError(
                "Interactive attach is not yet supported for Windows containers."
            )

        variables = " ".join(
            f"export {k}={shlex.quote(str(v))}"
            for k, v in self._compose_env_vars(include_os_env=False).items()
        )

        # Build the -f flags for docker compose
        compose_file_args = []
        for path in self._docker_compose_paths:
            compose_file_args.extend(
                ["-f", shlex.quote(str(path.resolve().absolute()))]
            )

        project_name = _sanitize_docker_compose_project_name(self.session_id)
        compose_base = [
            "docker",
            "compose",
            "--project-name",
            project_name,
        ] + compose_file_args
        cleanup_mounts_compose = (
            f"; rm -rf {shlex.quote(self._mounts_compose_temp_dir.name)}"
            if self._mounts_compose_temp_dir
            else ""
        )

        os.execvp(
            "bash",
            [
                "bash",
                "-c",
                f"{variables}; "
                + " ".join(compose_base + ["exec", "-it", "main", "bash"])
                + "; "
                + " ".join(compose_base + ["down"])
                + cleanup_mounts_compose,
            ],
        )
