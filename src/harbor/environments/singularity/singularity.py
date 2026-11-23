# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Singularity/Apptainer environment for running tasks on HPC clusters.

This environment converts Docker images to Singularity .sif format and
runs a FastAPI server inside the container to handle command execution.
"""

import asyncio
import asyncio.subprocess
import logging
import os
import shlex
import shutil
import signal
import socket
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

if sys.platform != "win32":
    import fcntl

import httpx

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

if TYPE_CHECKING:
    from harbor.environments.launch import LaunchSpec


def _proc_descendants(root_pid: int) -> list[int]:
    """``root_pid`` and every descendant of it, nearest first.

    Pids that exit while the tree is being walked are skipped.
    """
    seen: set[int] = set()
    order: list[int] = []
    queue = deque([root_pid])

    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        order.append(current)
        try:
            for tid_dir in Path(f"/proc/{current}/task").iterdir():
                try:
                    children = (tid_dir / "children").read_text().split()
                except OSError:
                    continue
                queue.extend(int(c) for c in children if c.isdigit())
        except OSError:
            continue

    return order


def _proc_stat(pid: int) -> tuple[str, int] | None:
    """The ``(state, process group id)`` of a pid, or None if it is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        # The comm field is parenthesised and may itself contain spaces and parens;
        # state, ppid and pgid are the three fields that follow it.
        state, _ppid, pgid = stat[stat.rindex(")") + 2 :].split(maxsplit=3)[:3]
        return state, int(pgid)
    except (IndexError, ValueError):
        return None


def _proc_pid_namespace(pid: int) -> tuple[int, int] | None:
    """The PID namespace of a pid, as the ``(st_dev, st_ino)`` of its ``ns/pid`` link.

    None if the pid is gone or we may not inspect it. The pair names a namespace only
    while that namespace is alive: once it dies the kernel can give its inode number to
    a new one.
    """
    try:
        st = os.stat(f"/proc/{pid}/ns/pid")
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _fd_pid_namespace(fd: int) -> tuple[int, int]:
    """The ``(st_dev, st_ino)`` of the PID namespace an open ``ns/pid`` fd refers to."""
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


def _proc_cmdline(pid: int) -> str:
    """The command line of a pid, space-joined; empty when it cannot be read."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(raw.decode(errors="replace").split("\0")).strip()


def _send_signal(target: int, sig: int, *, group: bool = False) -> None:
    """Signal a pid, or a whole process group; a target that is already gone is fine."""
    try:
        if group:
            os.killpg(target, sig)
        else:
            os.kill(target, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _describe_exit_status(rc: int) -> str:
    """What the launch process's exit status says about how the exec server ended."""
    if rc < 0:
        # Only reachable when the host signalled the launch process itself. A payload
        # death never lands here: the producer seal wraps the payload in bash, which
        # reports a signalled child as 128+N and then exits normally with that status,
        # so the negative form is destroyed before harbor sees it.
        name = signal.Signals(-rc).name if -rc in set(signal.Signals) else "?"
        return f"server killed by signal {-rc} ({name})"
    if 128 < rc < 128 + signal.NSIG:
        # So this is the only form an in-container kill ever takes. Hedged rather than
        # asserted: a program may exit 137 of its own accord, and the status alone
        # cannot tell that from SIGKILL.
        sig = rc - 128
        name = signal.Signals(sig).name if sig in set(signal.Signals) else "?"
        cause = (
            ", likely the OOM killer or a kill from inside the sandbox"
            if sig == signal.SIGKILL
            else ""
        )
        return (
            f"server exited with code {rc} (128+{sig}: likely killed by {name}{cause})"
        )
    return f"server exited with code {rc}"


class MemoryLimitExceededError(Exception):
    """Raised when a container exceeds its memory limit."""

    pass


#: How long the exec server is given to answer before the start fails, whatever the
#: poll interval (``LaunchSpec.ready_poll_sec``).
SERVER_READY_TIMEOUT_SEC = 60.0


class SingularityEnvironment(BaseEnvironment):
    """
    Singularity-based environment for HPC clusters.

    This environment:
    1. Pulls Docker images and converts them to .sif format
    2. Runs a FastAPI server inside the container for command execution
    3. Uses bind mounts for file transfer

    Kwargs (via --ek / environment.kwargs in trial config):
        singularity_image_cache_dir: Path to cache .sif files.
        singularity_force_pull: Force re-pull even if cached (default False).
        singularity_no_mount: Comma-separated mount types to suppress
            (default "home,tmp,bind-paths"). Use "" to allow all Singularity mounts.
    """

    # GPU passthrough flag. When True the container launch adds ``--nv`` so the
    # NVIDIA driver + devices visible to the job (Slurm-set ``CUDA_VISIBLE_DEVICES``)
    # are exposed inside the container. Defaults False for the trial-driven CPU path;
    # the lightweight ``launch`` path sets it from ``LaunchSpec.gpus`` /
    # ``gpu_device_ids``. Declared as a class attribute so it exists if read during
    # construction before the instance attribute is assigned.
    _nv: bool = False
    # Whether to pass ``--fakeroot``. Defaults False: on a rootless install without
    # /etc/subuid, --fakeroot makes the --writable-tmpfs overlay read-only, breaking the
    # server's setup and the agent's writes. The keepalive build sandbox runs as the
    # invoking user (writable overlay) instead. (Class attribute so it exists if read
    # during construction before the instance attribute is assigned.)
    _fakeroot: bool = False

    # Marks the lightweight, trial-free construction path (see ``launch``). Only that
    # path has no on-disk Dockerfile/.def, so only it may skip definition validation.
    # Defaults False so the trial-driven ``__init__`` path validates.
    _prebuilt: bool = False

    # Opaque argv prefix prepended to the ``singularity exec`` keepalive command so the
    # subprocess launches inside an externally-built network seal (the HLS rootless
    # producer seal: ``pasta`` + iptables). Empty -> open mode (no
    # seal). Set from ``LaunchSpec.seal_command_prefix``; a class attribute so it exists
    # if read before the instance attribute is assigned.
    _seal_command_prefix: tuple[str, ...] = ()

    # Mount a writable tmpfs at the workdir (apptainer ``--scratch``). Set under the
    # producer seal, where the container root is read-only (no overlay in a nested userns
    # on older kernels) so the agent needs a writable workdir. Class attr for the same
    # construction-order reason as above.
    _scratch_workdir: bool = False

    @classmethod
    def launch(cls, spec: "LaunchSpec") -> "SingularityEnvironment":
        """Construct a SingularityEnvironment that runs a prebuilt image directly.

        The lightweight, non-trial construction path mirroring
        ``DockerEnvironment.launch``: no on-disk task definition is required -- the
        image (a ``.sif`` path, or a ``docker://``/registry ref the env converts) is
        the environment definition. The trial-driven ``__init__`` path is unaffected.

        ``spec.gpus`` (or ``spec.gpu_device_ids``) turns on ``--nv`` GPU passthrough;
        ``spec.env`` becomes the container's persistent env (reaches exec'd processes);
        ``spec.mounts`` are bind-mounted. ``serve_command`` is not used here -- the
        Singularity backend runs a keepalive exec server (the agent drives it via
        ``exec``); HLS serves artifacts through the substrate's own apptainer launch.
        The spec is validated against this backend's capabilities before returning.
        """
        import uuid

        if not isinstance(spec.image, str):
            raise ValueError(
                "SingularityEnvironment.launch requires an image path/ref (str); "
                f"got {type(spec.image).__name__}. An image handle produced by another "
                "backend cannot be launched on Singularity."
            )
        cls._validate_sandbox_fields(spec)

        task_env_config = EnvironmentConfig(
            docker_image=spec.image,
            gpus=spec.gpus or None,
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
            allow_internet=spec.allow_internet,
            workdir=spec.workdir,
        )

        # With no trial, root everything in a throwaway temp directory so the env is
        # self-contained and never touches an on-disk task definition (mirrors Docker).
        environment_dir = Path(tempfile.mkdtemp(prefix="hb-singularity-launch-"))
        trial_paths = TrialPaths(trial_dir=environment_dir)

        short_id = uuid.uuid4().hex[:8]
        nv = bool(spec.gpus) or bool(spec.gpu_device_ids)
        env = cls(
            environment_dir=environment_dir,
            environment_name=f"launch-{short_id}",
            session_id=f"launch-{short_id}",
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            mounts=list(spec.mounts) if spec.mounts else None,
            persistent_env=dict(spec.env) if spec.env else None,
            _prebuilt=True,
            _nv=nv,
            # The HLS rootless producer seal: an argv prefix (``pasta`` + iptables
            # wrapper) the keepalive ``singularity exec`` is spawned inside.
            _seal_command_prefix=tuple(spec.seal_command_prefix),
            _scratch_workdir=spec.scratch_workdir,
            _overlay_dir=spec.overlay_dir or "",
            _lower_overlay_dirs=tuple(spec.lower_overlay_dirs),
            _server_python=spec.server_python or "",
            _start_in_staging_dir=spec.start_in_staging_dir,
            _ready_poll_sec=spec.ready_poll_sec,
        )
        env._validate_launch_capabilities(spec)
        return env

    @staticmethod
    def _validate_sandbox_fields(spec: "LaunchSpec") -> None:
        """Refuse a value of the sandbox fields that the launch could not honour."""
        if not spec.ready_poll_sec > 0:
            raise ValueError(
                f"ready_poll_sec must be positive; got {spec.ready_poll_sec!r}."
            )
        if spec.server_python is not None and (
            not os.path.isabs(spec.server_python) or "," in spec.server_python
        ):
            # apptainer splits ``--env`` values on commas.
            raise ValueError(
                "server_python must be an absolute path without commas; "
                f"got {spec.server_python!r}."
            )
        if spec.start_in_staging_dir:
            host_paths = [
                *([str(spec.image)] if str(spec.image).endswith(".sif") else []),
                *(m["source"] for m in spec.mounts if m.get("type", "bind") == "bind"),
                *([spec.overlay_dir] if spec.overlay_dir else []),
                *spec.lower_overlay_dirs,
            ]
            relative = [p for p in host_paths if not os.path.isabs(p)]
            if relative:
                raise ValueError(
                    "start_in_staging_dir runs the launch from its staging directory, "
                    f"where these relative host paths name other files: {relative}."
                )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        singularity_image_cache_dir: Path | str | None = None,
        singularity_force_pull: bool = False,
        singularity_no_mount: str | None = None,
        _prebuilt: bool = False,
        _nv: bool = False,
        _fakeroot: bool = False,
        _seal_command_prefix: tuple[str, ...] = (),
        _scratch_workdir: bool = False,
        _overlay_dir: str = "",
        _lower_overlay_dirs: tuple[str, ...] = (),
        _server_python: str = "",
        _start_in_staging_dir: bool = False,
        _ready_poll_sec: float = 1.0,
        **kwargs,
    ):
        if singularity_image_cache_dir:
            self._image_cache_dir = Path(singularity_image_cache_dir)
        else:
            self._image_cache_dir = Path(tempfile.mkdtemp(prefix="singularity_cache_"))

        self._force_pull = singularity_force_pull
        self._singularity_no_mount = singularity_no_mount
        # Set before ``super().__init__`` because the base ``__init__`` calls
        # ``_validate_definition`` (which reads ``_prebuilt``) before returning.
        self._prebuilt = _prebuilt
        self._nv = _nv
        self._fakeroot = _fakeroot
        self._seal_command_prefix = tuple(_seal_command_prefix)
        self._scratch_workdir = _scratch_workdir
        # See the LaunchSpec fields of the same names.
        self._overlay_dir = _overlay_dir
        self._lower_overlay_dirs = tuple(_lower_overlay_dirs)
        self._server_python = _server_python
        self._start_in_staging_dir = _start_in_staging_dir
        self._ready_poll_sec = _ready_poll_sec

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **kwargs,
        )

        self._server_process: asyncio.subprocess.Process | None = None
        self._server_port: int | None = None
        #: Base URL for the control channel: ``http://localhost:<port>`` (TCP) or
        #: ``http://localhost`` (when the client uses a UNIX-socket transport, sealed mode).
        self._control_base: str | None = None
        self._staging_dir: Path | None = None
        self._sif_path: Path | None = None
        self._stream_task: asyncio.Task | None = None
        self._memory_watchdog_task: asyncio.Task | None = None
        self._http_client: httpx.AsyncClient | None = None
        #: Process group of the launch process, covering everything that runs outside the
        #: container: the network seal, the apptainer runtime parent and its fuse helpers.
        self._sandbox_pgid: int | None = None
        #: The container's PID-namespace init, as recorded by ``_record_container_init``.
        self._container_init_pid: int | None = None
        #: An open fd on the namespace that init leads, held from
        #: ``_record_container_init`` until ``_terminate_sandbox`` ends. While it is open
        #: the namespace cannot be freed, so its identity cannot pass to another
        #: namespace that teardown would then kill. Set exactly when the init pid is.
        self._container_pid_ns_fd: int | None = None

        memory_mb = self._effective_memory_mb
        self._memory_limit_bytes = (
            memory_mb * 1024 * 1024 if memory_mb is not None else None
        )
        self._memory_limit_exceeded: str | None = None

        self._workdir = self._resolve_workdir()

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.SINGULARITY

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # gpus: ``--nv`` exposes the job's GPUs into the container (the keepalive build
        # sandbox smoke-tests the server on a GPU). egress_allowlist: the HLS substrate
        # seals the launch via a ``seal_command_prefix`` (rootless ``pasta`` +
        # iptables wrapper around the ``singularity exec`` spawn), which
        # this backend prepends to its launch argv -- so it CAN enforce an egress
        # allowlist, just not via a Docker-style sidecar or a Modal-native allowlist.
        # disable_internet: see ``_isolate_network``. overlay_dirs: apptainer
        # ``--overlay``.
        return EnvironmentCapabilities(
            mounted=True,
            gpus=True,
            egress_allowlist=True,
            disable_internet=True,
            overlay_dirs=True,
        )

    @property
    def _isolate_network(self) -> bool:
        """Whether the container gets a network namespace with no route out.

        From ``allow_internet=False``, on the trial path and the ``launch`` path alike.
        Apptainer's user guide names ``none`` as the one network an unprivileged user
        may request: it leaves the container only a loopback interface, so there is no
        egress to filter and nothing inside to reopen.
        """
        return not self.task_env_config.allow_internet

    @property
    def is_prebuilt(self) -> bool:
        return self._prebuilt

    @property
    def _docker_image(self) -> str | None:
        return self.task_env_config.docker_image

    @property
    def _staging(self) -> Path:
        """Return staging dir, raising if not yet initialized."""
        if self._staging_dir is None:
            raise RuntimeError("Staging directory not initialized — call start() first")
        return self._staging_dir

    @property
    def _is_sif_image(self) -> bool:
        """True when docker_image points to a pre-built .sif file."""
        return bool(self._docker_image and self._docker_image.endswith(".sif"))

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self):
        """Validate that required files and configuration exist."""
        if not self._docker_image:
            raise ValueError(
                "Singularity environment requires 'docker_image' in task.toml [environment]. "
                "Set it to a Docker image name (e.g. 'ubuntu:22.04') or a .sif file path."
            )

        # The lightweight ``launch`` path is prebuilt: the image itself is the complete
        # environment, so a ``.sif`` ref need not exist on disk yet at construction
        # (it may be a ``docker://`` ref converted at start()). The existence check
        # below is a convenience for the trial path; skip it when prebuilt.
        if self._prebuilt:
            return

        if self._is_sif_image:
            sif_path = Path(self._docker_image)
            if not sif_path.exists():
                raise FileNotFoundError(
                    f".sif file not found: {sif_path}. "
                    "Please convert Docker images to .sif format first."
                )
            self.logger.debug(f"Using pre-built .sif image: {sif_path}")

    def _resolve_workdir(self) -> str:
        """Resolve the container workdir.

        Order: an explicit ``task_env_config.workdir`` (set by ``launch`` from
        ``LaunchSpec.workdir``, e.g. ``/workspace`` for a prebuilt image) wins; else the
        Dockerfile's ``WORKDIR``; else ``/app``. Honoring the configured workdir matters
        for the ``launch`` path -- there is no Dockerfile, and defaulting to ``/app`` makes
        ``--pwd /app`` fail with "chdir /app: no such file or directory" on images whose
        workdir is elsewhere."""
        configured = getattr(self.task_env_config, "workdir", None)
        if configured:
            return configured
        if self._dockerfile_path.exists():
            workdir = "/app"
            try:
                for line in self._dockerfile_path.read_text().splitlines():
                    line = line.strip()
                    if line.upper().startswith("WORKDIR "):
                        workdir = line.split(None, 1)[1].strip()
            except Exception:
                pass
            return workdir
        return "/app"

    def _reserve_port(self) -> tuple[socket.socket, int]:
        """Reserve a free port by keeping the socket bound.

        Returns a tuple of (socket, port). The caller must close the socket
        when ready to use the port, minimizing the race condition window.

        Uses SO_REUSEADDR so the port can be immediately reused after closing.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        return s, port

    async def _convert_docker_to_sif(
        self, docker_image: str, *, force_pull: bool = False
    ) -> Path:
        """Convert a Docker image to Singularity .sif format.

        Uses file locking to prevent race conditions when multiple concurrent
        tasks try to convert the same image simultaneously.
        """
        if ":" not in docker_image:
            docker_image = f"{docker_image}:latest"

        safe_name = docker_image.replace("/", "_").replace(":", "_")
        sif_path = self._image_cache_dir / f"{safe_name}.sif"
        lock_path = self._image_cache_dir / f"{safe_name}.sif.lock"

        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

        if not force_pull and sif_path.exists():
            self.logger.debug(f"Using cached Singularity image: {sif_path}")
            return sif_path

        self.logger.debug(f"Acquiring lock for image conversion: {docker_image}")
        lock_file = open(lock_path, "w")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            )
            self.logger.debug(f"Lock acquired for: {docker_image}")

            if force_pull and sif_path.exists():
                self.logger.debug(
                    f"Force pull enabled, removing cached image: {sif_path}"
                )
                sif_path.unlink()

            if sif_path.exists():
                self.logger.debug(
                    f"Using cached Singularity image (created by another process): "
                    f"{sif_path}"
                )
                return sif_path

            self.logger.info(f"Converting Docker image to Singularity: {docker_image}")

            tmp_sif_path = (
                self._image_cache_dir / f"{safe_name}.sif.tmp.{self.session_id}"
            )

            cmd = [
                "singularity",
                "pull",
                str(tmp_sif_path),
                f"docker://{docker_image}",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                if tmp_sif_path.exists():
                    tmp_sif_path.unlink()
                error_msg = stderr.decode(errors="replace")
                raise RuntimeError(f"Failed to convert Docker image: {error_msg}")

            tmp_sif_path.rename(sif_path)

            self.logger.info(f"Created Singularity image: {sif_path}")
            return sif_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    async def _cleanup_server_resources(self) -> None:
        """Clean up any existing server resources from a previous run."""
        if hasattr(self, "_server_process") and self._server_process:
            await self._terminate_sandbox()
        if (
            hasattr(self, "_stream_task")
            and self._stream_task
            and not self._stream_task.done()
        ):
            # Let the reader finish first. The sandbox is already torn down above, so the
            # server's stdout is at EOF and the task ends on its own within a moment --
            # carrying the last lines it wrote, which in a crash are the only account of
            # why. Cancelling straight away drops whatever is still unread in the pipe.
            try:
                await asyncio.wait_for(asyncio.shield(self._stream_task), timeout=2)
            except (TimeoutError, asyncio.TimeoutError):
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        if hasattr(self, "_http_client") and self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        if (
            hasattr(self, "_memory_watchdog_task")
            and self._memory_watchdog_task
            and not self._memory_watchdog_task.done()
        ):
            self._memory_watchdog_task.cancel()
            try:
                await self._memory_watchdog_task
            except asyncio.CancelledError:
                pass
        if self._staging_dir and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None

    def _build_singularity_command(
        self,
        *,
        sif_path: str,
        staging_dir: str,
        port: int | None = None,
        uds: str | None = None,
        workdir: str,
        bind_mounts: list[str] | None = None,
    ) -> list[str]:
        """Construct the ``singularity exec`` argv that runs the keepalive exec server.

        Extracted from ``_start_server`` so the launch command (in particular the
        ``--nv`` GPU flag) is unit-testable without starting a container. ``--nv`` is
        inserted right after ``exec`` (a singularity exec flag, before the image) when
        ``self._nv`` is set, so the NVIDIA driver + the job's visible GPUs are exposed
        inside the container. ``bind_mounts`` is the pre-built ``-B src:dst`` list; when
        omitted only the staging bind is used (the unit-test path). The server listens on a
        UNIX socket (``uds``) when sealed -- reachable across the seal's private netns -- or
        a TCP ``port`` otherwise; exactly one must be given.
        """
        if bind_mounts is None:
            bind_mounts = ["-B", f"{staging_dir}:/staging"]
        listen_args = ["--uds", uds] if uds else ["--port", str(port)]

        no_mount_args: list[str] = []
        singularity_no_mount = self._singularity_no_mount
        if singularity_no_mount is None:
            singularity_no_mount = "home,tmp,bind-paths"
        sealed = bool(self._seal_command_prefix)
        if singularity_no_mount:
            for part in singularity_no_mount.split(","):
                part = part.strip()
                if not part:
                    continue
                # Under the seal the writable overlay is unavailable (rootless nested userns
                # on older kernels), so /tmp and $HOME must come from --containall's tmpfs.
                # ``--no-mount tmp``/``home`` would disable that, leaving them read-only and
                # breaking the agent's config dir + the collection's transfer tar -- so keep
                # them mounted when sealed.
                if sealed and part in ("tmp", "home"):
                    continue
                no_mount_args.extend(["--no-mount", part])

        bootstrap_cmd = [
            "bash",
            "-c",
            'exec /staging/bootstrap.sh "$@"',
            "bash",
            workdir,
            "/staging/_hbexec.py",
            *listen_args,
            "--workdir",
            workdir,
        ]
        # ``--nv`` (GPU passthrough) goes before the image arg.
        nv_args = ["--nv"] if self._nv else []
        # ``--fakeroot`` is opt-in: on a rootless install without /etc/subuid it forces
        # the fakeroot-command + root-mapped namespace, under which the --writable-tmpfs
        # overlay upper is NOT writable (apptainer warns "Consider using a different
        # overlay upper layer filesystem type"), so the server's directory setup and the
        # agent's writes fail with EACCES. Without it the container runs as the invoking
        # user with a fully writable overlay -- correct for the keepalive build sandbox.
        fakeroot_args = ["--fakeroot"] if self._fakeroot else []
        # Seal backstop: when the producer seal is active, the payload runs as root in the
        # seal's user namespace and would otherwise hold CAP_NET_ADMIN over the sealed
        # netns -- enough to ``iptables -P OUTPUT ACCEPT`` and walk straight past the
        # kernel seal. Drop the network-admin capabilities for the payload (apptainer takes
        # the full set at startup, then drops these for the contained process), so the seal
        # cannot be undone from inside. A build needs neither cap. Only applied under a seal.
        cap_args = (
            ["--drop-caps", "CAP_NET_ADMIN,CAP_NET_RAW"]
            if self._seal_command_prefix
            else []
        )
        # Under the producer seal the container root is read-only (no writable overlay in a
        # rootless nested userns on older kernels), so the dirs the agent writes to must be
        # writable tmpfs mounts (apptainer --scratch). ``_scratch_workdir`` (set by the
        # substrate when nothing is bound at the workdir) gates this; we scratch the workdir
        # plus /logs (the agent's CLAUDE_CONFIG_DIR + logs live there -- EnvironmentPaths).
        scratch_args: list[str] = []
        if self._scratch_workdir:
            for _t in (workdir, "/logs"):
                scratch_args.extend(["--scratch", _t])
        # ``--network`` is only accepted alongside ``--net``.
        net_args = ["--net", "--network", "none"] if self._isolate_network else []
        # Apptainer stacks overlays in the order given, so the read-only lowers come
        # first and the writable layer last.
        writable_args = [
            arg
            for lower in self._lower_overlay_dirs
            for arg in ("--overlay", f"{lower}:ro")
        ]
        writable_args += (
            ["--overlay", self._overlay_dir]
            if self._overlay_dir
            else ["--writable-tmpfs"]
        )
        # Read by bootstrap.sh; ``--containall`` passes no host environment.
        env_args = (
            ["--env", f"HARBOR_SERVER_PYTHON={self._server_python}"]
            if self._server_python
            else []
        )
        exec_cmd = [
            "singularity",
            "exec",
            *nv_args,
            *no_mount_args,
            "--pwd",
            workdir,
            *writable_args,
            *fakeroot_args,
            *cap_args,
            *net_args,
            *scratch_args,
            *env_args,
            "--containall",
            "--pid",
            *bind_mounts,
            sif_path,
            *bootstrap_cmd,
        ]
        # Producer seal (HLS Apptainer substrate): prepend the externally-built wrapper
        # (``pasta`` + iptables) so the keepalive exec server is
        # spawned INSIDE the sealed namespace -- its egress reaches only the forward
        # proxy. Empty in open mode.
        return [*self._seal_command_prefix, *exec_cmd]

    async def _start_server(self) -> None:
        """Start the FastAPI server inside the Singularity container.

        Uses port reservation with retry logic to handle race conditions where
        another process might grab the port between reservation and binding.
        """
        await self._cleanup_server_resources()

        self._staging_dir = Path(tempfile.mkdtemp(prefix="singularity_staging_"))
        self._staging_dir.chmod(0o755)

        server_script = Path(__file__).parent / "server.py"
        staging_server = self._staging_dir / "_hbexec.py"
        shutil.copy(server_script, staging_server)

        bootstrap_src = Path(__file__).parent / "bootstrap.sh"
        bootstrap_script = self._staging_dir / "bootstrap.sh"
        shutil.copy(bootstrap_src, bootstrap_script)
        bootstrap_script.chmod(0o755)

        # --memory/--cpus flags are NOT used (they need cgroups/systemd, typically absent on
        # HPC); resource limits are enforced at the SLURM level (--mem, --cpus-per-task).
        # ``sealed``: the container has a network namespace of its own, so the host
        # cannot reach a TCP listener in it. The prefix seal and isolation both do that.
        sealed = bool(self._seal_command_prefix) or self._isolate_network
        env_files_dir = self.environment_dir / "files"

        def _build_bind_mounts() -> list[str]:
            bm = ["-B", f"{self._staging_dir}:/staging"]
            # ``--no-mount bind-paths`` (the default) disables apptainer's resolv.conf bind, so
            # we bind the host resolver + hosts file explicitly. The pasta producer seal copies
            # the HOST's resolvers into its network namespace (and the seal's UDP rule permits
            # DNS to exactly those nameservers), so the container resolves via the host
            # /etc/resolv.conf whether sealed or not; there is no private stub resolver to
            # point at.
            for _netfile in ("/etc/resolv.conf", "/etc/hosts"):
                if os.path.exists(_netfile):
                    bm.extend(["-B", f"{_netfile}:{_netfile}"])
            for mount in self._mounts:
                # Default a missing ``type`` to "bind" (the only kind apptainer takes via -B);
                # honor ``read_only`` as the apptainer ``:ro`` bind option.
                if mount.get("type", "bind") != "bind":
                    continue
                spec = f"{mount['source']}:{mount['target']}"
                if mount.get("read_only"):
                    spec += ":ro"
                bm.extend(["-B", spec])
            if env_files_dir.exists():
                bm.extend(["-B", f"{env_files_dir}:/staging/env_files"])
            return bm

        # Control channel: a UNIX socket on the (bind-mounted) staging dir when sealed -- the
        # only way the host reaches the server across the seal's private netns -- else TCP
        # loopback with port-collision retry (the unsealed path shares the host netns).
        container_uds = "/staging/control.sock"
        host_uds = self._staging_dir / "control.sock"
        max_port_retries = 3
        attempts = 1 if sealed else max_port_retries
        last_error = None

        for attempt in range(attempts):
            reserved_socket = None
            if sealed:
                self._control_base = (
                    "http://localhost"  # host is ignored over a uds transport
                )
                cmd = self._build_singularity_command(
                    sif_path=str(self._sif_path),
                    staging_dir=str(self._staging_dir),
                    uds=container_uds,
                    workdir=self._workdir,
                    bind_mounts=_build_bind_mounts(),
                )
            else:
                reserved_socket, port = self._reserve_port()
                self._server_port = port
                self._control_base = f"http://localhost:{port}"
                cmd = self._build_singularity_command(
                    sif_path=str(self._sif_path),
                    staging_dir=str(self._staging_dir),
                    port=port,
                    workdir=self._workdir,
                    bind_mounts=_build_bind_mounts(),
                )

            listen_desc = "uds" if sealed else f"port {self._server_port}"
            self.logger.info(
                f"Starting Singularity container on {listen_desc} (attempt {attempt + 1}/{attempts})"
            )
            if reserved_socket is not None:
                reserved_socket.close()

            self._server_process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self._staging if self._start_in_staging_dir else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            # Leading a new session makes the launch process its own group leader, so
            # everything it spawns outside the container can be signalled as one group.
            self._sandbox_pgid = self._server_process.pid

            self._stream_task = asyncio.create_task(self._stream_server_output())

            # trust_env=False: never route the control calls through a host http(s)_proxy.
            transport = httpx.AsyncHTTPTransport(uds=str(host_uds)) if sealed else None
            self._http_client = httpx.AsyncClient(
                timeout=30.0, trust_env=False, transport=transport
            )
            server_ready = False

            # A deadline, so the budget does not shrink with the poll interval.
            deadline = time.monotonic() + SERVER_READY_TIMEOUT_SEC
            while time.monotonic() < deadline:
                try:
                    response = await self._http_client.get(
                        f"{self._control_base}/health"
                    )
                    if response.status_code == 200:
                        # Verify OUR server is still alive before declaring ready (an unsealed
                        # TCP path shares the host netns, so another trial may grab the port).
                        if self._server_process.returncode is not None:
                            await self._stream_task
                            last_error = RuntimeError(
                                "health check succeeded but the server process died (port collision?)"
                            )
                            break
                        self.logger.info("Singularity FastAPI server is ready")
                        if self._memory_limit_bytes is not None:
                            self._memory_watchdog_task = asyncio.create_task(
                                self._memory_watchdog()
                            )
                        server_ready = True
                        break
                except httpx.RequestError:
                    pass

                if self._server_process.returncode is not None:
                    await self._stream_task
                    last_error = RuntimeError(
                        "Server process died before readiness. Check trial.log for server output."
                    )
                    break

                # Never past the deadline, which would report the failure late.
                await asyncio.sleep(
                    min(self._ready_poll_sec, max(0.0, deadline - time.monotonic()))
                )

            if server_ready:
                self._record_container_init()
                return

            # Clean up the failed sandbox before retrying.
            await self._terminate_sandbox()
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

        raise last_error or RuntimeError("Failed to start Singularity FastAPI server")

    def _record_container_init(self) -> None:
        """Record the container's PID-namespace init and pin its namespace.

        ``--pid`` gives the payload its own PID namespace. Its init adopts every process
        the agent starts, including ones detached with ``setsid``, and a SIGKILL to it
        makes the kernel kill the rest of the namespace. It is the nearest descendant of
        the launch process that lives in a different PID namespace than we do.

        Teardown matches processes by namespace, and a namespace's identity is an inode
        number the kernel frees when the namespace dies and can give to the next one
        created. A sandbox that died on its own would lend its number to whatever
        sandbox started next, and teardown would kill that one. Holding an fd on the
        namespace keeps it, and its number, allocated until teardown ends.
        """
        if not self._server_process:
            return

        own_ns = _proc_pid_namespace(os.getpid())
        for pid in _proc_descendants(self._server_process.pid):
            ns = _proc_pid_namespace(pid)
            if ns is None or ns == own_ns:
                continue
            try:
                fd: int | None = os.open(
                    f"/proc/{pid}/ns/pid", os.O_RDONLY | os.O_CLOEXEC
                )
            except OSError:
                fd = None
            # The init may have exited between the two reads and its pid gone to another
            # process. Pin only the namespace we saw it lead, never some other one.
            if fd is not None and _fd_pid_namespace(fd) != ns:
                os.close(fd)
                fd = None
            if fd is None:
                self.logger.warning(
                    f"The container's PID namespace pid:[{ns[1]}] was already gone when "
                    f"its init {pid} was recorded; teardown will only cover the launch "
                    "process group."
                )
                return
            self._container_init_pid = pid
            self._container_pid_ns_fd = fd
            self.logger.debug(
                f"Container PID namespace pid:[{ns[1]}] is led by pid {pid}"
            )
            return

        self.logger.warning(
            "Could not identify the container's PID-namespace init; teardown will only "
            "cover the launch process group."
        )

    async def _describe_server_exit(self) -> str:
        """Why the exec server is gone, as far as its exit status can say.

        Waits for the status before describing it. The caller reaches here the instant an
        exec raises, and the HTTP EOF and the process death arrive together, so the launch
        process is typically not yet reaped and ``returncode`` is still ``None``. Reading
        it unawaited returns "still running" in the one case where the process is gone,
        which points the next reader away from the death being reported."""
        if not self._server_process:
            return "no server process"
        if self._server_process.returncode is None:
            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=5)
            except (TimeoutError, asyncio.TimeoutError):
                return (
                    "server process still running after 5s "
                    "(the channel dropped, not the process)"
                )
        rc = self._server_process.returncode
        if rc is None:
            return "server process still running (the channel dropped, not the process)"
        return _describe_exit_status(rc)

    async def _stream_server_output(self) -> None:
        """Stream server stdout/stderr to logger in real-time.

        Logged at WARNING, not debug. This is the container's ENTIRE stdout and stderr --
        the seal script's `FATAL:` lines, pasta's errors, the egress proxy's traceback and
        uvicorn's last words before it dies. Callers run at INFO, so at debug every one of
        those would be discarded, and a sandbox that died mid-run would leave no server-side
        record. A quiet server prints nothing, so this costs nothing on a healthy run.
        """
        if not self._server_process or not self._server_process.stdout:
            return

        try:
            async for line in self._server_process.stdout:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    self.logger.warning(f"[server] {decoded}")
        except Exception as e:
            # !r, not str: many of the exceptions that reach here (ConnectionReset,
            # IncompleteRead) stringify to nothing at all.
            self.logger.warning(f"Server output stream ended: {e!r}")

    def _get_process_tree_memory(self, pid: int) -> int:
        """Get total PSS memory of a process and all its descendants.

        Uses PSS (Proportional Set Size) which properly accounts for shared
        memory by dividing it proportionally among sharing processes. Falls
        back to RSS if PSS is unavailable. Directly reads /proc for efficiency.
        Returns memory in bytes, or 0 if unable to read.
        """

        def get_process_memory(p: int) -> int:
            # Try PSS from smaps_rollup (Linux 4.14+)
            try:
                for line in Path(f"/proc/{p}/smaps_rollup").read_text().splitlines():
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
            except (OSError, PermissionError, ValueError, IndexError):
                pass

            # Fallback to RSS from statm (second field, in pages)
            try:
                rss_pages = int(Path(f"/proc/{p}/statm").read_text().split()[1])
                return rss_pages * os.sysconf("SC_PAGE_SIZE")
            except (OSError, PermissionError, ValueError, IndexError):
                pass

            return 0

        try:
            return sum(get_process_memory(p) for p in _proc_descendants(pid))
        except Exception:
            return 0

    async def _memory_watchdog(self) -> None:
        """Monitor memory usage and kill container if it exceeds the limit.

        Features:
        - Adaptive intervals: checks every 1s when memory >50%, every 3s otherwise
        - Explosion detection: warns if growth rate would hit limit in <5s
        - Kill threshold at 95%: leaves headroom before actual OOM
        """
        if self._memory_limit_bytes is None:
            return

        base_interval = 3
        fast_interval = 1
        warning_threshold = 0.5
        kill_threshold = 0.95

        last_mem_usage = 0
        last_check_time = 0.0

        self.logger.debug(
            f"Memory watchdog started: "
            f"limit={self._memory_limit_bytes // 1024 // 1024}MB, "
            f"kill_at={kill_threshold * 100:.0f}%, "
            f"intervals={fast_interval}s/{base_interval}s"
        )

        while True:
            try:
                if (
                    not self._server_process
                    or self._server_process.returncode is not None
                ):
                    self.logger.debug("Memory watchdog: server process ended, stopping")
                    break

                current_time = asyncio.get_event_loop().time()
                mem_usage = self._get_process_tree_memory(self._server_process.pid)
                mem_mb = mem_usage / 1024 / 1024
                limit_mb = self._memory_limit_bytes / 1024 / 1024
                usage_pct = (
                    mem_usage / self._memory_limit_bytes
                    if self._memory_limit_bytes > 0
                    else 0
                )

                if last_check_time > 0 and last_mem_usage > 0:
                    time_delta = current_time - last_check_time
                    if time_delta > 0:
                        growth_rate = (mem_usage - last_mem_usage) / time_delta
                        if growth_rate > 0:
                            remaining_bytes = (
                                self._memory_limit_bytes * kill_threshold - mem_usage
                            )
                            time_to_limit = (
                                remaining_bytes / growth_rate
                                if growth_rate > 0
                                else float("inf")
                            )
                            if 0 < time_to_limit < 5:
                                self.logger.warning(
                                    f"Memory explosion detected: {mem_mb:.0f}MB, "
                                    f"growing {growth_rate / 1024 / 1024:.0f}MB/s, "
                                    f"~{time_to_limit:.1f}s until limit"
                                )

                last_mem_usage = mem_usage
                last_check_time = current_time

                if mem_usage > self._memory_limit_bytes * kill_threshold:
                    error_msg = (
                        f"Container exceeded memory limit "
                        f"({mem_mb:.0f}MB > "
                        f"{limit_mb * kill_threshold:.0f}MB)"
                    )
                    self.logger.error(
                        f"Memory limit exceeded: {mem_mb:.0f}MB > "
                        f"{limit_mb * kill_threshold:.0f}MB "
                        f"({usage_pct * 100:.0f}%). Killing container."
                    )
                    self._memory_limit_exceeded = error_msg
                    self._server_process.kill()
                    return

                interval = (
                    fast_interval if usage_pct > warning_threshold else base_interval
                )
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                self.logger.debug("Memory watchdog cancelled")
                raise
            except Exception as e:
                self.logger.debug(f"Memory watchdog error (continuing): {e}")
                await asyncio.sleep(base_interval)

    async def start(self, force_build: bool) -> None:
        """Start the Singularity environment."""
        if sys.platform == "win32":
            raise RuntimeError(
                "SingularityEnvironment is only supported on Linux/macOS. "
                "Singularity/Apptainer is not available on Windows."
            )
        if not self._docker_image:
            raise ValueError("docker_image must be set in task.toml [environment]")
        if self._is_sif_image:
            self._sif_path = Path(self._docker_image)
        else:
            force_pull = force_build or self._force_pull
            self._sif_path = await self._convert_docker_to_sif(
                self._docker_image, force_pull=force_pull
            )

        await self._start_server()

    async def stop(self, delete: bool) -> None:
        """Stop the Singularity environment and all child processes."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._memory_watchdog_task and not self._memory_watchdog_task.done():
            self._memory_watchdog_task.cancel()
            try:
                await self._memory_watchdog_task
            except asyncio.CancelledError:
                pass

        if self._server_process and self._server_process.returncode is not None:
            # The server was already gone when we came to stop it: the sandbox died on its
            # own, and the orchestrator saw only "Server disconnected without sending a
            # response". The returncode is the one bit that names the broken link, and the
            # block below only runs while the process is still alive. The status is read the
            # same way the exec path reads it, so one death is described the same in both
            # places.
            rc = self._server_process.returncode
            self.logger.warning(
                "Singularity server process %d had ALREADY EXITED before stop(): "
                "returncode=%d (%s). This is the sandbox dying on its own rather than "
                "being torn down.",
                self._server_process.pid,
                rc,
                _describe_exit_status(rc),
            )

        if self._server_process:
            await self._terminate_sandbox()

        if hasattr(self, "_stream_task") and self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._staging_dir and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)

        if delete:
            self.logger.debug(
                f"Singularity image preserved at {self._sif_path} for reuse"
            )

    async def _terminate_sandbox(self) -> None:
        """Kill the container and everything the launch spawned around it.

        The sandbox straddles two PID namespaces, so teardown needs both handles: a
        SIGKILL to the container's init, which the kernel turns into a SIGKILL of every
        process inside it, and the launch process group, which covers what runs outside
        -- the network seal, the apptainer runtime parent and its fuse helpers. Killing
        the container first lets the runtime parent unmount on its way out. A final sweep
        kills and reports whatever either step missed.

        The namespace fd taken by ``_record_container_init`` is closed when this returns
        or raises, and not before: the sweep relies on it to name the namespace.
        """
        try:
            if not self._server_process or self._sandbox_pgid is None:
                return

            if (
                self._container_init_pid is not None
                and self._container_pid_ns_fd is not None
            ):
                if _proc_pid_namespace(self._container_init_pid) == _fd_pid_namespace(
                    self._container_pid_ns_fd
                ):
                    _send_signal(self._container_init_pid, signal.SIGKILL)
                    self.logger.debug(
                        f"Sent SIGKILL to container init {self._container_init_pid}"
                    )
                else:
                    # The recorded pid is gone or has been reused, so signalling it would
                    # hit a stranger. Anything left in the namespace is the sweep's job.
                    self.logger.debug(
                        f"Container init {self._container_init_pid} no longer leads the "
                        "container's PID namespace"
                    )

            pgid = self._sandbox_pgid
            _send_signal(pgid, signal.SIGTERM, group=True)
            self.logger.debug(f"Sent SIGTERM to Singularity process group {pgid}")
            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.debug("Graceful shutdown timed out, force killing")
            # A process group id is not reused while the group still has members, so this
            # stays correctly targeted even though the launch process has been reaped.
            _send_signal(pgid, signal.SIGKILL, group=True)
            if self._server_process.returncode is None:
                await self._server_process.wait()

            survivors = await self._sweep_sandbox()
            if survivors:
                detail = "; ".join(f"{pid} ({_proc_cmdline(pid)})" for pid in survivors)
                self.logger.error(
                    f"Singularity teardown left {len(survivors)} process(es) running: "
                    f"{detail}"
                )

            self._sandbox_pgid = None
        finally:
            if self._container_pid_ns_fd is not None:
                os.close(self._container_pid_ns_fd)
            self._container_pid_ns_fd = None
            self._container_init_pid = None

    async def _sweep_sandbox(self) -> list[int]:
        """SIGKILL whatever is left of the sandbox; return what is still alive after."""
        for _ in range(5):
            leftover = self._sandbox_processes()
            if not leftover:
                return []
            for pid in leftover:
                _send_signal(pid, signal.SIGKILL)
            await asyncio.sleep(0.2)
        return self._sandbox_processes()

    def _sandbox_processes(self) -> list[int]:
        """Live pids still in the sandbox's PID namespace or its launch process group.

        A pid counts as in the namespace only when its ``ns/pid`` link names the same
        namespace as the fd we hold, so the namespace half is skipped when no fd was
        taken. A number recorded earlier would not do: once its namespace dies, the
        kernel may give it to an unrelated sandbox. Zombies are ignored: they hold
        nothing and their parent reaps them.
        """
        pinned = (
            _fd_pid_namespace(self._container_pid_ns_fd)
            if self._container_pid_ns_fd is not None
            else None
        )
        own_pid = os.getpid()
        found: list[int] = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == own_pid:
                continue
            stat = _proc_stat(pid)
            if stat is None or stat[0] == "Z":
                continue
            if stat[1] == self._sandbox_pgid or (
                pinned is not None and _proc_pid_namespace(pid) == pinned
            ):
                found.append(pid)
        return found

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute a command in the Singularity container via HTTP."""
        if not self._http_client or not self._control_base:
            raise RuntimeError("Singularity environment not started")

        if self._memory_limit_exceeded:
            raise MemoryLimitExceededError(self._memory_limit_exceeded)

        # ``su`` to a target user only works when the container runs as root, i.e. under
        # ``--fakeroot``. Without it (the default for the rootless build sandbox) the
        # process IS the invoking user, ``su`` would fail ("su: Authentication failure"),
        # and switching is unnecessary -- run the command directly.
        resolved_user = self._resolve_user(user) if self._fakeroot else None
        if resolved_user is not None:
            if isinstance(resolved_user, int):
                user_arg = f"$(getent passwd {resolved_user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(resolved_user))
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"

        merged_env = self._merge_env(env)

        try:
            # timeout_sec is the command's wall-clock budget (the in-container exec server
            # kills the command after it). The client HTTP read timeout must OUTLAST that,
            # so cap it at timeout_sec + 10s of slack. When timeout_sec is None the command
            # is explicitly UNbounded -- e.g. a long agent run, itself bounded by the
            # orchestrator's own wait_for(build_timeout_sec) -- so the HTTP call must be
            # unbounded too (None). A finite default here would sever a long-but-healthy
            # command mid-flight: a 40-turn build legitimately runs many minutes, and a
            # spurious client timeout would abort it even though the server is streaming.
            http_timeout = (timeout_sec + 10) if timeout_sec is not None else None

            response = await self._http_client.post(
                f"{self._control_base}/exec",
                json={
                    "command": command,
                    "cwd": cwd,
                    "env": merged_env,
                    "timeout_sec": timeout_sec,
                },
                timeout=http_timeout,
            )
            response.raise_for_status()
            result = response.json()

            exec_result = ExecResult(
                stdout=result.get("stdout"),
                stderr=result.get("stderr"),
                return_code=result.get("return_code", 1),
            )

            if exec_result.return_code != 0:
                error_output = exec_result.stderr or exec_result.stdout or "<no output>"
                self.logger.debug(
                    f"Command exited with rc={exec_result.return_code}: {error_output}"
                )

            return exec_result

        except httpx.TimeoutException:
            if self._memory_limit_exceeded:
                raise MemoryLimitExceededError(self._memory_limit_exceeded)
            raise asyncio.TimeoutError(
                f"HTTP request timed out after {http_timeout} seconds"
                if http_timeout
                else "HTTP request timed out"
            )
        # ReadError belongs with the other two: the connection died partway through
        # reading the reply, which is the same lost control channel arriving a moment
        # later. Left out, it escapes with an empty message ("ReadError: "), none of the
        # diagnosis below runs, and a dead exec server looks like a command that failed.
        # Once this fires it is usually terminal: every later command would fail too.
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError):
            if self._memory_limit_exceeded:
                raise MemoryLimitExceededError(self._memory_limit_exceeded)
            # The control channel dropped and the memory watchdog did not fire, so the
            # server died for some other reason. Its exit status is the only thing that
            # says which; ``_describe_exit_status`` explains how to read it. Reported at
            # error level because losing a sandbox mid-run ends the rollout.
            self.logger.error(
                f"Exec channel lost: {await self._describe_server_exit()}"
            )
            raise

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a file to the container via staging directory."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        staging_file = self._staging / source.name
        shutil.copy2(source, staging_file)

        result = await self.exec(
            f"cp {shlex.quote('/staging/' + source.name)} {shlex.quote(target_path)}"
        )
        staging_file.unlink(missing_ok=True)
        if result.return_code != 0:
            error_output = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to upload file: {error_output}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a directory to the container via staging directory."""
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        staging_subdir = self._staging / source.name
        if staging_subdir.exists():
            shutil.rmtree(staging_subdir)
        shutil.copytree(source, staging_subdir)

        await self.exec(f"mkdir -p {shlex.quote(target_dir)}")
        result = await self.exec(
            f"cp -r /staging/{shlex.quote(source.name)}/. {shlex.quote(target_dir)}/"
        )
        if result.return_code != 0:
            result = await self.exec(
                f"cp -r {shlex.quote('/staging/' + source.name)} {shlex.quote(target_dir)}"
            )
            if result.return_code != 0:
                shutil.rmtree(staging_subdir, ignore_errors=True)
                error_output = result.stderr or result.stdout or "<no output>"
                raise RuntimeError(f"Failed to upload directory: {error_output}")
        shutil.rmtree(staging_subdir, ignore_errors=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the container via staging directory."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        filename = Path(source_path).name
        staging_path = f"/staging/download_{filename}"
        result = await self.exec(
            f"cp {shlex.quote(source_path)} {shlex.quote(staging_path)}"
        )
        if result.return_code != 0:
            error_output = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to download file: {error_output}")

        staging_file = self._staging / f"download_{filename}"
        if staging_file.exists():
            shutil.copy2(staging_file, target)
            staging_file.unlink()
        else:
            raise RuntimeError(f"File not found in staging: {staging_file}")

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the container via staging directory."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        dirname = Path(source_dir).name
        staging_path = f"/staging/download_{dirname}"
        result = await self.exec(
            f"cp -r {shlex.quote(source_dir)} {shlex.quote(staging_path)}"
        )
        if result.return_code != 0:
            error_output = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to download directory: {error_output}")

        staging_subdir = self._staging / f"download_{dirname}"
        if staging_subdir.exists():
            for item in staging_subdir.iterdir():
                dest = target / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            shutil.rmtree(staging_subdir)
        else:
            raise RuntimeError(f"Directory not found in staging: {staging_subdir}")
