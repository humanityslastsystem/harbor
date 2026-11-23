"""Capability flags describing what an environment type can do.

Feature capabilities (``EnvironmentCapabilities``) are exposed via
``BaseEnvironment.capabilities``. Resource policy capabilities
(``EnvironmentResourceCapabilities``) are declared on each environment class
via ``resource_capabilities()`` and used for job preflight and trial validation.
"""

from pydantic import BaseModel


class EnvironmentCapabilities(BaseModel):
    gpus: bool = False
    """Whether the environment can allocate GPUs to containers."""

    disable_internet: bool = False
    """Whether the environment can run containers without internet access."""

    windows: bool = False
    """Whether the environment can run Windows containers."""

    mounted: bool = False
    """Whether the environment mounts log directories as host filesystems."""

    docker_compose: bool = False
    """Whether the environment can run Docker Compose task environments."""

    serves: bool = False
    """Whether the environment can expose a service port and provide endpoint()."""

    can_commit: bool = False
    """Whether a running environment can be snapshotted to a reusable image."""

    gpu_pinning: bool = False
    """Whether specific GPU device IDs can be pinned (vs. requesting a count)."""

    egress_allowlist: bool = False
    """Whether the environment can seal a sandbox to an egress allowlist (a
    dual-homed proxy sidecar on Docker, an outbound CIDR allowlist on Modal),
    permitting only specified destinations while blocking all other egress."""

    overlay_dirs: bool = False
    """Whether a launched container's writable layer can be a host directory, with
    read-only directories stacked under it (``LaunchSpec.overlay_dir`` and
    ``lower_overlay_dirs``)."""


class EnvironmentResourceCapabilities(BaseModel):
    cpu_limit: bool = False
    """Whether CPU resources can be applied as a hard ceiling."""

    cpu_request: bool = False
    """Whether CPU resources can be applied as a resource request/reservation."""

    memory_limit: bool = False
    """Whether memory resources can be applied as a hard ceiling."""

    memory_request: bool = False
    """Whether memory resources can be applied as a resource request/reservation."""
