"""LaunchSpec - backend-agnostic description of how to run a prebuilt image
as an environment. Consumed by ``BaseEnvironment.launch`` (see base.py)."""

from __future__ import annotations
from dataclasses import dataclass, field

from harbor.models.trial.config import ServiceVolumeConfig


@dataclass(frozen=True)
class EgressGatewaySpec:
    """A sidecar that owns a sealed environment's entire network namespace and
    transparently filters its egress.

    The sealed ``main`` service joins the gateway's netns (Docker
    ``network_mode: service:<gateway>``) with no interfaces and no network-privileged
    capabilities of its own (the writer drops ``NET_ADMIN``/``NET_RAW``), so every outbound
    connection traverses the gateway's redirect rules and the client cannot forge the
    firewall mark that exempts them. The gateway holds the capabilities in ``cap_add``
    (``NET_ADMIN`` to program the netns), installs the rules, and runs the filtering
    interceptor. Consumed by Docker; backends without the ``egress_allowlist`` capability
    reject it.

    ``image``/``command``/``env`` define the gateway container (``command`` empty -> the
    image's own ENTRYPOINT). ``healthcheck`` is an optional Compose healthcheck block the
    sealed service depends on, so it never starts unsealed -- it waits until the gateway
    reports healthy (rules installed + interceptor listening).
    """

    image: str
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cap_add: tuple[str, ...] = ("NET_ADMIN",)
    healthcheck: "dict | None" = None


@dataclass(frozen=True)
class LaunchSpec:
    """Run a prebuilt image as an environment. Portable core + a serve piece.

    serve_command None  -> keepalive sandbox (agent execs into it)
    serve_command set   -> long-lived service (requires the ``serves`` capability)
    gpu_device_ids set  -> requires the ``gpu_pinning`` capability
    allow_internet False-> requires the ``disable_internet`` capability
    service_port set    -> the port the served process listens on; required by
                           backends that expose ingress (Modal ``encrypted_ports``)
                           and used by Docker ``endpoint()`` resolution.
    region set          -> the geographic region to place the environment in;
                           used to co-locate a load driver with the artifact it
                           measures (Modal ``Sandbox.create(region=...)``).
                           Backends without region placement ignore it.

    ``image`` is usually a registry tag (str), but a backend that produces an
    image handle from a checkpoint (Modal's ``commit_checkpoint`` returns a
    ``modal.Image``) may pass that handle back in as ``image`` for a relaunch;
    such a handle is opaque to the portable core and consumed only by the
    backend that produced it.
    """

    image: str | object
    serve_command: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[ServiceVolumeConfig] = field(default_factory=list)
    cpus: int | None = None
    memory_mb: int | None = None
    gpus: int = 0
    gpu_kind: str | None = None
    """Preferred GPU model (e.g. "H100", "A100", "L4"). When set, the backend
    will request this specific GPU type.  ``None`` lets the backend choose.
    Only honoured by backends that support typed GPU requests (e.g. Modal).
    Docker/local backends ignore it (they use gpu_device_ids instead)."""
    gpu_device_ids: list[str] | None = None
    service_port: int | None = None
    region: str | None = None
    allow_internet: bool = True
    workdir: str = "/workspace"
    shared_network: str | None = None
    """When set, the container joins this named Docker network in ADDITION to
    its default network. Used to allow containers in different compose projects
    (e.g. the artifact and the driver) to communicate on a shared intra-eval
    network. The network must exist before start() is called. Ignored by Modal
    (Modal uses region pinning for co-location, not Docker networks)."""
    outbound_cidr_allowlist: list[str] | None = None
    """When set, the launched sandbox's outbound egress is restricted to these
    CIDRs (Modal ``Sandbox.create(outbound_cidr_allowlist=...)``). Requires the
    ``egress_allowlist`` capability. Ignored by Docker (which seals transparently via
    the egress gateway instead)."""
    outbound_domain_allowlist: list[str] | None = None
    """When set, the launched sandbox's outbound egress is restricted to these domain
    names (Modal ``Sandbox.create(outbound_domain_allowlist=...)``; supports ``*.``
    wildcard prefixes). This is the name-based producer seal: the sandbox may reach
    only the model gateway + the pull-through pip index + any explicitly-allowed
    domains. Requires the ``egress_allowlist`` capability. Ignored by Docker."""
    egress_gateway: "EgressGatewaySpec | None" = None
    """When set, the launched environment is sealed behind this egress gateway
    sidecar: it shares the gateway's network namespace (no NICs/caps of its own) and
    the gateway transparently intercepts and filters all egress. Consumed by Docker.
    Requires the ``egress_allowlist`` capability."""
    seal_command_prefix: tuple[str, ...] = ()
    """An opaque argv PREFIX the backend prepends to the command it spawns to launch
    the environment, so the launched process is wrapped inside a network seal the
    BACKEND-EXTERNAL caller built. Used by the Singularity/Apptainer backend, where the
    HLS substrate's rootless producer seal (``pasta`` + iptables) is a self-contained
    wrapper script that must run
    the ``singularity exec`` keepalive subprocess inside its sealed namespace. The prefix
    is prepended VERBATIM to the launch argv (the backend appends its own command after
    it). Empty -> no seal (open mode). Requires the ``egress_allowlist`` capability.
    Backends that don't consume it ignore it (it is a launch-mechanism detail, not a
    portable policy like ``outbound_domain_allowlist``)."""
    scratch_workdir: bool = False
    """When True, mount a writable tmpfs at ``workdir`` (apptainer ``--scratch``). Used by
    the Apptainer backend under the producer seal: the container root is read-only there
    (no writable overlay in a rootless nested userns on older kernels), so the agent needs
    a writable workdir. Set by the substrate only when nothing is already bound at workdir.
    Ignored by backends that don't consume it."""
    overlay_dir: str | None = None
    """Host directory to use as the container's writable layer (apptainer ``--overlay``)
    in place of the default tmpfs. The tmpfs lives in apptainer's session directory,
    which apptainer.conf caps at ``sessiondir max size`` (64 MiB by default), and
    everything the container writes shares it. Unlike a bind over the workdir, an
    overlay leaves the image's own files visible underneath. One directory per
    container, since two sharing an upper layer would write over each other. The caller
    creates and removes it. None -> the tmpfs. Requires the ``overlay_dirs``
    capability."""
    lower_overlay_dirs: tuple[str, ...] = ()
    """Host directories stacked read-only under the writable layer (apptainer
    ``--overlay <dir>:ro``), lowest first, for a container that starts from state an
    earlier container wrote. An upper layer mounted back as a lower restores what it
    holds, deletions included, because the overlay driver's markers are in that
    directory; a copy made elsewhere can lose them, as they are encoded per filesystem.
    Several containers may share one lower at once. The caller owns these directories.
    Empty -> the container starts from its image alone. Requires the ``overlay_dirs``
    capability."""
    server_python: str | None = None
    """Absolute path, inside the container, of an interpreter that can run the exec
    server (it imports uvicorn and fastapi), tried after the image's own. For an image
    that cannot host the server in a container that cannot install it: the bootstrap's
    fallback needs a network or root. The caller makes it available, for example by a
    read-only bind in ``mounts``. None -> the image's interpreters, then the fallback.
    Consumed by Singularity; other backends run no exec server and ignore it."""
    start_in_staging_dir: bool = False
    """Run the launch process from its staging directory instead of the caller's
    working directory. Apptainer's starter opens its working directory before it starts
    the container, and under a user namespace that maps only root (a root launch
    without CAP_SYS_ADMIN) a directory owned by another uid cannot be opened. The
    staging directory belongs to the launching process. Host paths in the spec must
    then be absolute. Consumed by Singularity; other backends ignore it."""
    ready_poll_sec: float = 1.0
    """How often the launch asks the exec server whether it is up. The start is given
    the same budget whatever the interval, so a short one only shortens how long a
    start that has succeeded goes unnoticed. Consumed by Singularity; other backends
    ignore it."""
