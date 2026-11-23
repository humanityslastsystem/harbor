import json
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.models.trial.config import ServiceVolumeConfig

if TYPE_CHECKING:
    from harbor.environments.launch import EgressGatewaySpec

# Shared compose file paths used by both local Docker and Daytona DinD environments.
COMPOSE_DIR = Path(__file__).parent
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"
COMPOSE_WINDOWS_KEEPALIVE_PATH = COMPOSE_DIR / "docker-compose-windows-keepalive.yaml"
RESOURCES_COMPOSE_NAME = "docker-compose-resources.json"


def write_mounts_compose_file(path: Path, mounts: list[ServiceVolumeConfig]) -> Path:
    """Write a compose override that declares services.main.volumes."""
    compose = {"services": {"main": {"volumes": list(mounts)}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_command_compose_file(path: Path, command: list[str]) -> Path:
    """Write a compose override that declares services.main.command.

    Used by service-mode (EVAL) environments to REPLACE the base
    prebuilt/build keepalive command (``sh -c sleep infinity``) with the
    image's own server entrypoint args (e.g. ``--host``/``--port``), so the
    container's ENTRYPOINT runs as the long-lived process. Must be ordered
    after the base prebuilt/build compose so Compose's last-wins scalar/
    sequence merge replaces the keepalive command.
    """
    compose = {"services": {"main": {"command": list(command)}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_env_compose_file(path: Path, env: dict[str, str]) -> Path:
    """Write a compose override that declares services.main.environment.

    Used to inject environment variables into the long-lived service process
    of a prebuilt-image env (e.g. ``HF_HUB_OFFLINE=1`` so a sealed EVAL server
    runs offline). The base prebuilt/build compose has no ``environment:``
    block and per-exec env doesn't reach the service process, so this override
    is the only way to set env on the running service container.
    """
    compose = {"services": {"main": {"environment": dict(env)}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_network_compose_file(path: Path, *, internal: bool) -> Path:
    """Write a compose override attaching ``main`` to an internal network.

    When ``internal`` is True the override declares an ``hls_internal``
    network with ``internal: true`` and attaches the ``main`` service to it.
    An internal network blocks the container's outbound internet egress while
    still letting the host reach the container at its IP — used by sealed
    EVAL environments that must serve HTTP to the host without any internet
    access. Unlike ``network_mode: none`` (which removes the network namespace
    entirely and would make the container unreachable), this keeps the
    container addressable.

    When ``internal`` is False the override is a no-op (declares nothing),
    leaving the normal default network in place.
    """
    if internal:
        compose: dict[str, object] = {
            "services": {"main": {"networks": ["hls_internal"]}},
            "networks": {"hls_internal": {"internal": True}},
        }
    else:
        compose = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_egress_gateway_compose_file(
    path: Path,
    *,
    gateway: "EgressGatewaySpec",
    service_name: str = "egress-gateway",
) -> Path:
    """Write a compose override sealing ``main`` behind a transparent egress gateway.

    ``main`` joins the gateway's network namespace (``network_mode: service:<name>``)
    with no NICs and no network-privileged capabilities of its own (NET_ADMIN/NET_RAW
    dropped), so every outbound connection traverses the gateway's redirect rules -- there
    is no second interface to route around them, and no capability to forge the mark that
    exempts them. The gateway holds ``NET_ADMIN``, installs the rules, runs the interceptor.

    ``main`` ``depends_on`` the gateway: with a healthcheck it waits for
    ``service_healthy`` (rules installed + interceptor listening) so it never runs
    unsealed; without one it only waits for start order.
    """
    svc: dict[str, object] = {
        "image": gateway.image,
        "cap_add": list(gateway.cap_add),
    }
    if gateway.command:
        svc["command"] = list(gateway.command)
    if gateway.env:
        svc["environment"] = dict(gateway.env)
    if gateway.healthcheck:
        svc["healthcheck"] = dict(gateway.healthcheck)

    # Drop the network-privileged capabilities from the sealed client. The seal's redirect
    # exemption is keyed on a firewall mark, and SETTING a socket mark (SO_MARK) needs only
    # CAP_NET_RAW since Linux 5.17 (CAP_NET_ADMIN before) -- and CAP_NET_RAW is in Docker's
    # DEFAULT capability set. Without dropping these, ``main`` could forge the gateway's
    # fwmark and RETURN-exempt its own egress past the interceptor, or edit the redirect
    # rules outright (CAP_NET_ADMIN), voiding the seal. We drop ONLY the network caps (not
    # ALL): the client still needs ordinary build caps -- e.g. CAP_FOWNER for a setuptools
    # ``pip install .`` that timestamps files staged into the container from another uid.
    main: dict[str, object] = {
        "network_mode": f"service:{service_name}",
        "cap_drop": ["NET_ADMIN", "NET_RAW"],
    }
    if gateway.healthcheck:
        main["depends_on"] = {service_name: {"condition": "service_healthy"}}
    else:
        main["depends_on"] = [service_name]

    compose = {"services": {service_name: svc, "main": main}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_resources_compose_file(
    path: Path,
    *,
    cpu_request: int | None = None,
    cpu_limit: int | None = None,
    memory_request_mb: int | None = None,
    memory_limit_mb: int | None = None,
    gpu_count: int | None = None,
    gpu_device_ids: list[str] | None = None,
) -> Path:
    """Write a compose override for services.main resource requests/limits.

    When ``gpu_count`` > 0 a Compose GPU device reservation is added
    (``deploy.resources.reservations.devices`` — the documented Compose GPU
    mechanism), reserving that many NVIDIA GPUs for the main service. Requires
    the NVIDIA Container Toolkit on the host.
    """
    resources: dict[str, object] = {}
    limits: dict[str, str] = {}
    reservations: dict[str, object] = {}

    if cpu_limit is not None:
        limits["cpus"] = str(cpu_limit)
    if memory_limit_mb is not None:
        limits["memory"] = f"{memory_limit_mb}M"
    if cpu_request is not None:
        reservations["cpus"] = str(cpu_request)
    if memory_request_mb is not None:
        reservations["memory"] = f"{memory_request_mb}M"
    if gpu_device_ids:
        reservations["devices"] = [
            {
                "driver": "nvidia",
                "device_ids": list(gpu_device_ids),
                "capabilities": ["gpu"],
            }
        ]
    elif gpu_count is not None and gpu_count > 0:
        reservations["devices"] = [
            {"driver": "nvidia", "count": gpu_count, "capabilities": ["gpu"]}
        ]

    if limits:
        resources["limits"] = limits
    if reservations:
        resources["reservations"] = reservations
    main = {"deploy": {"resources": resources}} if resources else {}
    compose = {"services": {"main": main}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def self_bind_mount(mount: ServiceVolumeConfig) -> ServiceVolumeConfig:
    """Return a copy of *mount* with ``source`` set equal to ``target``.

    Used by cloud providers whose docker compose "host" filesystem is the VM,
    not the user's machine. Binding ``target → target`` lets task-author
    compose files share the same dir between services without each one
    having to know the cloud provider's internal VM path layout.
    """
    new_mount: ServiceVolumeConfig = {
        "type": mount["type"],
        "source": mount["target"],
        "target": mount["target"],
    }
    if mount.get("read_only"):
        new_mount["read_only"] = True
    if "bind" in mount:
        new_mount["bind"] = mount["bind"]
    if "volume" in mount:
        new_mount["volume"] = mount["volume"]
    if "image" in mount:
        new_mount["image"] = mount["image"]
    return new_mount
