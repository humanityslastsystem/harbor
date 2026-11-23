"""Unit tests for the egress-allowlist capability: LaunchSpec fields, the
EnvironmentCapabilities flag, capability validation, the Docker egress-gateway
compose override, and Modal outbound_cidr_allowlist plumbing. No network is
touched -- the Modal SDK is mocked."""

import json

import pytest

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.launch import EgressGatewaySpec, LaunchSpec


# --- LaunchSpec fields + capability flag + validation -------------------------


def test_launchspec_has_egress_fields():
    spec = LaunchSpec(image="x", outbound_cidr_allowlist=["10.0.0.1/32"])
    assert spec.outbound_cidr_allowlist == ["10.0.0.1/32"]
    assert spec.egress_gateway is None


def test_launchspec_egress_defaults_none():
    spec = LaunchSpec(image="x")
    assert spec.outbound_cidr_allowlist is None
    assert spec.egress_gateway is None


def test_egress_gateway_spec_fields():
    gw = EgressGatewaySpec(
        image="gateway:latest",
        command=["/entrypoint.sh"],
        env={"HLS_EGRESS_POLICY": "{}"},
    )
    assert gw.image == "gateway:latest"
    assert gw.command == ["/entrypoint.sh"]
    assert gw.env == {"HLS_EGRESS_POLICY": "{}"}
    assert gw.cap_add == ("NET_ADMIN",)
    assert gw.healthcheck is None


def test_egress_gateway_spec_defaults():
    gw = EgressGatewaySpec(image="gateway:latest")
    assert gw.command == []
    assert gw.env == {}
    assert gw.cap_add == ("NET_ADMIN",)
    assert gw.healthcheck is None


def test_launchspec_carries_egress_gateway():
    gw = EgressGatewaySpec(image="gateway:latest", cap_add=("NET_ADMIN", "NET_RAW"))
    spec = LaunchSpec(image="x", egress_gateway=gw)
    assert spec.egress_gateway is gw
    assert spec.egress_gateway.cap_add == ("NET_ADMIN", "NET_RAW")


def test_capability_flag_default_false():
    assert EnvironmentCapabilities().egress_allowlist is False


def test_capability_flag_can_be_set_true():
    assert EnvironmentCapabilities(egress_allowlist=True).egress_allowlist is True


class _NoEgressEnv:
    """Minimal stand-in exposing only what _validate_launch_capabilities reads:
    a ``capabilities`` property and ``type()``. Bound to the real validator so we
    exercise the actual capability gate without a heavy backend construction."""

    from harbor.environments.base import BaseEnvironment

    _validate_launch_capabilities = BaseEnvironment._validate_launch_capabilities

    @staticmethod
    def type() -> str:
        return "no-egress"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # disable_internet/serves true so only the egress gate can fail here.
        return EnvironmentCapabilities(
            disable_internet=True, serves=True, egress_allowlist=False
        )


def test_validate_rejects_cidr_allowlist_without_capability():
    env = _NoEgressEnv()
    spec = LaunchSpec(image="x", outbound_cidr_allowlist=["10.0.0.1/32"])
    with pytest.raises(ValueError, match="egress allowlist"):
        env._validate_launch_capabilities(spec)


def test_validate_rejects_egress_gateway_without_capability():
    env = _NoEgressEnv()
    gw = EgressGatewaySpec(image="gateway:latest")
    spec = LaunchSpec(image="x", egress_gateway=gw)
    with pytest.raises(ValueError, match="egress allowlist"):
        env._validate_launch_capabilities(spec)


def test_validate_allows_egress_when_capable():
    class _CapableEnv(_NoEgressEnv):
        @property
        def capabilities(self) -> EnvironmentCapabilities:
            return EnvironmentCapabilities(
                disable_internet=True, serves=True, egress_allowlist=True
            )

    env = _CapableEnv()
    spec = LaunchSpec(image="x", outbound_cidr_allowlist=["10.0.0.1/32"])
    # Must not raise.
    env._validate_launch_capabilities(spec)


def test_validate_no_egress_request_is_noop():
    env = _NoEgressEnv()
    spec = LaunchSpec(image="x")  # no egress policy
    env._validate_launch_capabilities(spec)


# --- Docker egress-gateway sidecar --------------------------------------------


def test_write_egress_gateway_compose_file(tmp_path):
    from harbor.environments.docker import write_egress_gateway_compose_file

    gw = EgressGatewaySpec(
        image="gateway:latest",
        env={"HLS_EGRESS_POLICY": "{}"},
        healthcheck={"test": ["CMD", "true"], "interval": "1s"},
    )
    path = tmp_path / "sub" / "docker-compose-egress.json"
    result = write_egress_gateway_compose_file(path, gateway=gw)
    assert result == path
    compose = json.loads(path.read_text())

    # main joins the gateway's netns and waits for it to be healthy (never unsealed).
    main = compose["services"]["main"]
    assert main["network_mode"] == "service:egress-gateway"
    assert main["depends_on"] == {"egress-gateway": {"condition": "service_healthy"}}
    # main drops the network-privileged caps: the redirect exemption is keyed on a firewall
    # mark, and setting SO_MARK needs only CAP_NET_RAW (a Docker default cap) since kernel
    # 5.17, so a NET_RAW/NET_ADMIN-bearing client could forge the mark and bypass the seal.
    # Only the network caps are dropped (ordinary build caps like FOWNER stay). Root
    # invariant -- enforced here, not just documented.
    assert main["cap_drop"] == ["NET_ADMIN", "NET_RAW"]

    # the gateway carries NET_ADMIN, its env, and its healthcheck.
    gw_svc = compose["services"]["egress-gateway"]
    assert gw_svc["image"] == "gateway:latest"
    assert gw_svc["cap_add"] == ["NET_ADMIN"]
    assert gw_svc["environment"] == {"HLS_EGRESS_POLICY": "{}"}
    assert gw_svc["healthcheck"] == {"test": ["CMD", "true"], "interval": "1s"}


def test_write_egress_gateway_compose_file_no_healthcheck(tmp_path):
    from harbor.environments.docker import write_egress_gateway_compose_file

    gw = EgressGatewaySpec(image="gateway:latest")
    path = tmp_path / "docker-compose-egress.json"
    write_egress_gateway_compose_file(path, gateway=gw, service_name="gw")
    compose = json.loads(path.read_text())
    main = compose["services"]["main"]
    assert main["network_mode"] == "service:gw"
    # Without a healthcheck, main only orders after the gateway (no health wait).
    assert main["depends_on"] == ["gw"]
    # An empty command leaves the image ENTRYPOINT in charge.
    assert "command" not in compose["services"]["gw"]


def test_docker_capability_egress_allowlist_true():
    from harbor.environments.docker.docker import DockerEnvironment

    env = DockerEnvironment.launch(LaunchSpec(image="busybox:latest"))
    assert env.capabilities.egress_allowlist is True


def test_docker_launch_stores_egress_gateway():
    from harbor.environments.docker.docker import DockerEnvironment

    gw = EgressGatewaySpec(image="gateway:latest")
    env = DockerEnvironment.launch(
        LaunchSpec(image="busybox:latest", egress_gateway=gw)
    )
    assert env._egress_gateway is gw


def test_docker_no_egress_gateway_by_default():
    from harbor.environments.docker.docker import DockerEnvironment

    env = DockerEnvironment.launch(LaunchSpec(image="busybox:latest"))
    assert env._egress_gateway is None


def test_egress_gateway_is_authoritative_over_allow_internet(tmp_path):
    """A gateway-sealed env must NOT also apply the native-block override when
    allow_internet=False: ``main`` joins the gateway netns (``network_mode: service``),
    which is mutually exclusive with the internal-network / none overrides. The seal
    field is authoritative over the coarse bool, so a mis-set allow_internet can't
    produce a conflicting compose merge."""
    from harbor.environments.docker.docker import DockerEnvironment

    gw = EgressGatewaySpec(image="gateway:latest")
    env = DockerEnvironment.launch(
        LaunchSpec(image="busybox:latest", egress_gateway=gw, allow_internet=False)
    )
    # Simulate the post-setup compose-file state that start() would write.
    egress = tmp_path / "egress.yml"
    egress.write_text("services: {}\n")
    net = tmp_path / "net.yml"
    net.write_text("services: {}\n")
    env._egress_compose_path = egress
    env._network_compose_path = net

    paths = env._docker_compose_paths
    assert egress in paths  # the gateway seal IS applied
    assert net not in paths  # the native block is NOT also applied (gateway wins)


def test_native_block_applies_without_gateway(tmp_path):
    """With no gateway, allow_internet=False still applies the native internal-network
    block (air-gap must keep working)."""
    from harbor.environments.docker.docker import DockerEnvironment

    env = DockerEnvironment.launch(
        LaunchSpec(image="busybox:latest", allow_internet=False)
    )
    net = tmp_path / "net.yml"
    net.write_text("services: {}\n")
    env._network_compose_path = net

    paths = env._docker_compose_paths
    assert net in paths  # native block applied (no gateway to supersede it)


def test_network_holder_is_gateway_when_sealed():
    """When sealed behind the egress gateway, the netns/identity holder -- the service the
    eval network attaches to and endpoint() resolves -- is the gateway, not main (which has
    no NIC of its own under network_mode:service). This is what makes a SERVING container
    reachable behind the gateway."""
    from harbor.environments.docker.docker import DockerEnvironment

    gw = EgressGatewaySpec(image="gateway:latest")
    sealed = DockerEnvironment.launch(
        LaunchSpec(image="busybox:latest", egress_gateway=gw)
    )
    assert sealed._network_holder() == "egress-gateway"

    plain = DockerEnvironment.launch(LaunchSpec(image="busybox:latest"))
    assert plain._network_holder() == "main"


# --- Modal outbound_cidr_allowlist plumbing -----------------------------------

pytest.importorskip("modal")


def _make_modal_serve_env(temp_dir, *, outbound_cidr_allowlist=None):
    """Construct a serve-mode ModalEnvironment with an egress allowlist (no SDK
    calls). Mirrors the helper in tests/unit/environments/test_modal_serve.py."""
    from harbor.environments.modal import ModalEnvironment
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

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
            docker_image="img:latest",
            allow_internet=True,
        ),
        _serve_mode=True,
        _service_command=["python", "server.py"],
        _service_port=None,  # keepalive: no inbound tunnel, just outbound seal
        _serve_image="img:latest",
        _launch_outbound_cidr_allowlist=outbound_cidr_allowlist,
    )


def _patch_modal_sdk(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from harbor.environments import modal as modal_mod

    fake_sandbox = MagicMock(name="sandbox")
    fake_image = MagicMock(name="image")
    fake_app = MagicMock(name="app")
    create = AsyncMock(return_value=fake_sandbox)
    lookup = AsyncMock(return_value=fake_app)
    from_registry = MagicMock(return_value=fake_image)
    from_dict = MagicMock(return_value=MagicMock(name="env-secret"))

    monkeypatch.setattr(modal_mod.Sandbox, "create", MagicMock(aio=create))
    monkeypatch.setattr(modal_mod.App, "lookup", MagicMock(aio=lookup))
    monkeypatch.setattr(modal_mod.Image, "from_registry", from_registry)
    monkeypatch.setattr(modal_mod.Secret, "from_dict", from_dict)
    return SimpleNamespace(create=create, sandbox=fake_sandbox, image=fake_image)


def test_modal_serve_capability_egress_allowlist_true(temp_dir):
    env = _make_modal_serve_env(temp_dir)
    assert env.capabilities.egress_allowlist is True


def test_modal_launch_stores_outbound_cidr_allowlist():
    from harbor.environments.modal import ModalEnvironment

    env = ModalEnvironment.launch(
        LaunchSpec(
            image="img:latest",
            serve_command=["python", "s.py"],
            service_port=8000,
            outbound_cidr_allowlist=["10.0.0.1/32"],
        )
    )
    assert env._launch_outbound_cidr_allowlist == ["10.0.0.1/32"]


def test_modal_launch_outbound_cidr_defaults_none():
    from harbor.environments.modal import ModalEnvironment

    env = ModalEnvironment.launch(
        LaunchSpec(image="img:latest", serve_command=["x"], service_port=8000)
    )
    assert env._launch_outbound_cidr_allowlist is None


async def test_modal_start_forwards_outbound_cidr_allowlist(temp_dir, monkeypatch):
    env = _make_modal_serve_env(
        temp_dir, outbound_cidr_allowlist=["10.0.0.1/32", "10.0.0.2/32"]
    )
    sdk = _patch_modal_sdk(monkeypatch)

    await env.start(force_build=False)

    _, kwargs = sdk.create.call_args
    assert kwargs["outbound_cidr_allowlist"] == ["10.0.0.1/32", "10.0.0.2/32"]
    # An allowlist cannot combine with block_network=True; sealing is realized as
    # outbound-restricted, not block-all.
    assert kwargs["block_network"] is False


async def test_modal_start_no_allowlist_omits_kwarg(temp_dir, monkeypatch):
    env = _make_modal_serve_env(temp_dir, outbound_cidr_allowlist=None)
    sdk = _patch_modal_sdk(monkeypatch)

    await env.start(force_build=False)

    _, kwargs = sdk.create.call_args
    assert "outbound_cidr_allowlist" not in kwargs
