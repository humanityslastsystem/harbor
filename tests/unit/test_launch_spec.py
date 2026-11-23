from unittest.mock import Mock

import pytest

from harbor.environments.base import BaseEnvironment
from harbor.environments.launch import LaunchSpec
from harbor.environments.capabilities import EnvironmentCapabilities


def test_launchspec_minimal_defaults():
    spec = LaunchSpec(image="img:latest")
    assert spec.serve_command is None
    assert spec.gpus == 0 and spec.gpu_device_ids is None
    assert spec.allow_internet is True and spec.workdir == "/workspace"
    assert spec.env == {} and spec.mounts == []


def test_capability_flags_default_false():
    caps = EnvironmentCapabilities()
    assert (
        caps.serves is False and caps.can_commit is False and caps.gpu_pinning is False
    )


def test_launchspec_sandbox_fields_default_to_todays_launch():
    assert EnvironmentCapabilities().overlay_dirs is False
    spec = LaunchSpec(image="img.sif")
    assert spec.overlay_dir is None and spec.lower_overlay_dirs == ()
    assert spec.server_python is None
    assert spec.start_in_staging_dir is False
    assert spec.ready_poll_sec == 1.0


@pytest.mark.parametrize(
    "fields",
    [{"overlay_dir": "/overlays/s0"}, {"lower_overlay_dirs": ("/prefixes/a",)}],
)
def test_a_backend_without_overlay_dirs_refuses_them(fields):
    # Ignoring them would start the container from its image alone, without the state
    # the caller asked it to start from.
    env = Mock()
    env.capabilities = EnvironmentCapabilities()
    env.type.return_value = "dummy"
    with pytest.raises(ValueError, match="overlay_dirs=False"):
        BaseEnvironment._validate_launch_capabilities(
            env, LaunchSpec(image="img", **fields)
        )


def test_a_backend_with_overlay_dirs_accepts_them():
    env = Mock()
    env.capabilities = EnvironmentCapabilities(overlay_dirs=True)
    BaseEnvironment._validate_launch_capabilities(
        env, LaunchSpec(image="img", overlay_dir="/o", lower_overlay_dirs=("/p",))
    )
