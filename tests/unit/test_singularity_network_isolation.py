"""Network isolation for an environment that declares ``allow_internet=False``.

Apptainer's ``none`` network is the one an unprivileged user may request, and it leaves
the container with only a loopback interface, so the seal is by absence: there is no
egress to filter and nothing inside to reopen.

Asserts on the constructed command without starting a container. Whether a real
rootless apptainer accepts the flags is checked by
``tests/integration/test_singularity_network_isolation_live.py``.
"""

from pathlib import Path

from harbor.environments.launch import LaunchSpec
from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _sif(tmp_path: Path) -> str:
    p = tmp_path / "img.sif"
    p.write_bytes(b"")
    return str(p)


def _command(env, **kwargs):
    defaults = dict(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    return env._build_singularity_command(**{**defaults, **kwargs})


def test_a_sealed_spec_isolates_the_network(tmp_path):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), allow_internet=False)
    )
    assert env._isolate_network is True
    cmd = _command(env)
    # --network is only accepted alongside --net, and both precede the image.
    assert cmd[cmd.index("--net") : cmd.index("--net") + 3] == [
        "--net",
        "--network",
        "none",
    ]
    assert cmd.index("--net") < cmd.index("/x/img.sif")


def test_an_open_spec_adds_no_network_flags(tmp_path):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), allow_internet=True)
    )
    assert env._isolate_network is False
    cmd = _command(env)
    assert "--net" not in cmd
    assert "--network" not in cmd


def test_a_trial_that_declares_no_internet_is_isolated_too(tmp_path):
    # The trial path reads the same declaration. Accepting it there without the flags
    # would leave the container open.
    env = SingularityEnvironment(
        environment_dir=tmp_path,
        environment_name="t",
        session_id="t",
        trial_paths=TrialPaths(trial_dir=tmp_path),
        task_env_config=EnvironmentConfig(
            docker_image=_sif(tmp_path), allow_internet=False
        ),
    )
    assert "--net" in _command(env)


def test_isolation_does_not_drag_in_the_producer_seal(tmp_path):
    # The prefix seal filters egress and drops CAP_NET_ADMIN so the payload cannot
    # rewrite its own firewall. Isolation has no firewall, so it needs neither.
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), allow_internet=False)
    )
    cmd = _command(env)
    assert "--drop-caps" not in cmd
    assert env._seal_command_prefix == ()


def test_the_control_channel_is_a_socket_when_isolated(tmp_path):
    # A TCP listener inside a private netns is unreachable from the host, so the
    # server listens on a UNIX socket on the bind-mounted staging directory.
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), allow_internet=False)
    )
    cmd = _command(env, port=None, uds="/staging/control.sock")
    assert "--uds" in cmd
    assert "--port" not in cmd


def test_the_control_channel_is_tcp_when_open(tmp_path):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), allow_internet=True)
    )
    cmd = _command(env)
    assert "--port" in cmd
    assert "--uds" not in cmd


def test_singularity_declares_disable_internet(tmp_path):
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path)))
    assert env.capabilities.disable_internet is True
