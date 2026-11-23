"""SingularityEnvironment.launch(LaunchSpec) adapter + --nv GPU support.

Mirrors test_docker_launch: a LaunchSpec-built env runs a prebuilt image with no
on-disk task definition. The GPU case threads ``--nv`` into the apptainer command;
the CPU case omits it. We assert on the constructed command (via the extracted
command builder) without starting a real container.
"""

import tempfile
from pathlib import Path

import pytest

from harbor.environments.launch import LaunchSpec
from harbor.environments.singularity.singularity import SingularityEnvironment


def _sif(tmp_path: Path) -> str:
    p = tmp_path / "img.sif"
    p.write_bytes(
        b""
    )  # a real (empty) file so _validate_definition's existence check passes
    return str(p)


def test_launch_is_prebuilt_keepalive(tmp_path):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), gpus=1, env={"FOO": "1"}, workdir="/workspace")
    )
    assert env.is_prebuilt is True
    assert env._nv is True
    # env threaded onto the persistent env so it reaches exec'd processes.
    assert env._persistent_env.get("FOO") == "1"
    # prebuilt validation must not raise (no on-disk Dockerfile/def needed):
    env._validate_definition()


def test_launch_gpu_command_includes_nv(tmp_path):
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path), gpus=1))
    cmd = env._build_singularity_command(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    assert "--nv" in cmd
    # --nv precedes the .sif image argument (it is a singularity exec flag).
    assert cmd.index("--nv") < cmd.index("/x/img.sif")


def test_launch_cpu_command_omits_nv(tmp_path):
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path), gpus=0))
    assert env._nv is False
    cmd = env._build_singularity_command(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    assert "--nv" not in cmd


def test_launch_rejects_device_id_pinning(tmp_path):
    # Singularity does not pin GPU device IDs -- Slurm owns GPU allocation and ``--nv``
    # exposes whatever the job was granted. A LaunchSpec that asks for device-ID pinning
    # is rejected loudly (gpu_pinning=False), so the substrate must request ``gpus`` (a
    # count), not ``gpu_device_ids``.
    with pytest.raises(ValueError, match="(?i)gpu"):
        SingularityEnvironment.launch(
            LaunchSpec(image=_sif(tmp_path), gpus=1, gpu_device_ids=["0", "1"])
        )


def test_launch_rejects_non_sif_handle():
    # A non-str image (e.g. a foreign backend's image handle) is rejected loudly.
    with pytest.raises(ValueError, match="(?i)image"):
        SingularityEnvironment.launch(LaunchSpec(image=object(), gpus=0))


def test_launch_seal_command_prefix_wraps_exec(tmp_path):
    # The HLS producer seal passes an argv PREFIX (a ``pasta`` + iptables
    # wrapper); the backend prepends it VERBATIM to the singularity exec command so the
    # keepalive server spawns inside the sealed namespace. "$@" inside the wrapper then
    # is exactly the singularity exec argv.
    prefix = ("bash", "-c", 'echo seal; exec "$@"', "hls-seal")
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), gpus=0, seal_command_prefix=prefix)
    )
    assert env._seal_command_prefix == prefix
    cmd = env._build_singularity_command(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    # prefix is at the very front, immediately followed by the real exec command
    assert cmd[: len(prefix)] == list(prefix)
    assert cmd[len(prefix)] == "singularity"
    assert cmd[len(prefix) + 1] == "exec"


def test_launch_no_seal_prefix_is_unchanged(tmp_path):
    # Open mode (empty prefix) -> the launch command starts with singularity exec.
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path), gpus=0))
    assert env._seal_command_prefix == ()
    cmd = env._build_singularity_command(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    assert cmd[0] == "singularity"
    assert cmd[1] == "exec"


def test_launch_seal_drops_net_admin_caps(tmp_path):
    # Under the producer seal the payload runs as root in the seal's user namespace; without
    # dropping CAP_NET_ADMIN it holds firewall control over the sealed netns and could
    # ``iptables -P OUTPUT ACCEPT`` to escape. The sealed launch must drop the network-admin
    # caps for the payload (apptainer ``--drop-caps``); the unsealed path must not.
    prefix = ("bash", "-c", 'exec "$@"', "hls-seal")
    sealed = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), gpus=0, seal_command_prefix=prefix)
    )._build_singularity_command(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    assert "--drop-caps" in sealed
    assert sealed[sealed.index("--drop-caps") + 1] == "CAP_NET_ADMIN,CAP_NET_RAW"
    assert sealed.index("--drop-caps") < sealed.index(
        "/x/img.sif"
    )  # an exec flag, before the image

    unsealed = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), gpus=0)
    )._build_singularity_command(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/workspace"
    )
    assert "--drop-caps" not in unsealed


def test_native_ctor_defaults_nv_false(tmp_path):
    # The harbor-native constructor path keeps GPU passthrough OFF by default.
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    env_dir = Path(tempfile.mkdtemp())
    env = SingularityEnvironment(
        environment_dir=env_dir,
        environment_name="t",
        session_id="t",
        trial_paths=TrialPaths(trial_dir=env_dir),
        task_env_config=EnvironmentConfig(docker_image=_sif(tmp_path)),
    )
    assert env._nv is False


def _command(env, **kwargs):
    args = dict(
        sif_path="/x/img.sif", staging_dir="/staging", port=9000, workdir="/app"
    )
    return env._build_singularity_command(**{**args, **kwargs})


#: The command a launch that sets none of the optional fields runs.
DEFAULT_COMMAND = [
    "singularity",
    "exec",
    "--no-mount",
    "home",
    "--no-mount",
    "tmp",
    "--no-mount",
    "bind-paths",
    "--pwd",
    "/app",
    "--writable-tmpfs",
    "--containall",
    "--pid",
    "-B",
    "/staging:/staging",
    "/x/img.sif",
    "bash",
    "-c",
    'exec /staging/bootstrap.sh "$@"',
    "bash",
    "/app",
    "/staging/_hbexec.py",
    "--port",
    "9000",
    "--workdir",
    "/app",
]


def test_a_launch_that_sets_no_optional_field_runs_the_default_command(tmp_path):
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path)))
    assert _command(env) == DEFAULT_COMMAND


def test_the_trial_constructor_runs_the_default_command(tmp_path):
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    env = SingularityEnvironment(
        environment_dir=tmp_path,
        environment_name="t",
        session_id="t",
        trial_paths=TrialPaths(trial_dir=tmp_path),
        task_env_config=EnvironmentConfig(docker_image=_sif(tmp_path), workdir="/app"),
    )
    assert _command(env) == DEFAULT_COMMAND


def test_launch_overlay_dir_replaces_the_writable_tmpfs(tmp_path):
    """The tmpfs lives in apptainer's session directory, which ``sessiondir max size``
    in apptainer.conf caps (64 MiB by default), so a payload that writes real output
    files runs out of space. The two are exclusive: keeping --writable-tmpfs would
    stack a capped layer over the directory."""
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), overlay_dir="/scratch/overlays/s0")
    )
    cmd = _command(env)
    assert cmd[cmd.index("--overlay") + 1] == "/scratch/overlays/s0"
    assert "--writable-tmpfs" not in cmd
    assert cmd.index("--overlay") < cmd.index("/x/img.sif")


def test_launch_without_an_overlay_dir_keeps_the_writable_tmpfs(tmp_path):
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path)))
    cmd = _command(env)
    assert "--writable-tmpfs" in cmd
    assert "--overlay" not in cmd


def test_launch_stacks_read_only_lowers_under_the_writable_layer(tmp_path):
    """Apptainer stacks overlays in the order given, so the writable layer comes last.
    The other way round, the container's writes would land under the state it starts
    from and reads would return the old contents."""
    env = SingularityEnvironment.launch(
        LaunchSpec(
            image=_sif(tmp_path),
            overlay_dir="/scratch/overlays/s0",
            lower_overlay_dirs=("/prefixes/a.state", "/prefixes/b.state"),
        )
    )
    cmd = _command(env)
    overlays = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--overlay"]
    assert overlays == [
        "/prefixes/a.state:ro",
        "/prefixes/b.state:ro",
        "/scratch/overlays/s0",
    ]
    assert "--writable-tmpfs" not in cmd


def test_a_read_only_lower_is_mounted_read_only(tmp_path):
    """Several containers share one lower at once, so none of them may write it."""
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), lower_overlay_dirs=("/prefixes/a.state",))
    )
    cmd = _command(env)
    assert cmd[cmd.index("--overlay") + 1] == "/prefixes/a.state:ro"
    # With no writable directory named, the tmpfs is still the writable layer.
    assert "--writable-tmpfs" in cmd


def test_launch_without_lowers_is_unchanged(tmp_path):
    plain = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), overlay_dir="/scratch/overlays/s0")
    )
    explicit = SingularityEnvironment.launch(
        LaunchSpec(
            image=_sif(tmp_path),
            overlay_dir="/scratch/overlays/s0",
            lower_overlay_dirs=(),
        )
    )
    assert _command(plain) == _command(explicit)


def test_singularity_declares_overlay_dirs(tmp_path):
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path)))
    assert env.capabilities.overlay_dirs is True


def test_a_named_server_python_reaches_the_container(tmp_path):
    env = SingularityEnvironment.launch(
        LaunchSpec(image=_sif(tmp_path), server_python="/opt/harbor-server/bin/python3")
    )
    cmd = _command(env)
    assert (
        cmd[cmd.index("--env") + 1]
        == "HARBOR_SERVER_PYTHON=/opt/harbor-server/bin/python3"
    )
    assert cmd.index("--env") < cmd.index("/x/img.sif")


def test_no_server_python_adds_no_env(tmp_path):
    env = SingularityEnvironment.launch(LaunchSpec(image=_sif(tmp_path)))
    assert "--env" not in _command(env)


@pytest.mark.parametrize(
    "path",
    ["opt/harbor-server/bin/python3", "/opt/a,b/bin/python3"],
)
def test_a_server_python_apptainer_cannot_carry_is_refused(tmp_path, path):
    # Relative paths have no meaning in the container, and apptainer splits ``--env``
    # on commas.
    with pytest.raises(ValueError, match="server_python"):
        SingularityEnvironment.launch(
            LaunchSpec(image=_sif(tmp_path), server_python=path)
        )
