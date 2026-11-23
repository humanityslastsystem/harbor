import asyncio
from unittest.mock import Mock

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.launch import LaunchSpec


class _Proc:
    def __init__(self, rc, out):
        self.returncode = rc
        self._out = out

    async def communicate(self):
        return (self._out, b"")


def _docker_env_self(image):
    self = Mock()
    self.task_env_config.docker_image = image
    return self


def test_docker_image_digest_returns_id(monkeypatch):
    async def _fake_exec(*a, **k):
        return _Proc(0, b"sha256:abc123\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    digest = asyncio.run(DockerEnvironment.image_digest(_docker_env_self("img:latest")))
    assert digest == "sha256:abc123"


def test_docker_image_digest_none_on_inspect_failure(monkeypatch):
    async def _fake_exec(*a, **k):
        return _Proc(1, b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    assert (
        asyncio.run(DockerEnvironment.image_digest(_docker_env_self("img:latest")))
        is None
    )


def test_docker_image_digest_none_when_no_image():
    assert asyncio.run(DockerEnvironment.image_digest(_docker_env_self(None))) is None


def test_docker_image_digest_none_when_exec_raises(monkeypatch):
    # The never-raises contract: a failure to even launch docker (e.g. docker
    # not installed) returns None, not an exception.
    async def _raise(*a, **k):
        raise OSError("docker not found")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise)
    self = _docker_env_self("img:latest")
    assert asyncio.run(DockerEnvironment.image_digest(self)) is None
    self.logger.debug.assert_called_once()


def test_docker_launch_is_prebuilt_and_serves():
    env = DockerEnvironment.launch(
        LaunchSpec(
            image="busybox:latest",
            serve_command=["--port", "8000"],
            env={"FOO": "1"},
            gpu_device_ids=["0"],
            allow_internet=False,
        )
    )
    assert env.is_prebuilt is True
    assert (
        env.capabilities.serves
        and env.capabilities.gpu_pinning
        and env.capabilities.can_commit
    )
    assert env._service_command == ["--port", "8000"]
    assert env._service_env == {"FOO": "1"}
    assert env._gpu_device_ids == ["0"]
    # prebuilt validation must not raise (no on-disk Dockerfile needed):
    env._validate_definition()
