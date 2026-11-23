"""Unit test for DockerEnvironment.container_logs (no real Docker)."""

import pytest

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.launch import LaunchSpec


def test_container_logs_returns_combined_output(monkeypatch):
    env = DockerEnvironment.launch(LaunchSpec(image="x:latest"))
    monkeypatch.setattr(env, "_compose_ps_quiet", lambda service: "abc123")

    class _Result:
        returncode = 0
        stdout = "hello-from-container\n"
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        assert cmd[:2] == ["docker", "logs"]
        assert "abc123" in cmd
        return _Result()

    monkeypatch.setattr("harbor.environments.docker.docker.subprocess.run", _fake_run)
    assert "hello-from-container" in env.container_logs("main")


def test_container_logs_raises_when_not_running(monkeypatch):
    env = DockerEnvironment.launch(LaunchSpec(image="x:latest"))
    monkeypatch.setattr(env, "_compose_ps_quiet", lambda service: "")
    with pytest.raises(RuntimeError, match="not running"):
        env.container_logs("main")
