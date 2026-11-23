"""The provider block takes its base URL from the exec's environment, never from its command.

A relay's base URL is a secret like the credential. A command's text is logged (by harbor at
debug, by an exec server, in a failed command's error) where its environment is not, so the
block's command names the variable and the shell expands it, as the auth file's credential is.
"""

import os
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.codex import Codex

KEY = "sk-test-0123456789"
HOST = "relay.example"
URL = f"https://{HOST}/v1"
_CONFIG = '"$CODEX_HOME/config.toml"'


async def _calls(agent: Codex) -> list[tuple[str, dict[str, str]]]:
    """Every command run() sends the environment, with the env it sends it with, in order."""
    env = AsyncMock()
    env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    await agent.run("the brief", env, AsyncMock())
    return [
        (c.kwargs["command"], c.kwargs["env"] or {}) for c in env.exec.call_args_list
    ]


def _agent(tmp_path: Path, monkeypatch, extra_env: dict[str, str]) -> Codex:
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_AUTH_JSON_PATH",
        "CODEX_FORCE_AUTH_JSON",
    ):
        monkeypatch.delenv(var, raising=False)
    return Codex(
        logs_dir=tmp_path / "logs", model_name="openai/gpt-5.5", extra_env=extra_env
    )


def _bash(command: str, home: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", command],
        env={"PATH": os.environ["PATH"], **env, "CODEX_HOME": str(home)},
        capture_output=True,
        text=True,
    )


@pytest.mark.asyncio
async def test_no_command_holds_the_base_url_and_the_block_is_written_from_the_env(
    tmp_path, monkeypatch
):
    agent = _agent(
        tmp_path, monkeypatch, {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": URL}
    )
    calls = await _calls(agent)

    for command, _ in calls:
        assert URL not in command and HOST not in command and KEY not in command
    home = tmp_path / "codex-home"
    home.mkdir()
    for command, env in calls:
        if _CONFIG in command and "codex-config.toml" not in command:
            assert _bash(command, home, env).returncode == 0, command
    cfg = tomllib.loads((home / "config.toml").read_text())
    assert cfg["model_provider"] == "custom"
    assert cfg["model_providers"]["custom"] == {
        "name": "custom",
        "base_url": URL,
        "wire_api": "responses",
        "requires_openai_auth": True,
    }


@pytest.mark.asyncio
async def test_the_provider_block_without_a_base_url_in_its_env_fails_loud(
    tmp_path, monkeypatch
):
    agent = _agent(
        tmp_path, monkeypatch, {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": URL}
    )
    [(provider, _)] = [
        (c, e) for c, e in await _calls(agent) if "[model_providers.custom]" in c
    ]
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text('web_search = "disabled"\n')

    done = _bash(provider, home, {})

    assert done.returncode != 0
    assert (home / "config.toml").read_text() == 'web_search = "disabled"\n'


def test_without_a_base_url_there_is_no_provider_block(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, {"OPENAI_API_KEY": KEY})
    assert agent._build_provider_config_command() is None
    with_url = _agent(
        tmp_path, monkeypatch, {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": URL}
    )
    assert "[model_providers.custom]" in with_url._build_provider_config_command()


@pytest.mark.asyncio
async def test_the_block_is_written_as_it_reads_even_with_printf_directives_in_it(
    tmp_path, monkeypatch
):
    """The block is printf's argument, never its format: a % or a \\ in it stays as it is."""
    template = Codex._PROVIDER_TOML.replace(
        "[model_providers.custom]",
        "# 100% sure: a \\ and a \\n stay\n[model_providers.custom]",
    )
    monkeypatch.setattr(Codex, "_PROVIDER_TOML", template)
    agent = _agent(
        tmp_path, monkeypatch, {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": URL}
    )
    [(provider, env)] = [
        (c, e) for c, e in await _calls(agent) if "[model_providers.custom]" in c
    ]
    home = tmp_path / "codex-home"
    home.mkdir()

    assert _bash(provider, home, env).returncode == 0
    assert (home / "config.toml").read_text() == template.replace("{base_url}", URL)
