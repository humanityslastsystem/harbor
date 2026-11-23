"""Codex's hosted web search is off for every run, and a run that used it still parses.

Codex offers a hosted `web_search` tool unless its config says otherwise. The search
runs on the model provider's side, so no egress rule in the build sandbox sees it. The
adapter disables it in the config.toml it writes and again with a `-c` override on both
codex calls, since a `.codex/config.toml` in the agent's working directory outranks the
one in $CODEX_HOME.
"""

import json
import os
import shlex
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.codex import Codex
from harbor.models.task.config import MCPServerConfig

_CONFIG = '"$CODEX_HOME/config.toml"'


async def _run(agent: Codex) -> list[str]:
    """Every command run() sends the environment, in order."""
    env = AsyncMock()
    env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    await agent.run("the brief", env, AsyncMock())
    return [c.kwargs["command"] for c in env.exec.call_args_list]


def _write_config(commands: list[str], home: Path) -> dict:
    """Run the commands that write config.toml with bash and parse what they wrote.

    Executed rather than inspected, so the result is the file codex would load: the
    order of the writes decides which table a bare key lands in, and a repeated key makes
    the file unloadable.
    """
    for command in commands:
        if _CONFIG in command and "codex-config.toml" not in command:
            subprocess.run(
                ["bash", "-c", command],
                check=True,
                env={**os.environ, "CODEX_HOME": str(home)},
            )
    return tomllib.loads((home / "config.toml").read_text())


def _agent(tmp_path: Path, monkeypatch, **kwargs) -> Codex:
    """A Codex adapter whose secrets directory lives under tmp_path."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    monkeypatch.setattr(Codex, "_REMOTE_CODEX_SECRETS_DIR", PurePosixPath(secrets))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    return Codex(logs_dir=tmp_path / "logs", model_name="openai/gpt-5.5", **kwargs)


@pytest.mark.asyncio
async def test_generated_config_disables_web_search_ahead_of_every_table(
    tmp_path, monkeypatch
):
    """The key is top level even though the provider and MCP blocks open tables after it."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://relay.example")
    servers = [
        MCPServerConfig(
            name="budget", transport="stdio", command="python3", args=["/s.py"]
        )
    ]
    agent = _agent(tmp_path, monkeypatch, mcp_servers=servers)
    home = tmp_path / "codex-home"
    home.mkdir()

    cfg = _write_config(await _run(agent), home)

    assert cfg["web_search"] == "disabled"
    assert cfg["model_provider"] == "custom"
    assert "web_search" not in cfg["model_providers"]["custom"]
    assert "web_search" not in cfg["mcp_servers"]["budget"]


@pytest.mark.asyncio
async def test_config_disables_web_search_without_a_provider_or_mcp(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = _agent(tmp_path, monkeypatch)
    home = tmp_path / "codex-home"
    home.mkdir()

    assert _write_config(await _run(agent), home) == {"web_search": "disabled"}


@pytest.mark.asyncio
async def test_setup_run_twice_rebuilds_the_same_config(tmp_path, monkeypatch):
    """A second setup into the same $CODEX_HOME must not repeat keys or tables."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://relay.example")
    servers = [MCPServerConfig(name="budget", transport="stdio", command="python3")]
    agent = _agent(tmp_path, monkeypatch, mcp_servers=servers)
    home = tmp_path / "codex-home"
    home.mkdir()
    commands = await _run(agent)

    first = _write_config(commands, home)
    second = _write_config(commands, home)

    assert second == first


@pytest.mark.asyncio
async def test_both_codex_calls_override_web_search(tmp_path, monkeypatch):
    """The override is what beats a `.codex/config.toml` the agent writes itself."""
    agent = Codex(
        logs_dir=tmp_path, model_name="openai/gpt-5.5", goal_kickoff="Create a goal"
    )
    calls = [c for c in await _run(agent) if "codex exec " in c]
    assert len(calls) == 2

    for call in calls:
        tokens = shlex.split(call.split(" -- ", 1)[0])
        overrides = [tokens[i + 1] for i, t in enumerate(tokens) if t == "-c"]
        web_search = [v for v in overrides if v.startswith("web_search")]
        assert web_search, call
        assert all(tomllib.loads(v) == {"web_search": "disabled"} for v in web_search)


def _session(directory: Path, items: list[dict]) -> None:
    records = [
        {
            "timestamp": "2026-01-01T00:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "root", "cli_version": "0.146.0", "source": "exec"},
        },
        {"type": "turn_context", "payload": {"model": "gpt-5.5"}},
    ]
    records += [
        {"timestamp": "2026-01-01T00:05:00.000Z", "type": "response_item", "payload": p}
        for p in items
    ]
    (directory / "rollout-root.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )


def test_web_search_items_still_parse_into_tool_calls(tmp_path):
    """A session that used web search carries web_search_call items; they stay readable.

    Both shapes are ones codex 0.146.0 writes: an item with no action at all, and one
    whose action names the query.
    """
    _session(
        tmp_path,
        [
            {
                "type": "web_search_call",
                "id": "ws_0001",
                "status": "completed",
            },
            {
                "type": "web_search_call",
                "id": "ws_0002",
                "status": "completed",
                "action": {"type": "search", "query": "test", "queries": ["test"]},
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        ],
    )
    agent = Codex(logs_dir=tmp_path / "logs", model_name="openai/gpt-5.5")

    trajectory = agent._convert_events_to_trajectory(tmp_path)

    assert trajectory is not None
    searches = [s for s in trajectory.steps if s.tool_calls]
    calls = [call for s in searches for call in s.tool_calls or []]
    assert [c.function_name for c in calls] == ["web_search_call", "web_search_call"]
    assert calls[0].arguments == {"action_type": ""}
    assert calls[1].arguments == {
        "action_type": "search",
        "query": "test",
        "queries": ["test"],
    }
    assert all(s.extra == {"status": "completed"} for s in searches)
    assert trajectory.steps[-1].message == "done"
