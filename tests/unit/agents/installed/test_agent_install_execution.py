"""Tests for agent install() method execution via environment.exec()."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from harbor.agents.installed.aider import Aider
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.cursor_cli import CursorCli
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.agents.installed.goose import Goose
from harbor.agents.installed.hermes import Hermes
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.openhands import OpenHands
from harbor.agents.installed.qwen_code import QwenCode
from harbor.agents.installed.swe_agent import SweAgent

ALL_AGENTS = [
    Aider,
    ClaudeCode,
    Codex,
    CursorCli,
    GeminiCli,
    Goose,
    Hermes,
    MiniSweAgent,
    OpenCode,
    OpenHands,
    QwenCode,
    SweAgent,
]


class TestAgentInstallExecution:
    """Test that agent install() methods execute without errors against a mock environment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_class", ALL_AGENTS)
    async def test_install_calls_exec_setup(self, agent_class, temp_dir):
        """Test that install() calls environment.exec() with expected patterns."""
        with patch.dict(os.environ, clear=False):
            agent = agent_class(logs_dir=temp_dir)
            environment = AsyncMock()

            async def _exec(*args, **kwargs):
                # Simulate the agent NOT yet installed so idempotent installers (e.g.
                # ClaudeCode / Codex, which probe ``command -v <cli>`` and skip when
                # present) still take the real install path and issue the root setup
                # call. Codex skips for a load-bearing reason: an air-gapped sandbox
                # bakes the CLI because its seal blocks the npm registry, and running
                # the install there turns a working environment into a setup failure.
                cmd = kwargs.get("command", "")
                rc = (
                    1
                    if ("command -v claude" in cmd or "command -v codex" in cmd)
                    else 0
                )
                return SimpleNamespace(return_code=rc, stdout="", stderr="")

            environment.exec.side_effect = _exec
            environment.upload_file.return_value = None

            try:
                await agent.install(environment)
            except Exception:
                pass  # Some agents may fail due to missing resources

            # All agents should call exec at least once
            assert environment.exec.call_count >= 1

            # At least one call should use user="root" for system package installation
            root_calls = [
                call
                for call in environment.exec.call_args_list
                if call.kwargs.get("user") == "root"
            ]
            assert len(root_calls) >= 1, (
                f"{agent_class.__name__} should have at least one root exec call"
            )

    @pytest.mark.asyncio
    async def test_claude_code_install_skips_when_already_present(self, temp_dir):
        """ClaudeCode build images bake claude-code; when it is already on PATH and no
        version is pinned, install() probes and returns WITHOUT running the unbounded
        ``curl https://claude.ai/install.sh | bash`` (which hangs the whole setup with no
        timeout when that host is unreachable from a sealed sandbox)."""
        agent = ClaudeCode(logs_dir=temp_dir)  # _version is None

        async def _exec(*args, **kwargs):
            return SimpleNamespace(
                return_code=0, stdout="/usr/local/bin/claude", stderr=""
            )

        environment = AsyncMock()
        environment.exec.side_effect = _exec
        await agent.install(environment)

        cmds = [c.kwargs.get("command", "") for c in environment.exec.call_args_list]
        assert any("command -v claude" in c for c in cmds)  # the probe ran
        assert not any(
            "install.sh" in c or "npm install -g @anthropic-ai" in c for c in cmds
        )
        assert not any(
            c.kwargs.get("user") == "root" for c in environment.exec.call_args_list
        )

    @pytest.mark.asyncio
    async def test_claude_code_install_runs_and_bounds_curl_when_absent(self, temp_dir):
        """When claude-code is NOT already present, install() proceeds with the root
        setup and a connect/total-bounded curl (so an unreachable claude.ai fails fast)."""
        agent = ClaudeCode(logs_dir=temp_dir)

        async def _exec(*args, **kwargs):
            cmd = kwargs.get("command", "")
            return SimpleNamespace(
                return_code=(1 if "command -v claude" in cmd else 0),
                stdout="",
                stderr="",
            )

        environment = AsyncMock()
        environment.exec.side_effect = _exec
        await agent.install(environment)

        cmds = [c.kwargs.get("command", "") for c in environment.exec.call_args_list]
        assert any("install.sh" in c for c in cmds)  # installer ran
        assert any("--max-time" in c for c in cmds)  # curl is bounded
        assert any(
            c.kwargs.get("user") == "root" for c in environment.exec.call_args_list
        )
