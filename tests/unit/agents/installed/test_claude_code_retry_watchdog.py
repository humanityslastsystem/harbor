"""The API retry watchdog is enabled for unattended eval-harness runs.

`CLAUDE_CODE_RETRY_WATCHDOG=1` makes Claude Code (>= 2.1.186) wait out a transient
API capacity/rate-limit outage (429/529) -- retrying indefinitely with up to a
5-minute backoff -- instead of giving up after the (15-capped) retry count and
exiting non-zero mid-build. Every Harbor run is an unattended session, so the flag
is set by default; a host may override it to "0" for debugging.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from harbor.agents.installed.claude_code import ClaudeCode


async def _run_env(logs_dir, environ: dict[str, str]) -> dict[str, str]:
    """Run the agent under a patched environment; return the env dict passed to
    the main `claude` exec (the last exec call in run())."""
    mock_env = AsyncMock()
    mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    with patch.dict(os.environ, environ, clear=True):
        agent = ClaudeCode(logs_dir=logs_dir)
        await agent.run("do something", mock_env, AsyncMock())
    return mock_env.exec.call_args_list[-1].kwargs["env"]


class TestRetryWatchdog:
    @pytest.mark.asyncio
    async def test_enabled_by_default(self, tmp_path):
        env = await _run_env(tmp_path, {})
        assert env["CLAUDE_CODE_RETRY_WATCHDOG"] == "1"

    @pytest.mark.asyncio
    async def test_host_override_respected(self, tmp_path):
        env = await _run_env(tmp_path, {"CLAUDE_CODE_RETRY_WATCHDOG": "0"})
        assert env["CLAUDE_CODE_RETRY_WATCHDOG"] == "0"
