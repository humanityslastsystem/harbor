"""How an installed agent cuts a failed command's output for its error and its log."""

import asyncio

import pytest

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.environments.base import ExecResult


class _Failing:
    """An environment whose every command fails with *stdout*."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        return ExecResult(stdout=self.stdout, stderr=None, return_code=1)


def test_an_output_is_cut_at_its_first_characters_by_default(tmp_path):
    agent = ClaudeCode(logs_dir=tmp_path)
    assert agent._truncate_output("x" * 1200) == "x" * 1000 + " ... [truncated]"
    assert agent._truncate_output("short") == "short"
    assert agent._truncate_output(None) == "None"


@pytest.mark.parametrize(
    "make",
    [
        lambda tmp_path, cut: ClaudeCode(logs_dir=tmp_path, cut_output=cut),
        lambda tmp_path, cut: Codex(
            logs_dir=tmp_path, model_name="openai/gpt-5.5", cut_output=cut
        ),
    ],
    ids=["claude-code", "codex"],
)
def test_the_callers_cut_decides_where_a_long_output_is_cut(tmp_path, make):
    """The caller's cut is given the whole text and the most characters to keep, and its head is
    what the error and the log quote."""
    seen = []

    def cut(text: str, max_len: int) -> str:
        seen.append((len(text), max_len))
        return text[: max_len - 3]

    agent = make(tmp_path, cut)
    assert agent._truncate_output("x" * 1200) == "x" * 997 + " ... [truncated]"
    assert agent._truncate_output("short") == "short" and seen == [(1200, 1000)]
    with pytest.raises(NonZeroAgentExitCodeError) as caught:
        asyncio.run(agent.exec_as_agent(_Failing("y" * 1500), command="false"))
    assert f"stdout: {'y' * 997} ... [truncated]" in str(caught.value)
