"""Caller-supplied guidance on the completion confirmation re-prompt.

The stock re-prompt tells the model that confirming triggers grading. A run whose
budget must be spent -- where stopping early is the failure -- needs different text
there, and needs it recomputed at every confirmation because what it says (time left)
changes while the run proceeds.
"""

from types import SimpleNamespace

import pytest

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2
from harbor.llms.base import LLMResponse
from harbor.models.agent.context import AgentContext

STOCK_JSON = (
    "Current terminal state:\nroot@host:/app# \n\n"
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)

STOCK_XML = (
    "Current terminal state:\nroot@host:/app# \n\n"
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    "make any further corrections. If so, include "
    "<task_complete>true</task_complete> again."
)

TERMINAL_OUTPUT = "root@host:/app# "


def _agent(temp_dir, monkeypatch, **kwargs) -> Terminus2:
    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **_: object())
    return Terminus2(logs_dir=temp_dir, model_name="fake/model", **kwargs)


@pytest.mark.parametrize(
    "parser_name,stock", [("json", STOCK_JSON), ("xml", STOCK_XML)]
)
def test_default_confirmation_is_byte_identical(
    temp_dir, monkeypatch, parser_name, stock
):
    agent = _agent(temp_dir, monkeypatch, parser_name=parser_name)
    assert agent._get_completion_confirmation_message(TERMINAL_OUTPUT) == stock


@pytest.mark.parametrize(
    "parser_name,stock", [("json", STOCK_JSON), ("xml", STOCK_XML)]
)
def test_guidance_follows_the_stock_prompt(temp_dir, monkeypatch, parser_name, stock):
    agent = _agent(
        temp_dir,
        monkeypatch,
        parser_name=parser_name,
        completion_guidance=lambda: "11h30m of your budget is left; keep working.",
    )
    message = agent._get_completion_confirmation_message(TERMINAL_OUTPUT)
    assert message == f"{stock}\n\n11h30m of your budget is left; keep working."


def test_replace_mode_drops_the_stock_prompt_but_keeps_the_terminal_state(
    temp_dir, monkeypatch
):
    agent = _agent(
        temp_dir,
        monkeypatch,
        completion_guidance=lambda: "Do not stop yet.",
        completion_guidance_mode="replace",
    )
    message = agent._get_completion_confirmation_message(TERMINAL_OUTPUT)
    assert message == f"Current terminal state:\n{TERMINAL_OUTPUT}\n\nDo not stop yet."
    assert "Are you sure" not in message


@pytest.mark.parametrize("guidance", [None, ""])
@pytest.mark.parametrize("mode", ["append", "replace"])
def test_empty_guidance_leaves_the_stock_prompt_alone(
    temp_dir, monkeypatch, guidance, mode
):
    # Replace mode with nothing to replace must not strip the prompt: a provider that
    # only speaks up near the deadline returns nothing for most of the run.
    agent = _agent(
        temp_dir,
        monkeypatch,
        completion_guidance=lambda: guidance,
        completion_guidance_mode=mode,
    )
    assert agent._get_completion_confirmation_message(TERMINAL_OUTPUT) == STOCK_JSON


def test_unknown_mode_is_rejected_at_construction(temp_dir, monkeypatch):
    with pytest.raises(ValueError, match="completion_guidance_mode"):
        _agent(temp_dir, monkeypatch, completion_guidance_mode="prepend")


async def _drive_loop(agent, completions: list[bool]) -> list[str]:
    """Run the real episode loop over a scripted sequence of task_complete signals.

    Returns the prompts the loop fed to the LLM, so a test can assert on what the
    confirmation turn actually said.
    """
    prompts: list[str] = []
    signals = iter(completions)

    async def handle_llm_interaction(chat, prompt, logging_paths, *args, **kwargs):
        prompts.append(prompt)
        return (
            [Command(keystrokes="ls\n", duration_sec=0.1)],
            next(signals),
            "",
            "",
            "",
            LLMResponse(content="response"),
        )

    async def execute_commands(commands, session):
        return False, TERMINAL_OUTPUT

    agent._handle_llm_interaction = handle_llm_interaction
    agent._execute_commands = execute_commands
    agent._context = AgentContext()
    agent._session = SimpleNamespace(is_session_alive=_alive)

    chat = SimpleNamespace(
        total_input_tokens=0,
        total_output_tokens=0,
        total_cache_tokens=0,
        total_cost=0.0,
    )
    await agent._run_agent_loop("initial prompt", chat)
    return prompts


async def _alive() -> bool:
    return True


@pytest.mark.asyncio
async def test_loop_asks_for_guidance_afresh_at_each_confirmation(
    temp_dir, monkeypatch
):
    remaining = iter(["11h30m", "6h15m"])
    agent = _agent(
        temp_dir,
        monkeypatch,
        completion_guidance=lambda: f"{next(remaining)} of your budget is left.",
    )

    # Complete, abandon the completion, complete again, then confirm. The two
    # confirmations are separated by an ordinary episode, as they are in a real run.
    prompts = await _drive_loop(agent, [True, False, True, True])

    confirmations = [p for p in prompts if "Are you sure" in p]
    assert len(confirmations) == 2
    assert "11h30m of your budget is left." in confirmations[0]
    assert "6h15m of your budget is left." in confirmations[1]
