"""Ralph kickoff in ClaudeCode.run().

The brief is a markdown FILE in the harness's working directory, never a message and
never argv. With `ralph_kickoff`, stdin is stream-json carrying a SINGLE user message --
the kickoff -- whose slash command is recognized because it begins that message; the
kickoff names the workspace brief, which is how the agent gets the task. Without a
kickoff, run() pipes that same workspace file on plain stdin.
"""

import json
from unittest.mock import AsyncMock

import pytest

import harbor.agents.installed.claude_code as cc
from harbor.agents.installed.claude_code import ClaudeCode


def _env():
    env = AsyncMock()
    env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    return env


def _commands(env):
    return [c.kwargs["command"] for c in env.exec.call_args_list]


@pytest.mark.asyncio
async def test_kickoff_sends_a_single_stream_json_message(temp_dir):
    agent = ClaudeCode(logs_dir=temp_dir, ralph_kickoff="/goal follow the brief")
    env = _env()
    await agent.run("brief", env, AsyncMock())
    commands = _commands(env)
    assert len(commands) == 4  # setup, brief write, messages write, claude
    assert f"> {cc.MESSAGES_PATH}" in commands[2]
    assert "--input-format=stream-json" in commands[3]
    assert f"< {cc.MESSAGES_PATH}" in commands[3]


def test_kickoff_message_is_the_goal_alone(temp_dir):
    # The instruction must NOT be a message: concatenated after a leading /goal it
    # would be swallowed as goal text and truncated at 4000 chars.
    agent = ClaudeCode(logs_dir=temp_dir, ralph_kickoff="/goal follow the brief")
    lines = agent._kickoff_messages().splitlines()
    msgs = [json.loads(line) for line in lines]
    assert [m["message"]["content"] for m in msgs] == ["/goal follow the brief"]
    assert all(m["type"] == "user" and m["message"]["role"] == "user" for m in msgs)


@pytest.mark.asyncio
async def test_brief_is_written_to_the_working_directory(temp_dir):
    agent = ClaudeCode(logs_dir=temp_dir, ralph_kickoff="/goal follow the brief")
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    write = _commands(env)[1]
    assert f"> {cc.WORKSPACE_BRIEF_PATH}" in write
    assert cc.WORKSPACE_BRIEF_PATH.startswith(cc.WORKSPACE_DIR + "/")


@pytest.mark.asyncio
async def test_persisted_brief_is_a_copy_of_the_agents_own_file(temp_dir):
    # Persisted by `cp`, not a second render, so the recorded bytes ARE the bytes the
    # agent was given -- the two cannot drift.
    agent = ClaudeCode(logs_dir=temp_dir, ralph_kickoff="/goal follow the brief")
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    write = _commands(env)[1]
    assert f"cp {cc.WORKSPACE_BRIEF_PATH} {cc.BRIEF_PATH}" in write


@pytest.mark.asyncio
async def test_brief_never_rides_argv(temp_dir):
    # The brief names server.py many times; in argv the agent's own `pkill -f
    # server.py` would match the claude process and SIGTERM it mid-run.
    agent = ClaudeCode(logs_dir=temp_dir, ralph_kickoff="/goal follow the brief")
    env = _env()
    await agent.run("build server.py and keep it running", env, AsyncMock())
    assert "build server.py and keep it running" not in _commands(env)[3]


@pytest.mark.asyncio
async def test_no_kickoff_pipes_the_workspace_brief(temp_dir):
    agent = ClaudeCode(logs_dir=temp_dir)
    env = _env()
    await agent.run("brief", env, AsyncMock())
    commands = _commands(env)
    assert len(commands) == 3
    assert f"< {cc.WORKSPACE_BRIEF_PATH}" in commands[2]
    assert "--input-format" not in commands[2]


def test_total_cost_of_one_process_is_its_last_result_event(temp_dir):
    # One result event per query the process ran; total_cost_usd is the process's
    # running total, so its last event is what it spent.
    agent = ClaudeCode(logs_dir=temp_dir)
    (temp_dir / "claude-code.txt").write_text(
        '{"type":"result","total_cost_usd":0.5}\n'
        "noise line\n"
        '{"type":"result","total_cost_usd":1.25}\n'
    )
    assert agent._parse_total_cost_from_stream_json() == 1.25


def test_total_cost_sums_the_processes_of_a_reentered_session(temp_dir):
    # As Claude Code 2.1.251 writes it: the resumed process restarts its total at
    # zero, opening with a zero-turn query for the background task its predecessor
    # left behind.
    agent = ClaudeCode(logs_dir=temp_dir)
    marker = json.dumps({"type": cc.REENTRY_MARKER_TYPE, "reentry": 1})
    (temp_dir / "claude-code.txt").write_text(
        '{"type":"result","total_cost_usd":0.125}\n'
        f"{marker}\n"
        '{"type":"result","total_cost_usd":0,"origin":{"kind":"task-notification"}}\n'
        '{"type":"result","total_cost_usd":2.375}\n'
    )
    assert agent._parse_total_cost_from_stream_json() == pytest.approx(2.5)


def test_total_cost_is_none_without_a_reported_figure(temp_dir):
    agent = ClaudeCode(logs_dir=temp_dir)
    marker = json.dumps({"type": cc.REENTRY_MARKER_TYPE, "reentry": 1})
    (temp_dir / "claude-code.txt").write_text(f'{{"type":"result"}}\n{marker}\n')
    assert agent._parse_total_cost_from_stream_json() is None
