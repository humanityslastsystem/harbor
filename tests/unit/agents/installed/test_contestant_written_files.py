"""The readers of files an agent's own commands can write: Claude Code's stream tee and
session transcripts, and codex's session files.

No line the agent wrote may make a reader raise. A line nested past the JSON decoder's
depth raises RecursionError, one longer than the reader's bound is not parsed, and either
is skipped with a warning naming the file, while the other lines are read.
"""

import json
import logging
from pathlib import Path

import pytest

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext

DEEP = "[" * 50_000
SESSION = "0b6f7a52-3c1e-4d2a-9f1e-2a7c9d4e5f60"


def _claude(logs_dir: Path) -> ClaudeCode:
    return ClaudeCode(logs_dir=logs_dir, inherit_process_env=False)


def _codex(logs_dir: Path) -> Codex:
    return Codex(logs_dir=logs_dir, model_name="openai/gpt-5")


def _assistant(i: int) -> dict:
    return {
        "type": "assistant",
        "timestamp": f"2026-01-01T00:00:{i:02d}Z",
        "sessionId": SESSION,
        "version": "2.1.50",
        "message": {
            "id": f"m{i}",
            "model": "claude-opus-4-6",
            "role": "assistant",
            "content": [{"type": "text", "text": f"step {i}"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    }


@pytest.mark.unit
def test_json_objects_skip_a_deep_a_long_and_a_non_object_line(tmp_path, caplog):
    from harbor.utils import jsonl

    path = tmp_path / "t.jsonl"
    path.write_bytes(
        b'{"a": 1}\n'
        + DEEP.encode()
        + b"\n"
        + b'{"b": "'
        + b"x" * 5000
        + b'"}\n'
        + b"[1, 2]\n"
        + b'{"c": "\xff"}\n'
        + b'{"d": 4}'
    )
    with caplog.at_level("WARNING"):
        objs = list(
            jsonl.json_objects(
                path, max_line_bytes=4096, logger=logging.getLogger("harbor.test")
            )
        )
    assert objs == [{"a": 1}, {"c": "�"}, {"d": 4}]
    assert any(
        str(path) in r.getMessage() and "3 lines" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.unit
def test_a_line_at_the_bound_is_read_and_one_past_it_is_not(tmp_path):
    from harbor.utils import jsonl

    line = json.dumps({"x": "y" * 100})
    path = tmp_path / "t.jsonl"
    path.write_text(line + "\n" + line + "\n")
    log = logging.getLogger("harbor.test")
    assert (
        len(list(jsonl.json_objects(path, max_line_bytes=len(line), logger=log))) == 2
    )
    assert (
        list(jsonl.json_objects(path, max_line_bytes=len(line) - 1, logger=log)) == []
    )


@pytest.mark.unit
def test_claudes_cost_is_read_past_a_deep_line_in_its_stream(tmp_path):
    (tmp_path / "claude-code.txt").write_text(
        json.dumps({"type": "result", "total_cost_usd": 0.5})
        + "\n"
        + '{"type": "result", "total_cost_usd": '
        + DEEP
        + "\n"
    )
    assert _claude(tmp_path)._parse_total_cost_from_stream_json() == 0.5


@pytest.mark.unit
def test_claudes_trajectory_is_read_past_a_deep_line_and_written_to_its_directory(
    tmp_path, caplog
):
    """A deep line is skipped, and a turn whose timestamp is no string is dropped without
    losing the others. The trajectory is written to ``trajectory_dir`` when one is set,
    never among the agent's own files."""
    logs = tmp_path / "logs"
    session = logs / "sessions" / "projects" / "-workspace"
    session.mkdir(parents=True)
    path = session / f"{SESSION}.jsonl"
    path.write_text(
        json.dumps(_assistant(0))
        + "\n"
        + DEEP
        + "\n"
        + json.dumps({**_assistant(1), "timestamp": 7})
        + "\n"
        + json.dumps(_assistant(2))
        + "\n"
    )
    agent = _claude(logs)
    agent.trajectory_dir = tmp_path / "harness"
    agent.trajectory_dir.mkdir()
    with caplog.at_level("WARNING"):
        agent.populate_context_post_run(AgentContext())
    trajectory = json.loads((tmp_path / "harness" / "trajectory.json").read_text())
    assert [step["message"] for step in trajectory["steps"]] == ["step 0", "step 2"]
    assert not (logs / "trajectory.json").exists()
    assert any(str(path) in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_codex_reads_its_session_past_a_deep_line(tmp_path, caplog):
    logs = tmp_path / "logs"
    day = logs / "sessions" / "2026" / "01" / "01"
    day.mkdir(parents=True)
    meta = {
        "timestamp": "2026-01-01T00:00:00Z",
        "type": "session_meta",
        "payload": {"id": SESSION, "source": "exec", "thread_source": "user"},
    }
    message = {
        "timestamp": "2026-01-01T00:00:01Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        },
    }
    call = {
        "timestamp": "2026-01-01T00:00:02Z",
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": "c1",
            "name": "shell",
            "arguments": DEEP,
        },
    }
    output = {
        "timestamp": "2026-01-01T00:00:03Z",
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": "c1", "output": DEEP},
    }
    path = day / "rollout-x.jsonl"
    path.write_text(
        DEEP
        + "\n"
        + "\n".join(json.dumps(o) for o in (meta, message, call, output))
        + "\n"
    )
    other = day / "rollout-y.jsonl"
    other.write_text(DEEP + "\n")
    assert Codex._is_subagent_session(path) is False
    agent = _codex(logs)
    agent.trajectory_dir = tmp_path / "harness"
    agent.trajectory_dir.mkdir()
    with caplog.at_level("WARNING"):
        agent.populate_context_post_run(AgentContext())
    trajectory = json.loads((tmp_path / "harness" / "trajectory.json").read_text())
    assert trajectory["session_id"] == SESSION and len(trajectory["steps"]) >= 2
    assert not (logs / "trajectory.json").exists()
    assert any(str(path) in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_codex_sessions_the_agent_split_into_directories_are_no_trajectory(
    tmp_path, caplog
):
    """Two session directories at the same depth: no trajectory, a warning, no error."""
    logs = tmp_path / "logs"
    (logs / "sessions" / "2026" / "01" / "01").mkdir(parents=True)
    (logs / "sessions" / "2026" / "01" / "02").mkdir(parents=True)
    with caplog.at_level("WARNING"):
        _codex(logs).populate_context_post_run(AgentContext())
    assert any("Codex session directory" in r.getMessage() for r in caplog.records)
