"""Selecting the root Codex session file when a run spawned subagents.

A run that spawns subagents writes one JSONL per thread into the same directory. Picking
the wrong one reports that thread's steps and token usage as the whole run's.
"""

import json
import logging
from pathlib import Path

import pytest

from harbor.agents.installed.codex import Codex

ROOT_ID = "aaaaaaaa-0000-7000-8000-000000000001"


def _agent() -> Codex:
    """A Codex instance with just the logger the selection path needs."""
    agent = Codex.__new__(Codex)
    agent.logger = logging.getLogger(__name__)
    return agent


def _write_session(
    directory: Path, name: str, session_id: str, lines: int, parent: str | None = None
) -> Path:
    """Write a session file whose first record is its ``session_meta``."""
    payload: dict = {
        "id": session_id,
        "session_id": parent or session_id,
        "originator": "codex_exec",
    }
    if parent is None:
        payload["source"] = "exec"
        payload["thread_source"] = "user"
    else:
        payload["forked_from_id"] = parent
        payload["parent_thread_id"] = parent
        payload["thread_source"] = "subagent"

    records = [
        {"timestamp": "2026-01-01T00:00:00", "type": "session_meta", "payload": payload}
    ]
    records += [
        {"type": "response_item", "payload": {"type": "message"}} for _ in range(lines)
    ]
    path = directory / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


@pytest.mark.unit
def test_selects_root_when_subagents_sort_first_and_last(tmp_path: Path) -> None:
    """The root is neither the first nor the last file by name, and is not the smallest."""
    root = _write_session(
        tmp_path, "rollout-2026-01-01T00-00-00-root.jsonl", ROOT_ID, lines=500
    )
    _write_session(
        tmp_path, "rollout-2026-01-01T00-10-00-forkA.jsonl", "fork-a", 5, parent=ROOT_ID
    )
    _write_session(
        tmp_path,
        "rollout-2026-01-01T00-20-00-forkB.jsonl",
        "fork-b",
        20,
        parent=ROOT_ID,
    )

    # glob() returns directory order, so the root is not reliably first; hand the list
    # over with a subagent leading.
    files = sorted(tmp_path.glob("*.jsonl"), reverse=True)
    agent = _agent()
    assert agent._select_session_file(files) == root
    # selecting by directory order or by name picks a subagent instead
    assert files[0] != root
    assert max(files) != root


@pytest.mark.unit
def test_single_session_is_returned_unchanged(tmp_path: Path) -> None:
    only = _write_session(tmp_path, "rollout-solo.jsonl", ROOT_ID, lines=3)
    agent = _agent()
    assert agent._select_session_file([only]) == only


@pytest.mark.unit
def test_subagent_detection(tmp_path: Path) -> None:
    root = _write_session(tmp_path, "root.jsonl", ROOT_ID, lines=1)
    fork = _write_session(tmp_path, "fork.jsonl", "fork-a", 1, parent=ROOT_ID)
    assert Codex._is_subagent_session(root) is False
    assert Codex._is_subagent_session(fork) is True


@pytest.mark.unit
def test_ambiguous_falls_back_to_largest_and_warns(tmp_path: Path, caplog) -> None:
    """No identifiable root: keep going on the largest file, but say so."""
    small = _write_session(tmp_path, "a.jsonl", "a", lines=2, parent=ROOT_ID)
    big = _write_session(tmp_path, "b.jsonl", "b", lines=400, parent=ROOT_ID)

    agent = _agent()
    with caplog.at_level("WARNING"):
        assert agent._select_session_file([small, big]) == big
    assert "root Codex session" in caplog.text
