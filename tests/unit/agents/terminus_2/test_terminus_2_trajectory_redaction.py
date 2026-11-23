"""Credentials passed in `llm_kwargs` must not reach the persisted trajectory.

`llm_kwargs` is forwarded to the LLM constructor and splatted into every completion
call, so `api_key` and an `Authorization` header belong there, and Terminus 2 records
the bag in `trajectory.json` -- a file that is copied into run records and shared.
"""

import json
from datetime import datetime, timezone

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step

API_KEY = "sk-live-9f3a1c7e2b8d4a60"
BEARER = "cb-tok-51de9a44c07b2f83"


def _dump(temp_dir, monkeypatch, llm_kwargs: dict) -> str:
    """Dump a trajectory for an agent configured with `llm_kwargs`, return the file text."""
    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **_: object())
    agent = Terminus2(logs_dir=temp_dir, model_name="fake/model", llm_kwargs=llm_kwargs)
    agent._context = AgentContext()
    agent._trajectory_steps = [
        Step(
            step_id=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="agent",
            message="response",
        )
    ]
    agent._dump_trajectory()
    return (temp_dir / "trajectory.json").read_text()


def test_api_key_does_not_reach_the_trajectory_file(temp_dir, monkeypatch):
    text = _dump(temp_dir, monkeypatch, {"api_key": API_KEY, "max_retries": 3})

    assert API_KEY not in text
    # The bag itself is still recorded, so the assertion above is about redaction rather
    # than about llm_kwargs having been dropped.
    assert "api_key" in text
    assert "max_retries" in text


def test_nested_credentials_do_not_reach_the_trajectory_file(temp_dir, monkeypatch):
    text = _dump(
        temp_dir,
        monkeypatch,
        {
            "extra_headers": {"Authorization": f"Bearer {BEARER}"},
            "extra_body": {"cache_control": "ephemeral"},
        },
    )

    assert BEARER not in text
    assert "ephemeral" in text, "values worth debugging with stay readable"


def test_redacted_value_keeps_enough_to_identify_the_credential(temp_dir, monkeypatch):
    text = _dump(temp_dir, monkeypatch, {"api_key": API_KEY})

    extra = json.loads(text)["agent"]["extra"]
    assert extra["llm_kwargs"]["api_key"] == "sk-l****a60"


def test_a_self_referential_bag_still_dumps(temp_dir, monkeypatch):
    # The bag is recorded as the caller passed it, so redacting it must not fail on a
    # payload the agent would otherwise have written.
    llm_kwargs: dict = {"api_key": API_KEY, "client": {}}
    llm_kwargs["client"]["owner"] = llm_kwargs

    text = _dump(temp_dir, monkeypatch, llm_kwargs)

    assert API_KEY not in text
