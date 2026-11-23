"""Credentials must not reach the per-episode debug log.

The constructor's `llm_kwargs` and the per-call kwargs are splatted into every
completion call, so `api_key` and an `Authorization` header belong there. litellm
reports both back in the call details that harbor writes to `episode-<n>/debug.json`,
a file that is copied into run records and shared.
"""

import hashlib
import json

import pytest
from litellm import ModelResponse

from harbor.llms.lite_llm import LiteLLM

API_KEY = "sk-live-9f3a1c7e2b8d4a60"
BEARER = "cb-tok-51de9a44c07b2f83"


def _post_api_call_details(completion_kwargs: dict) -> dict:
    """The call details litellm hands ``logger_fn`` on the event harbor persists.

    litellm reports the credential three times: at the top level, again under
    ``litellm_params``, and in the request headers under ``additional_args``. The
    numbers alongside them are what the log is read for, and the unset fields and the
    callback it was handed are the rest of what a real event carries.
    """
    api_key = completion_kwargs.get("api_key")
    return {
        "log_event_type": "post_api_call",
        "model": completion_kwargs["model"],
        "messages": completion_kwargs["messages"],
        "api_key": api_key,
        "optional_params": {"max_tokens": 4096, "temperature": 0.0},
        "litellm_params": {
            "api_key": api_key,
            "api_base": completion_kwargs.get("api_base"),
            "preset_cache_key": None,
            "logger_fn": completion_kwargs["logger_fn"],
        },
        "additional_args": {
            "headers": {"Authorization": f"Bearer {api_key}"},
            "complete_input_dict": {
                key: value
                for key, value in completion_kwargs.items()
                if key != "logger_fn"
            },
        },
        "standard_logging_object": {
            "total_tokens": 30,
            "prompt_tokens": 10,
            "completion_tokens": 20,
        },
    }


def _fake_acompletion(recorded_details: list[dict]):
    """An ``acompletion`` that logs the call the way litellm does, then answers."""

    async def acompletion(**completion_kwargs):
        details = _post_api_call_details(completion_kwargs)
        recorded_details.append(details)
        completion_kwargs["logger_fn"](details)
        return ModelResponse(
            model=completion_kwargs["model"],
            choices=[
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    return acompletion


async def _write_debug_log(temp_dir, monkeypatch, llm: LiteLLM, **call_kwargs):
    """Run one call that logs to a debug file, return the file text and the raw details."""
    recorded_details: list[dict] = []
    monkeypatch.setattr("litellm.acompletion", _fake_acompletion(recorded_details))

    logging_path = temp_dir / "debug.json"
    await llm.call(prompt="hello", logging_path=logging_path, **call_kwargs)

    return logging_path.read_text(), recorded_details[0]


@pytest.mark.asyncio
async def test_per_call_api_key_does_not_reach_the_debug_log(temp_dir, monkeypatch):
    llm = LiteLLM(model_name="openai/fake-model")

    text, details = await _write_debug_log(temp_dir, monkeypatch, llm, api_key=API_KEY)

    assert API_KEY in str(details), "the credential reaches the logger unredacted"
    assert API_KEY not in text


@pytest.mark.asyncio
async def test_constructor_auth_header_does_not_reach_the_debug_log(
    temp_dir, monkeypatch
):
    llm = LiteLLM(
        model_name="openai/fake-model",
        llm_kwargs={"extra_headers": {"Authorization": f"Bearer {BEARER}"}},
    )

    text, details = await _write_debug_log(temp_dir, monkeypatch, llm)

    assert BEARER in str(details), "the credential reaches the logger unredacted"
    assert BEARER not in text


@pytest.mark.asyncio
async def test_top_level_key_is_recorded_as_a_digest(temp_dir, monkeypatch):
    llm = LiteLLM(model_name="openai/fake-model")

    text, _ = await _write_debug_log(temp_dir, monkeypatch, llm, api_key=API_KEY)
    written = json.loads(text)

    assert written["api_key_sha256"] == hashlib.sha256(API_KEY.encode()).hexdigest()
    assert "api_key" not in written


@pytest.mark.asyncio
async def test_credentials_below_the_top_level_are_redacted(temp_dir, monkeypatch):
    llm = LiteLLM(model_name="openai/fake-model")

    text, _ = await _write_debug_log(temp_dir, monkeypatch, llm, api_key=API_KEY)
    written = json.loads(text)

    assert written["litellm_params"]["api_key"] == "sk-l****a60"
    assert written["additional_args"]["headers"]["Authorization"] == "Bear****a60"
    assert written["additional_args"]["complete_input_dict"]["api_key"] == "sk-l****a60"


@pytest.mark.asyncio
async def test_counts_stay_readable(temp_dir, monkeypatch):
    """The counts the log is kept for survive the redaction.

    The pattern that flags a credential also matches ``max_tokens`` and the usage
    counts, which are numbers and so cannot be credentials.
    """
    llm = LiteLLM(model_name="openai/fake-model")

    text, _ = await _write_debug_log(temp_dir, monkeypatch, llm, api_key=API_KEY)
    written = json.loads(text)

    assert written["optional_params"]["max_tokens"] == 4096
    assert written["standard_logging_object"] == {
        "total_tokens": 30,
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }


@pytest.mark.asyncio
async def test_an_unset_field_is_recorded_as_null(temp_dir, monkeypatch):
    """An unset value stays null rather than reading as a redacted credential.

    litellm reports dozens of unset fields under key-like names, and a stringified
    ``None`` under one of them comes back as ``****``.
    """
    llm = LiteLLM(model_name="openai/fake-model")

    text, _ = await _write_debug_log(temp_dir, monkeypatch, llm, api_key=API_KEY)
    written = json.loads(text)

    assert written["litellm_params"]["preset_cache_key"] is None


@pytest.mark.asyncio
async def test_a_callback_is_not_recorded(temp_dir, monkeypatch):
    llm = LiteLLM(model_name="openai/fake-model")

    text, _ = await _write_debug_log(temp_dir, monkeypatch, llm, api_key=API_KEY)
    written = json.loads(text)

    assert "logger_fn" not in written["litellm_params"]
