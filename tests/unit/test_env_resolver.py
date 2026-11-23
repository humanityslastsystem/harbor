import json
import os

import pytest

from harbor.utils.env import (
    get_required_host_vars,
    parse_bool_env_value,
    redact_sensitive_mapping,
    resolve_env_vars,
)


class TestParseBoolEnvValue:
    def test_truthy_strings(self):
        for value in ("true", "True", "TRUE", "1", "yes", "Yes", " yes "):
            assert parse_bool_env_value(value, name="TEST_FLAG") is True

    def test_falsy_strings(self):
        for value in ("false", "False", "FALSE", "0", "no", "No", " no "):
            assert parse_bool_env_value(value, name="TEST_FLAG") is False

    def test_bool_values(self):
        assert parse_bool_env_value(True, name="TEST_FLAG") is True
        assert parse_bool_env_value(False, name="TEST_FLAG") is False

    def test_none_uses_default(self):
        assert parse_bool_env_value(None, name="TEST_FLAG", default=False) is False
        assert parse_bool_env_value(None, name="TEST_FLAG", default=True) is True

    def test_none_without_default_raises(self):
        with pytest.raises(ValueError, match="expected bool"):
            parse_bool_env_value(None, name="TEST_FLAG")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="cannot parse"):
            parse_bool_env_value("maybe", name="TEST_FLAG")


class TestResolveEnvVars:
    def test_literal_values_pass_through(self):
        """Test that literal values are passed through unchanged"""
        env_dict = {
            "MODEL_NAME": "claude-3-5-sonnet-20241022",
            "TEMPERATURE": "0.3",
            "MAX_TOKENS": "1024",
        }

        result = resolve_env_vars(env_dict)

        assert result == env_dict

    def test_template_substitution_from_environment(self):
        """Test that ${VAR_NAME} templates are resolved from os.environ"""
        # Set test environment variables
        os.environ["TEST_API_KEY"] = "sk-test-123"
        os.environ["TEST_MODEL"] = "gpt-4"

        try:
            env_dict = {
                "API_KEY": "${TEST_API_KEY}",
                "MODEL": "${TEST_MODEL}",
            }

            result = resolve_env_vars(env_dict)

            assert result == {
                "API_KEY": "sk-test-123",
                "MODEL": "gpt-4",
            }
        finally:
            # Clean up
            del os.environ["TEST_API_KEY"]
            del os.environ["TEST_MODEL"]

    def test_mixed_templates_and_literals(self):
        """Test that templates and literals can be mixed"""
        os.environ["TEST_SECRET"] = "secret-value"

        try:
            env_dict = {
                "SECRET_KEY": "${TEST_SECRET}",
                "MODEL_NAME": "claude-3-5-sonnet-20241022",
                "TEMPERATURE": "0.7",
            }

            result = resolve_env_vars(env_dict)

            assert result == {
                "SECRET_KEY": "secret-value",
                "MODEL_NAME": "claude-3-5-sonnet-20241022",
                "TEMPERATURE": "0.7",
            }
        finally:
            del os.environ["TEST_SECRET"]

    def test_missing_environment_variable_raises_error(self):
        """Test that missing environment variables raise ValueError"""
        env_dict = {
            "API_KEY": "${MISSING_ENV_VAR}",
        }

        with pytest.raises(ValueError) as exc_info:
            resolve_env_vars(env_dict)

        assert "MISSING_ENV_VAR" in str(exc_info.value)
        assert "not found in host environment" in str(exc_info.value)

    def test_empty_dict_returns_empty_dict(self):
        """Test that empty input returns empty output"""
        result = resolve_env_vars({})
        assert result == {}

    def test_partial_template_not_substituted(self):
        """Test that partial templates like $VAR or {VAR} are treated as literals"""
        env_dict = {
            "PARTIAL1": "$VAR",
            "PARTIAL2": "{VAR}",
            "PARTIAL3": "${",
            "PARTIAL4": "prefix_${VAR}_suffix",
        }

        result = resolve_env_vars(env_dict)

        # These should all pass through as literals since they don't match the pattern
        assert result == env_dict

    def test_special_characters_in_values(self):
        """Test that special characters in environment values are preserved"""
        os.environ["TEST_SPECIAL"] = "value=with=equals&symbols!"

        try:
            env_dict = {
                "SPECIAL": "${TEST_SPECIAL}",
            }

            result = resolve_env_vars(env_dict)

            assert result == {
                "SPECIAL": "value=with=equals&symbols!",
            }
        finally:
            del os.environ["TEST_SPECIAL"]

    def test_default_value_when_var_unset(self):
        """Test that ${VAR:-default} uses the default when VAR is unset"""
        # Ensure the variable is not set
        os.environ.pop("UNSET_VAR_FOR_TEST", None)

        env_dict = {
            "KEY": "${UNSET_VAR_FOR_TEST:-fallback}",
        }

        result = resolve_env_vars(env_dict)
        assert result == {"KEY": "fallback"}

    def test_default_value_when_var_set(self):
        """Test that ${VAR:-default} uses the env value when VAR is set"""
        os.environ["TEST_DEFAULT_SET"] = "real-value"

        try:
            env_dict = {
                "KEY": "${TEST_DEFAULT_SET:-fallback}",
            }

            result = resolve_env_vars(env_dict)
            assert result == {"KEY": "real-value"}
        finally:
            del os.environ["TEST_DEFAULT_SET"]

    def test_empty_default_value(self):
        """Test that ${VAR:-} uses empty string as default"""
        os.environ.pop("UNSET_VAR_FOR_TEST", None)

        env_dict = {
            "KEY": "${UNSET_VAR_FOR_TEST:-}",
        }

        result = resolve_env_vars(env_dict)
        assert result == {"KEY": ""}

    def test_whitespace_in_template(self):
        """Test that whitespace in templates is handled correctly"""
        os.environ["TEST_VAR"] = "test-value"

        try:
            # Template with no spaces should work
            env_dict = {
                "KEY": "${TEST_VAR}",
            }

            result = resolve_env_vars(env_dict)
            assert result["KEY"] == "test-value"
        finally:
            del os.environ["TEST_VAR"]


class TestGetRequiredHostVars:
    def test_extracts_template_vars(self):
        env_dict = {"A": "${X}", "B": "literal", "C": "${Y:-default}"}
        result = get_required_host_vars(env_dict)
        assert ("X", None) in result
        assert ("Y", "default") in result
        assert len(result) == 2

    def test_empty_dict(self):
        assert get_required_host_vars({}) == []

    def test_all_literals(self):
        env_dict = {"A": "value1", "B": "value2"}
        assert get_required_host_vars(env_dict) == []

    def test_empty_default(self):
        env_dict = {"A": "${VAR:-}"}
        result = get_required_host_vars(env_dict)
        assert result == [("VAR", "")]


class TestRedactSensitiveMapping:
    def test_sensitive_values_redacted_and_others_kept(self):
        result = redact_sensitive_mapping(
            {"api_key": "sk-live-9f3a1c7e2b8d4a60", "max_retries": 3}
        )
        assert result == {"api_key": "sk-l****a60", "max_retries": 3}

    def test_nested_dict_is_walked(self):
        result = redact_sensitive_mapping(
            {"extra_headers": {"Authorization": "Bearer cb-tok-51de9a44c07b2f83"}}
        )
        assert result == {"extra_headers": {"Authorization": "Bear****f83"}}

    def test_nested_list_is_walked(self):
        result = redact_sensitive_mapping(
            {"providers": [{"name": "gateway", "token": "cb-tok-51de9a44c07b2f83"}]}
        )
        assert result == {"providers": [{"name": "gateway", "token": "cb-t****f83"}]}

    def test_sensitive_key_holding_a_structure_is_redacted_whole(self):
        result = redact_sensitive_mapping(
            {"vertex_credentials": {"private_key": "-----BEGIN PRIVATE KEY-----"}}
        )
        assert result == {"vertex_credentials": "****"}

    def test_non_string_keys_do_not_raise(self):
        # kwargs bags are arbitrary Python, and the dump must not fail on a payload it
        # would otherwise have recorded.
        assert redact_sensitive_mapping({"ids": {1: "one"}}) == {"ids": {1: "one"}}

    def test_input_is_not_mutated(self):
        original = {"api_key": "sk-live-9f3a1c7e2b8d4a60"}
        redact_sensitive_mapping(original)
        assert original == {"api_key": "sk-live-9f3a1c7e2b8d4a60"}

    def test_numbers_under_a_sensitive_key_stay_readable(self):
        # The pattern that flags a credential also flags counters, and a number is
        # never a credential.
        result = redact_sensitive_mapping({"max_tokens": 4096, "total_tokens": 30})
        assert result == {"max_tokens": 4096, "total_tokens": 30}

    def test_tuples_keep_their_container_type(self):
        result = redact_sensitive_mapping({"stop": ("</s>", "\n\n")})
        assert result == {"stop": ("</s>", "\n\n")}

    def test_tuple_nested_in_a_dict_keeps_its_container_type(self):
        result = redact_sensitive_mapping({"extra_body": {"stop": ("</s>",)}})
        assert result == {"extra_body": {"stop": ("</s>",)}}

    def test_a_cycle_is_cut_rather_than_followed(self):
        bag: dict = {"api_key": "sk-live-9f3a1c7e2b8d4a60", "nested": {}}
        bag["nested"]["self"] = bag

        result = redact_sensitive_mapping(bag)

        assert result["api_key"] == "sk-l****a60"
        # The copy terminates, so it can be serialized.
        assert json.dumps(result)

    def test_deep_nesting_does_not_raise(self):
        bag: dict = {"leaf": "value"}
        for _ in range(5_000):
            bag = {"nested": bag}

        assert redact_sensitive_mapping(bag)
