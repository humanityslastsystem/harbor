import os
import re
from typing import Any

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_SENSITIVE_KEY_RE = re.compile(
    r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE
)
_TRUE_BOOL_VALUES = frozenset({"true", "1", "yes"})
_FALSE_BOOL_VALUES = frozenset({"false", "0", "no"})

# A kwargs bag is arbitrary Python and may hold a reference cycle or a few thousand
# levels of nesting, either of which exhausts the stack while being redacted. Bounding
# the depth covers both, where tracking visited ids would only catch the cycle. Past
# the bound the value is replaced rather than kept, because an unwalked value can
# carry the credential the redaction exists to hide.
_MAX_REDACTION_DEPTH = 32
_TOO_DEEP = f"<nested more than {_MAX_REDACTION_DEPTH} levels deep>"


def parse_bool_env_value(
    value: str | bool | None,
    *,
    name: str = "value",
    default: bool | None = None,
) -> bool:
    """Parse a string environment-style boolean value."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Invalid value for '{name}': expected bool, got None")

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_VALUES:
            return True
        if normalized in _FALSE_BOOL_VALUES:
            return False
        raise ValueError(
            f"Invalid value for '{name}': cannot parse '{value}' as bool "
            f"(expected true/false/1/0/yes/no)"
        )

    raise ValueError(
        f"Invalid value for '{name}': expected bool, got {value.__class__.__name__}"
    )


def is_env_template(value: str) -> bool:
    """Return True if ``value`` is an env var template like ``${VAR}`` or ``${VAR:-default}``."""
    return bool(_TEMPLATE_PATTERN.fullmatch(value))


def is_sensitive_env_key(key: str) -> bool:
    return bool(_SENSITIVE_KEY_RE.search(key))


def redact_sensitive_value(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-3:]


def redact_sensitive_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Copy ``mapping`` with the values under sensitive-looking keys redacted.

    For kwargs bags that get persisted, where a credential sits beside values worth
    keeping readable. The keys survive, so a reader still sees what was configured,
    and lists and tuples keep their container type.

    Nested dicts are walked: a bag holding ``extra_headers`` keeps its
    ``Authorization`` one level down. Numbers under a sensitive key are kept as they
    are, since a credential is a string or a structure of strings, and the pattern
    that flags a credential also flags counters such as ``max_tokens``. Anything
    nested past ``_MAX_REDACTION_DEPTH``, a cycle included, is replaced by a marker.
    """
    return _redact_mapping(mapping, _MAX_REDACTION_DEPTH)


def _redact_mapping(mapping: dict[str, Any], depth: int) -> dict[str, Any]:
    return {key: _redact_under_key(key, value, depth) for key, value in mapping.items()}


def _redact_under_key(key: Any, value: Any, depth: int) -> Any:
    if isinstance(key, str) and is_sensitive_env_key(key):
        if isinstance(value, str):
            return redact_sensitive_value(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        # A credentials dict, or an object whose repr could carry the secret, is
        # redacted whole rather than walked.
        return "****"
    return _redact_value(value, depth)


def _redact_value(value: Any, depth: int) -> Any:
    """Walk ``value`` for nested dicts; only a dict key can mark something sensitive."""
    if depth <= 0:
        return _TOO_DEEP
    if isinstance(value, dict):
        return _redact_mapping(value, depth - 1)
    if isinstance(value, (list, tuple)):
        walked = [_redact_value(item, depth - 1) for item in value]
        return tuple(walked) if isinstance(value, tuple) else walked
    return value


def templatize_sensitive_env(env: dict[str, str]) -> dict[str, str]:
    """Serialize sensitive env values for safe persistence and resume.

    - Value already a template: kept as-is.
    - Non-sensitive key: literal.
    - Sensitive key whose literal matches ``os.environ[key]``: ``${KEY}``
      template, so resume can pull from the host env.
    - Sensitive key otherwise: redacted; resume needs the user to re-provide it.
    """
    out: dict[str, str] = {}
    for key, value in env.items():
        if is_env_template(value) or not is_sensitive_env_key(key):
            out[key] = value
        elif os.environ.get(key) == value:
            out[key] = f"${{{key}}}"
        else:
            out[key] = redact_sensitive_value(value)
    return out


def sanitize_env_assignment(value: str) -> str:
    if "=" not in value:
        return value

    key, raw_value = value.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not is_sensitive_env_key(key):
        return f"{key}={raw_value}"
    if is_env_template(raw_value):
        return f"{key}={raw_value}"
    if os.environ.get(key) == raw_value:
        return f"{key}=${{{key}}}"
    return f"{key}={redact_sensitive_value(raw_value)}"


def resolve_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """
    Resolve environment variable templates in a dictionary.

    Templates like "${VAR_NAME}" are replaced with values from os.environ.
    Use "${VAR_NAME:-default}" to provide a default when the variable is unset.
    Literal values are passed through unchanged.

    Args:
        env_dict: Dictionary with potentially templated values

    Returns:
        Dictionary with resolved values

    Raises:
        ValueError: If a required environment variable is not found and no default
    """
    resolved = {}

    for key, value in env_dict.items():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if match:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in os.environ:
                resolved[key] = os.environ[var_name]
            elif default is not None:
                resolved[key] = default
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' not found in host environment"
                )
        else:
            # Literal value
            resolved[key] = value

    return resolved


def get_required_host_vars(
    env_dict: dict[str, str],
) -> list[tuple[str, str | None]]:
    """Extract host environment variable names referenced by templates.

    Returns a list of (var_name, default_or_None) for each ``${VAR}`` or
    ``${VAR:-default}`` entry.  Literal values are excluded.

    Args:
        env_dict: Dictionary with potentially templated values

    Returns:
        List of (var_name, default_value_or_None) tuples
    """
    result: list[tuple[str, str | None]] = []

    for value in env_dict.values():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if match:
            var_name = match.group(1)
            default = match.group(2)  # None when no :- clause
            result.append((var_name, default))

    return result
