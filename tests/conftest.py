"""Shared pytest fixtures for harbor tests."""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Force Rich/Click to render plain text, no matter the host terminal.

    The CLI's `rich.Console` instances decide whether to emit ANSI escape
    codes by inspecting env vars like `FORCE_COLOR` and `TERM` at
    construction time, not by checking whether the test's captured stream
    is a real tty. A developer or CI runner with `FORCE_COLOR` set would
    otherwise get colour codes injected into `CliRunner`/`capsys` output,
    splitting substrings that tests match against and making results
    depend on the environment the suite happens to run in. `NO_COLOR` is
    the standard override both Rich and Click honor even when a
    `FORCE_COLOR`-like signal is also present, so setting it here — before
    any test module (and the `rich.Console` instances it imports) is
    loaded — makes CLI output plain-text and deterministic everywhere.
    """
    os.environ["NO_COLOR"] = "1"


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use ProactorEventLoop on Windows to support asyncio subprocess.

    On Windows, the default SelectorEventLoop does not support subprocess
    operations. The ProactorEventLoop is required for asyncio.create_subprocess_exec
    and asyncio.create_subprocess_shell to work.
    """
    if sys.platform == "win32":
        return asyncio.WindowsProactorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    env = AsyncMock()
    env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    env.upload_file.return_value = None
    return env


@pytest.fixture
def mock_logs_dir(temp_dir):
    """Create a temporary logs directory."""
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir()
    return logs_dir


def run_validator_cli(trajectory_path: Path, extra_args: list[str] | None = None):
    """Run the trajectory validator CLI on a file.

    Args:
        trajectory_path: Path to the trajectory file to validate.
        extra_args: Optional list of additional CLI arguments.

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    cmd = [
        sys.executable,
        "-m",
        "harbor.utils.trajectory_validator",
        str(trajectory_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr
