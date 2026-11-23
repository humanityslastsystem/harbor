"""Session re-entry in ClaudeCode.run() (SessionReentry).

The shell is real: every command run() issues goes through bash, with the sandbox's
/logs/agent and /workspace mapped under a temp dir, and `claude` is a stub script on
PATH that plays one scripted invocation per launch. What is faked is the clock the loop
reads for invocation length and the budget it is given.
"""

import asyncio
import json
import os
import signal
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import harbor.agents.installed.claude_code as cc
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode, SessionReentry
from harbor.environments.base import ExecResult

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the re-entry wait reads /proc/<pid>/cmdline"
)

SESSION = "0b6f7a52-3c1e-4d2a-9f1e-2a7c9d4e5f60"
OTHER_SESSION = "7d1c2b3a-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
HOUR = 3600.0
# REENTRY_ERROR_PAUSE_SEC for these tests.
PAUSE = 0.05
STREAM_DROPPED = "API Error: upstream stream disconnected: unexpected EOF"

# One scripted invocation per launch, read from plan.json by launch number. Each step may
# set: rc, result (a subtype, or null for no result event), init (false: no init event),
# transcript (false: write no session file), session (the id it reports), sleep (seconds
# to run), leave_running (seconds a stand-in for a still-live claude outlives the exit),
# api_error (text the CLI reports for a failed model call, as the synthetic assistant
# message and the result event it writes then), stderr (a line printed there at the end),
# lines (raw lines printed to the stream before the result event) and result_fields (set on
# the result event).
FAKE_CLAUDE = r"""#!/usr/bin/env python3
import json, os, subprocess, sys, time

root = os.environ["STUB_ROOT"]
plan = json.load(open(os.path.join(root, "plan.json")))
count_path = os.path.join(root, "count")
n = int(open(count_path).read()) if os.path.exists(count_path) else 0
open(count_path, "w").write(str(n + 1))
step = plan[n]

def alive(pid):
    # A zombie has exited; only a process still running counts.
    try:
        stat = open("/proc/%%d/stat" %% pid).read()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"

# live.pid names the previous claude, or the stand-in that outlived it.
live = os.path.join(root, "live.pid")
overlapped = os.path.exists(live) and alive(int(open(live).read()))
open(live, "w").write(str(os.getpid()))
with open(os.path.join(root, "launches.jsonl"), "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "stdin": sys.stdin.read(),
                        "overlapped": overlapped}) + "\n")

sid = step.get("session", "%(session)s")
config_dir = os.environ["CLAUDE_CONFIG_DIR"]
if step.get("transcript", True):
    d = os.path.join(config_dir, "projects", "-workspace")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, sid + ".jsonl"), "a").write("{}\n")
if step.get("init", True):
    print(json.dumps({"type": "system", "subtype": "init", "session_id": sid}))
print(json.dumps({"type": "assistant", "session_id": sid,
                  "message": {"content": [{"type": "text", "text": "result"}]}}))
for line in step.get("lines", []):
    print(line)
sys.stdout.flush()
time.sleep(step.get("sleep", 0))
api_error = step.get("api_error")
if api_error is not None:
    # The shape Claude Code 2.1.251 writes when the model host drops a stream mid-reply.
    print(json.dumps({"type": "assistant", "session_id": sid, "is_api_error_message": True,
                      "message": {"model": "<synthetic>",
                                  "content": [{"type": "text", "text": api_error}]}}))
subtype = step.get("result", "success")
if subtype is not None:
    result = {"type": "result", "subtype": subtype, "session_id": sid,
              "total_cost_usd": round(0.1 * (n + 1), 1)}
    if api_error is not None:
        result.update(is_error=True, terminal_reason="api_error", result=api_error)
    result.update(step.get("result_fields", {}))
    print(json.dumps(result))
sys.stdout.flush()
if step.get("stderr"):
    print(step["stderr"], file=sys.stderr)
if step.get("leave_running"):
    p = subprocess.Popen(["bash", "-c", "sleep %%s; true" %% step["leave_running"], "stream-json"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    open(live, "w").write(str(p.pid))
    open(os.path.join(root, "leftover.pid"), "w").write(str(p.pid))
    open(os.path.join(os.path.dirname(config_dir), "claude.pid"), "w").write(str(p.pid))
sys.exit(step.get("rc", 0))
""" % {"session": SESSION}


class FakeClock:
    """The clock _drive reads through ``cc.time``. Each launch advances it by that
    launch's scripted duration; asyncio keeps the real clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class ShellEnv:
    """A sandbox stand-in that runs each command through real bash.

    ``durations[i]`` is how far the clock moves while launch ``i`` runs. A cancelled
    exec kills its process group, the way a harness cut ends the sandbox."""

    def __init__(self, root: Path, clock: FakeClock, durations: list[float]) -> None:
        self.root = root
        self.clock = clock
        self.durations = durations
        self.launches = 0
        self.home = root / "home"
        bin_dir = self.home / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        claude = bin_dir / "claude"
        claude.write_text(FAKE_CLAUDE)
        claude.chmod(0o755)
        (root / "logs" / "agent").mkdir(parents=True)
        (root / "workspace").mkdir()

    def _map(self, text: str) -> str:
        return text.replace("/logs/agent", f"{self.root}/logs/agent").replace(
            "/workspace", f"{self.root}/workspace"
        )

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        full_env = {
            **os.environ,
            **{k: self._map(v) for k, v in (env or {}).items()},
            "HOME": str(self.home),
            "STUB_ROOT": str(self.root),
        }
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            self._map(command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=full_env,
            cwd=self.root,
            start_new_session=True,
        )
        try:
            out, _ = await proc.communicate()
        except asyncio.CancelledError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            raise
        if "exec claude" in command:
            self.clock.now += self.durations[self.launches]
            self.launches += 1
        # communicate() returns only once the process has exited.
        rc = proc.returncode if proc.returncode is not None else 1
        return ExecResult(stdout=out.decode(), stderr=None, return_code=rc)


class Run:
    """One re-entry scenario: the agent, its shell, and what it recorded."""

    def __init__(
        self,
        root: Path,
        monkeypatch,
        plan,
        durations,
        budget,
        *,
        reentry=True,
        extra_env=None,
        secret_env=(),
        secret_forms=None,
    ):
        self.root = root
        (root / "plan.json").write_text(json.dumps(plan))
        self.clock = FakeClock()
        monkeypatch.setattr(
            cc, "time", types.SimpleNamespace(monotonic=self.clock.monotonic)
        )
        monkeypatch.setattr(cc, "_EXIT_POLL_SEC", 0.2)
        # Real seconds: the pause is an asyncio sleep, which the fake clock does not see.
        monkeypatch.setattr(cc, "REENTRY_ERROR_PAUSE_SEC", PAUSE)
        self.env = ShellEnv(root, self.clock, durations)
        self.events: list[tuple[str, dict]] = []
        self.prompts: list[float] = []
        self.agent = ClaudeCode(
            logs_dir=root / "logs" / "agent",
            inherit_process_env=False,
            extra_env=extra_env,
            secret_env=secret_env,
            **({"secret_forms": secret_forms} if secret_forms is not None else {}),
            ralph_kickoff="/goal follow the brief",
            reentry=(
                SessionReentry(
                    remaining_sec=lambda: budget - self.clock.now,
                    prompt=self._prompt,
                    on_event=lambda kind, fields: self.events.append((kind, fields)),
                )
                if reentry
                else None
            ),
        )

    def _prompt(self, remaining: float) -> str:
        self.prompts.append(remaining)
        return f"Budget left: {remaining:.0f} s. Keep working on server.py."

    async def go(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(
            self.agent.run("the brief", self.env, AsyncMock()), timeout
        )

    @property
    def launches(self) -> list[dict]:
        path = self.root / "launches.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    @property
    def stream(self) -> str:
        return (self.root / "logs" / "agent" / "claude-code.txt").read_text()

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]

    def kill_leftover(self) -> None:
        path = self.root / "leftover.pid"
        if path.exists():
            try:
                os.killpg(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_a_clean_exit_with_budget_left_resumes_the_same_session(
    temp_dir, monkeypatch
):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{}, {}, {}],
        durations=[HOUR] * 3,
        budget=3 * HOUR + 100,
    )
    await run.go()

    launches = run.launches
    assert len(launches) == 3
    assert "--resume" not in launches[0]["argv"]
    for launch in launches[1:]:
        argv = launch["argv"]
        assert argv[argv.index("--resume") + 1] == SESSION
        assert "--print" in argv
        assert "--input-format=stream-json" not in argv
    assert run.kinds() == ["reentered", "reentered", "session_ended"]
    first = run.events[0][1]
    assert first["reentries"] == 1
    assert first["session_id"] == SESSION
    assert first["result_subtype"] == "success"
    assert first["remaining_sec"] == pytest.approx(2 * HOUR + 100)
    assert first["invocation_sec"] == pytest.approx(HOUR)
    # A clean exit is re-entered at once and reports no error.
    assert first["exit_code"] == 0
    assert first["error"] is None
    assert first["paused_sec"] == 0
    ended = run.events[-1][1]
    assert ended["reason"] == "budget_spent"
    assert ended["reentries"] == 2


@pytest.mark.asyncio
async def test_the_reentry_prompt_arrives_on_stdin_never_argv(temp_dir, monkeypatch):
    # The prompt names server.py; on argv the agent's own `pkill -f server.py` would
    # match the claude process.
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{}, {}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    await run.go()
    second = run.launches[1]
    assert (
        second["stdin"]
        == f"Budget left: {HOUR + 100:.0f} s. Keep working on server.py."
    )
    assert not any("server.py" in arg for arg in second["argv"])
    assert run.prompts == [pytest.approx(HOUR + 100)]


@pytest.mark.asyncio
async def test_the_stream_tee_appends_across_reentries(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{}, {}, {}],
        durations=[HOUR] * 3,
        budget=3 * HOUR + 100,
    )
    (temp_dir / "logs" / "agent" / "claude-code.txt").write_text("stale\n")
    await run.go()
    events = [json.loads(line) for line in run.stream.splitlines()]
    # The first invocation truncates; each re-entry appends after its marker line.
    assert "stale" not in run.stream
    assert [e["type"] for e in events] == [
        "system", "assistant", "result",
        cc.REENTRY_MARKER_TYPE, "system", "assistant", "result",
        cc.REENTRY_MARKER_TYPE, "system", "assistant", "result",
    ]  # fmt: skip
    markers = [e for e in events if e["type"] == cc.REENTRY_MARKER_TYPE]
    assert [m["reentry"] for m in markers] == [1, 2]
    # Each stub process reports its own spend, so the run's cost is their sum.
    results = [e for e in events if e["type"] == "result"]
    assert [r["total_cost_usd"] for r in results] == [0.1, 0.2, 0.3]
    assert run.agent._parse_total_cost_from_stream_json() == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_no_reentry_once_the_budget_floor_is_reached(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{}],
        durations=[HOUR],
        budget=HOUR + cc.REENTRY_FLOOR_SEC,
    )
    await run.go()
    assert len(run.launches) == 1
    assert run.kinds() == ["session_ended"]
    assert run.events[0][1]["reason"] == "budget_spent"


@pytest.mark.asyncio
async def test_no_reentry_without_a_policy(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{}],
        durations=[HOUR],
        budget=12 * HOUR,
        reentry=False,
    )
    await run.go()
    assert len(run.launches) == 1


@pytest.mark.asyncio
async def test_a_nonzero_exit_with_budget_left_is_reentered(temp_dir, monkeypatch):
    # The model host dropped a stream mid-reply, and the CLI exited 1.
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"rc": 1, "api_error": STREAM_DROPPED}, {}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    await run.go()
    launches = run.launches
    assert len(launches) == 2
    argv = launches[1]["argv"]
    assert argv[argv.index("--resume") + 1] == SESSION
    assert run.kinds() == ["reentered", "session_ended"]
    first = run.events[0][1]
    assert first["exit_code"] == 1
    assert first["terminal_reason"] == "api_error"
    assert first["error"] == STREAM_DROPPED
    assert first["paused_sec"] == pytest.approx(PAUSE)
    ended = run.events[1][1]
    assert ended["reason"] == "budget_spent"
    assert ended["exit_code"] == 0


@pytest.mark.asyncio
async def test_without_a_policy_a_nonzero_exit_raises(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"rc": 1}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
        reentry=False,
    )
    with pytest.raises(NonZeroAgentExitCodeError):
        await run.go()
    assert len(run.launches) == 1


@pytest.mark.asyncio
async def test_fast_nonzero_exits_end_at_the_bound_and_raise(temp_dir, monkeypatch):
    # An invocation that fails the same way every time it starts.
    fast = cc.REENTRY_FAST_EXIT_SEC - 1
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"rc": 1, "api_error": "API Error: 401 invalid token"}] * 10,
        durations=[fast] * 10,
        budget=12 * HOUR,
    )
    with pytest.raises(NonZeroAgentExitCodeError):
        await run.go()
    assert len(run.launches) == 1 + cc.REENTRY_MAX_FAST_EXITS
    assert run.kinds() == ["reentered"] * cc.REENTRY_MAX_FAST_EXITS + ["session_ended"]
    # The pause doubles with each fast exit before it.
    paused = [fields["paused_sec"] for _, fields in run.events[:-1]]
    assert paused == pytest.approx(
        [PAUSE * 2**i for i in range(cc.REENTRY_MAX_FAST_EXITS)]
    )
    ended = run.events[-1][1]
    assert ended["reason"] == "fast_exits"
    assert ended["exit_code"] == 1
    assert ended["error"] == "API Error: 401 invalid token"
    assert ended["paused_sec"] == 0


@pytest.mark.asyncio
async def test_a_nonzero_exit_that_opened_no_session_raises_at_once(
    temp_dir, monkeypatch
):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[
            {
                "rc": 2,
                "init": False,
                "result": None,
                "transcript": False,
                "stderr": "error: unknown option '--bogus'",
            },
            {},
        ],
        durations=[1, HOUR],
        budget=12 * HOUR,
    )
    with pytest.raises(NonZeroAgentExitCodeError):
        await run.go()
    assert len(run.launches) == 1
    assert run.kinds() == ["session_ended"]
    ended = run.events[0][1]
    assert ended["reason"] == "no_session"
    assert ended["exit_code"] == 2
    # With no result event, the error is the last line the CLI printed.
    assert ended["error"] == "error: unknown option '--bogus'"
    assert ended["paused_sec"] == 0


@pytest.mark.asyncio
async def test_a_nonzero_exit_at_the_budget_floor_raises(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"rc": 1, "api_error": STREAM_DROPPED}, {}],
        durations=[HOUR, HOUR],
        budget=HOUR + cc.REENTRY_FLOOR_SEC,
    )
    with pytest.raises(NonZeroAgentExitCodeError):
        await run.go()
    assert len(run.launches) == 1
    assert run.events[0][1]["reason"] == "budget_spent"


@pytest.mark.asyncio
async def test_a_cut_during_the_pause_starts_nothing(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"rc": 1, "api_error": STREAM_DROPPED}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
    )
    monkeypatch.setattr(cc, "REENTRY_ERROR_PAUSE_SEC", 30.0)
    with pytest.raises(asyncio.TimeoutError):
        await run.go(timeout=2.0)
    assert len(run.launches) == 1
    assert run.events == []
    assert not (temp_dir / "logs" / "agent" / "reentry-prompt.txt").exists()


@pytest.mark.asyncio
async def test_a_nonzero_exit_that_cannot_be_examined_raises_the_exit(
    temp_dir, monkeypatch
):
    # A sandbox gone by the time the loop reads the stream must not turn the agent's exit
    # into some other error the build does not collect after.
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"rc": 1, "api_error": STREAM_DROPPED}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
    )
    monkeypatch.setattr(
        run.agent, "_read_exit", AsyncMock(side_effect=RuntimeError("sandbox gone"))
    )
    with pytest.raises(NonZeroAgentExitCodeError, match="exit 1"):
        await run.go()
    assert len(run.launches) == 1
    assert run.events == []


@pytest.mark.asyncio
async def test_a_harness_cut_ends_the_run_without_reentry(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"sleep": 30}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
    )
    with pytest.raises(asyncio.TimeoutError):
        await run.go(timeout=2.0)
    assert len(run.launches) == 1
    assert run.events == []
    assert not (temp_dir / "logs" / "agent" / "reentry-prompt.txt").exists()


@pytest.mark.asyncio
async def test_consecutive_fast_reentries_end_the_loop(temp_dir, monkeypatch):
    fast = cc.REENTRY_FAST_EXIT_SEC - 1
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{}] * 10,
        durations=[HOUR] + [fast] * 9,
        budget=12 * HOUR,
    )
    await run.go()
    assert len(run.launches) == 1 + cc.REENTRY_MAX_FAST_EXITS
    assert run.kinds() == ["reentered"] * cc.REENTRY_MAX_FAST_EXITS + ["session_ended"]
    ended = run.events[-1][1]
    assert ended["reason"] == "fast_exits"
    assert ended["fast_exits"] == cc.REENTRY_MAX_FAST_EXITS


@pytest.mark.asyncio
async def test_a_reentry_that_works_resets_the_fast_exit_count(temp_dir, monkeypatch):
    fast = cc.REENTRY_FAST_EXIT_SEC - 1
    durations = [HOUR, fast, fast, HOUR, fast, fast, fast, fast]
    run = Run(
        temp_dir, monkeypatch, plan=[{}] * 8, durations=durations, budget=12 * HOUR
    )
    await run.go()
    # Two fast re-entries, one slow one that resets the count, then three fast ones.
    assert len(run.launches) == 7
    assert run.events[-1][1]["reason"] == "fast_exits"


@pytest.mark.asyncio
async def test_a_cap_the_cli_enforced_is_not_resumed_past(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"result": "error_max_turns"}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
    )
    await run.go()
    assert len(run.launches) == 1
    assert run.events[0][1]["reason"] == "cap_reached"


@pytest.mark.asyncio
async def test_without_a_result_event_the_session_id_comes_from_init(
    temp_dir, monkeypatch
):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"result": None}, {}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    await run.go()
    argv = run.launches[1]["argv"]
    assert argv[argv.index("--resume") + 1] == SESSION
    assert run.events[0][1]["result_subtype"] is None


@pytest.mark.asyncio
async def test_no_session_id_in_the_stream_ends_the_loop(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"result": None, "init": False}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
    )
    await run.go()
    assert len(run.launches) == 1
    assert run.events[0][1]["reason"] == "no_session"


# A relay key shaped as Claude Code writes a session id.
UUID_KEY = "4c1d2e3f-5a6b-4c7d-8e9f-0a1b2c3d4e5f"
# A relay host shaped as one: a single label of hex digits and hyphens.
UUID_HOST = "9a8b7c6d-1e2f-4a3b-8c4d-5e6f7a8b9c0d"
SECRET_ENV = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def _with_host(name, value):
    """The caller's forms of a secret, as hls gives them: the value, and for the base URL its
    host, which the caller's rule says identifies a machine."""
    return (value, UUID_HOST) if name == "ANTHROPIC_BASE_URL" else (value,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session, key, url",
    [
        ("--continue", UUID_KEY, "https://relay.example/v1"),
        (SESSION.upper(), UUID_KEY, "https://relay.example/v1"),
        (UUID_KEY, UUID_KEY, "https://relay.example/v1"),
        (UUID_KEY, UUID_KEY.upper(), "https://relay.example/v1"),
        (UUID_HOST, UUID_KEY, f"https://{UUID_HOST}:8443/v1"),
    ],
    ids=["flag", "upper-case", "secret-value", "secret-value-any-case", "secret-host"],
)
async def test_a_session_id_claude_code_would_not_write_ends_the_loop(
    temp_dir, monkeypatch, session, key, url
):
    """The stream is the agent's to write: only a lowercase UUID holding none of the forms of
    the agent's secret extra_env values, in any case, is resumed; anything else is no session.
    The last three ids are UUIDs, so only the forms stop them: the value, and the host the
    caller's forms name."""
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"session": session}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
        extra_env={"ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_BASE_URL": url},
        secret_env=SECRET_ENV,
        secret_forms=_with_host,
    )
    await run.go()
    assert len(run.launches) == 1
    assert (run.events[0][1]["reason"], run.events[0][1]["session_id"]) == (
        "no_session",
        None,
    )


@pytest.mark.asyncio
async def test_a_session_with_no_transcript_on_disk_is_not_resumed(
    temp_dir, monkeypatch
):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"transcript": False}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
    )
    await run.go()
    assert len(run.launches) == 1
    assert run.events[0][1]["reason"] == "no_session"
    assert not (temp_dir / "logs" / "agent" / "reentry-prompt.txt").exists()


@pytest.mark.asyncio
async def test_the_loop_follows_the_session_the_last_invocation_reported(
    temp_dir, monkeypatch
):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{}, {"session": OTHER_SESSION}, {"session": OTHER_SESSION}],
        durations=[HOUR] * 3,
        budget=3 * HOUR + 100,
    )
    await run.go()
    argv = run.launches[2]["argv"]
    assert argv[argv.index("--resume") + 1] == OTHER_SESSION


@pytest.mark.asyncio
async def test_a_reentry_waits_until_the_previous_process_is_gone(
    temp_dir, monkeypatch
):
    # The first exec returns while a process standing in for claude is still alive
    # under the recorded PID; the second launch must not start until it has exited.
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"leave_running": 2}, {}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    try:
        await run.go()
    finally:
        run.kill_leftover()
    launches = run.launches
    assert len(launches) == 2
    assert [launch["overlapped"] for launch in launches] == [False, False]
    assert run.events[0][1]["waited_sec"] >= 1


@pytest.mark.asyncio
async def test_a_cut_during_that_wait_starts_nothing(temp_dir, monkeypatch):
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"leave_running": 30}, {}],
        durations=[HOUR, HOUR],
        budget=12 * HOUR,
    )
    try:
        with pytest.raises(asyncio.TimeoutError):
            await run.go(timeout=3.0)
    finally:
        run.kill_leftover()
    assert len(run.launches) == 1
    assert run.events == []


@pytest.mark.asyncio
async def test_invocations_never_overlap(temp_dir, monkeypatch):
    # Every launch checks, on start, whether the claude launched before it is running.
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"sleep": 0.3}] * 4,
        durations=[HOUR] * 4,
        budget=4 * HOUR + 100,
    )
    await run.go()
    assert len(run.launches) == 4
    assert not any(launch["overlapped"] for launch in run.launches)


@pytest.mark.asyncio
async def test_the_overlap_check_sees_a_launch_that_did_not_wait(temp_dir, monkeypatch):
    # The check above must be able to fail. Pointed at a PID file nothing live is in, the
    # loop has nothing to wait for, and the stub sees the stand-in still running.
    monkeypatch.setattr(cc, "PID_PATH", "/logs/agent/elsewhere.pid")
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"leave_running": 3}, {}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    try:
        await run.go()
    finally:
        run.kill_leftover()
    assert [launch["overlapped"] for launch in run.launches] == [False, True]


@pytest.mark.asyncio
async def test_a_session_id_holding_a_plain_setting_is_resumed(temp_dir, monkeypatch):
    """Only the secret values of extra_env are compared: a real id that holds a plain
    setting's value, here the output limit, is resumed."""
    session = "0a64000f-1111-4222-8333-444444444444"
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"session": session}, {"session": session}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
        extra_env={
            "ANTHROPIC_AUTH_TOKEN": UUID_KEY,
            "ANTHROPIC_BASE_URL": "https://relay.example/v1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
        },
        secret_env=SECRET_ENV,
    )
    await run.go()
    argv = run.launches[1]["argv"]
    assert argv[argv.index("--resume") + 1] == session


@pytest.mark.asyncio
async def test_a_host_the_callers_forms_do_not_name_is_no_secret(temp_dir, monkeypatch):
    """Which hosts mark a secret is the caller's rule, the one its scrubber redacts by (hls: a
    URL's host that identifies a machine), never harbor's own: without a form naming it, a
    session id that is the base URL's single-label host is resumed."""
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"session": UUID_HOST}, {"session": UUID_HOST}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
        extra_env={
            "ANTHROPIC_AUTH_TOKEN": UUID_KEY,
            "ANTHROPIC_BASE_URL": f"https://{UUID_HOST}:8443/v1",
        },
        secret_env=SECRET_ENV,
    )
    await run.go()
    argv = run.launches[1]["argv"]
    assert argv[argv.index("--resume") + 1] == UUID_HOST


def test_a_secret_name_the_extra_env_does_not_hold_is_refused(temp_dir):
    with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN"):
        ClaudeCode(
            logs_dir=temp_dir,
            extra_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"},
            secret_env=("ANTHROPIC_AUTH_TOKEN",),
        )


@pytest.mark.asyncio
async def test_a_stream_line_the_decoder_cannot_take_is_no_evidence(
    temp_dir, monkeypatch, caplog
):
    """A line holding "result" and nested 50,000 deep, among the stream's first lines and
    its last result lines: it is unreadable, a warning says so, and the session id comes
    from the init event."""
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[
            {"result": None, "lines": ['{"type": "result", "x": ' + "[" * 50_000]},
            {},
        ],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    with caplog.at_level("WARNING"):
        await run.go()
    argv = run.launches[1]["argv"]
    assert argv[argv.index("--resume") + 1] == SESSION
    assert any("could not be read" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"subtype": "x" * 1000},
        {"terminal_reason": "api error; " * 10},
        {"subtype": ["error_max_turns"], "terminal_reason": {"a": 1}},
    ],
    ids=["long", "words", "not-strings"],
)
async def test_a_result_field_that_is_no_short_word_is_none(
    temp_dir, monkeypatch, fields
):
    """The result's subtype and terminal_reason reach every event: each is kept only as a
    word of at most 64 letters, digits and ``_.:-``."""
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[{"result_fields": fields}, {}],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    await run.go()
    kind, event = run.events[0]
    assert kind == "reentered"
    assert (event["result_subtype"], event["terminal_reason"]) == (
        None if "subtype" in fields else "success",
        None,
    )


@pytest.mark.asyncio
async def test_a_result_line_past_the_cut_is_no_evidence(temp_dir, monkeypatch):
    """A result line longer than _EVENT_LINE_CHARS is cut, so it is unreadable, and the
    session id comes from the init event rather than from it."""
    run = Run(
        temp_dir,
        monkeypatch,
        plan=[
            {"result_fields": {"session_id": OTHER_SESSION, "pad": "x" * 70_000}},
            {},
        ],
        durations=[HOUR, HOUR],
        budget=2 * HOUR + 100,
    )
    await run.go()
    argv = run.launches[1]["argv"]
    assert argv[argv.index("--resume") + 1] == SESSION
