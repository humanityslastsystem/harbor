"""Codex goal kickoff: the two-call drive that keeps a long build alive.

Measured behaviour of the CLI (codex-cli 0.146.0), which this shape encodes:

  - `codex exec` runs ONE agentic turn and the process exits. With no goal it emits
    `task_complete`; with a goal outstanding it emits `turn_aborted` instead -- codex
    refuses to signal completion while a goal is unmet.
  - `codex exec resume` on a thread with an OUTSTANDING goal runs turn after turn in a
    single process (verified: 4 turns started, 3 completed, still going when killed at
    300s) and returns only when the goal is met -- verified separately: satisfying the
    goal made the same resume exit in 17s.

So the run is driven by codex's own goal machinery, not by an outer loop: call 1
establishes the goal, call 2 is the single long-lived drive bounded by the
orchestrator's wall-clock cap.
"""

import re
from unittest.mock import AsyncMock

import pytest

import harbor.agents.installed.codex as cx
from harbor.agents.installed.codex import Codex


def _env():
    env = AsyncMock()
    env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    return env


def _commands(env):
    return [c.kwargs["command"] for c in env.exec.call_args_list]


@pytest.mark.asyncio
async def test_goal_kickoff_drives_exec_then_resume(temp_dir):
    agent = Codex(logs_dir=temp_dir, model_name="m", goal_kickoff="Create a goal ...")
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    cmds = _commands(env)
    execs = [c for c in cmds if "codex exec " in c and "resume" not in c]
    resumes = [c for c in cmds if "codex exec resume " in c]
    assert len(execs) == 1, "exactly one goal-establishing exec"
    assert len(resumes) == 1, "exactly one drive; more would be an outer loop"
    # The drive resumes the thread call 1 opened, not `--last`: a model that spawns
    # sub-agents forks a newer thread per sub-agent, and `--last` would pick one of
    # those, which the app-server refuses direct input to.
    assert 'codex exec resume "${tid:---last}"' in resumes[0]
    assert "thread\\.started" in resumes[0]
    # Codex's own sandbox is redundant inside the container and its bubblewrap helper
    # cannot pivot_root there. BOTH calls must bypass it: bypassing only the kickoff
    # leaves the drive -- where a build spends its time -- sandboxed and failing every
    # shell command and apply_patch.
    for c in execs + resumes:
        assert "--dangerously-bypass-approvals-and-sandbox" in c
        assert "--skip-git-repo-check" in c
        assert "--ignore-rules" in c
    # Disabling the sandbox by feature flag does not work, and use_legacy_landlock is
    # deprecated in codex 0.146.0 and injects an error item into every turn.
    assert not any("--disable use_" in c for c in execs + resumes)
    assert cmds.index(execs[0]) < cmds.index(resumes[0]), "resume must follow exec"


@pytest.mark.asyncio
async def test_resume_runs_on_the_arms_model_not_codexs_default(temp_dir):
    # A resumed thread does not keep its recorded model. Without --model the drive
    # runs every turn after the first on codex's default model.
    agent = Codex(
        logs_dir=temp_dir, model_name="openai/gpt-5.5", goal_kickoff="Create a goal ..."
    )
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    cmds = _commands(env)
    execs = [c for c in cmds if "codex exec " in c and "resume" not in c]
    resumes = [c for c in cmds if "codex exec resume " in c]
    for c in execs + resumes:
        assert "--model gpt-5.5 " in c


@pytest.mark.asyncio
async def test_kickoff_not_the_brief_is_the_opening_prompt(temp_dir):
    # The brief is a FILE; only the kickoff rides the command line. A brief in argv
    # would let the agent's own `pkill -f server.py` match the harness process.
    agent = Codex(logs_dir=temp_dir, model_name="m", goal_kickoff="Create a goal ...")
    env = _env()
    await agent.run("build server.py and keep it running", env, AsyncMock())
    opening = [c for c in _commands(env) if "codex exec " in c and "resume" not in c][0]
    assert "Create a goal ..." in opening
    assert "build server.py and keep it running" not in opening


@pytest.mark.asyncio
async def test_brief_written_to_workspace_and_persisted_by_copy(temp_dir):
    agent = Codex(logs_dir=temp_dir, model_name="m", goal_kickoff="Create a goal ...")
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    write = next(c for c in _commands(env) if cx.Codex._WORKSPACE_BRIEF_PATH in c)
    assert f"> {cx.Codex._WORKSPACE_BRIEF_PATH}" in write
    # `cp`, not a second render: the recorded bytes ARE the agent's bytes.
    assert f"cp {cx.Codex._WORKSPACE_BRIEF_PATH} {cx.Codex._BRIEF_PATH}" in write


@pytest.mark.asyncio
async def test_no_kickoff_keeps_the_single_shot_shape(temp_dir):
    # Without a soft wall-clock budget there is no goal to drive, so the run is one
    # call -- no resume at all.
    agent = Codex(logs_dir=temp_dir, model_name="m")
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    assert not [c for c in _commands(env) if "resume" in c]


@pytest.mark.asyncio
async def test_session_survives_until_codex_home_is_removed(temp_dir):
    # $CODEX_HOME holds the thread `resume` re-enters; removing it between the two
    # calls would make the drive impossible. It may only go after both, and after the
    # destination the sessions symlink points at exists.
    agent = Codex(logs_dir=temp_dir, model_name="m", goal_kickoff="Create a goal ...")
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    cmds = _commands(env)
    rm = next(i for i, c in enumerate(cmds) if "rm -rf" in c and '"$CODEX_HOME"' in c)
    resume = next(i for i, c in enumerate(cmds) if "codex exec resume " in c)
    dest = next(
        i for i, c in enumerate(cmds) if c.endswith("mkdir -p /logs/agent/sessions")
    )
    assert resume < dest < rm


def _extraction_shell(cmds):
    """The `tid=...` prelude of the drive command, retargeted at $OUT so a test can run
    it against a fixture."""
    drive = next(c for c in cmds if "codex exec resume " in c)
    prelude = drive.split("codex exec resume ", 1)[0].split("fi; ", 1)[1]
    log = re.search(r"grep -m1 '[^']*' (\S+)", prelude).group(1)
    return prelude.replace(log, '"$OUT"')


@pytest.mark.parametrize(
    "name,stream,want",
    [
        # What codex 0.146.0 actually emits.
        (
            "canonical",
            '{"type":"thread.started","thread_id":"aaaaaaaa-0000-7000-8000-000000000001"}\n',
            "aaaaaaaa-0000-7000-8000-000000000001",
        ),
        # Field order and spacing are not part of any contract; a pattern that assumed them
        # would fall through to --last and quietly resume whatever thread is newest.
        (
            "reordered",
            '{"thread_id":"aaaaaaaa-0000-7000-8000-000000000001","type":"thread.started"}\n',
            "aaaaaaaa-0000-7000-8000-000000000001",
        ),
        (
            "spaced",
            '{ "type": "thread.started", "extra": 1, "thread_id": "aaaaaaaa-0000-7000-8000-000000000001" }\n',
            "aaaaaaaa-0000-7000-8000-000000000001",
        ),
        # The whole point: a sub-agent fork announces itself later in the same stream, and the
        # root thread is the one that must be resumed.
        (
            "fork_follows",
            '{"type":"thread.started","thread_id":"aaaaaaaa-0000-7000-8000-000000000001"}\n'
            '{"type":"item.completed","item":{"id":"item_1"}}\n'
            '{"type":"thread.started","thread_id":"aaaaaaaa-0000-7000-8000-000000000002"}\n',
            "aaaaaaaa-0000-7000-8000-000000000001",
        ),
        # A stream that never announced a thread leaves tid empty; the command falls back.
        ("no_thread", '{"type":"item.completed","item":{"id":"item_1"}}\n', ""),
    ],
)
@pytest.mark.asyncio
async def test_thread_id_extraction_survives_json_shape(temp_dir, name, stream, want):
    import subprocess

    agent = Codex(logs_dir=temp_dir, model_name="m", goal_kickoff="Create a goal ...")
    env = _env()
    await agent.run("the brief", env, AsyncMock())
    fixture = temp_dir / f"{name}.txt"
    fixture.write_text(stream)
    shell = _extraction_shell(_commands(env)) + 'printf "%s" "${tid:---last}"'
    out = subprocess.run(
        ["bash", "-c", shell],
        capture_output=True,
        text=True,
        env={"OUT": str(fixture), "PATH": "/usr/bin:/bin"},
    )
    assert out.stdout == (want or "--last"), out.stderr
    if not want:
        # The fallback may resume a sub-agent fork, so it must not happen silently.
        assert "no thread.started" in out.stderr
