import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import parse_bool_env_value
from harbor.utils.jsonl import json_objects
from harbor.utils.trajectory_utils import format_trajectory_json

# The bytes of a session file line the readers parse; a longer one is skipped. The
# longest measured held 143 KB; the rest is room for an image read.
_SESSION_LINE_BYTES = 4 << 20


class Codex(BaseInstalledAgent):
    """
    The Codex agent uses OpenAI's Codex CLI tool to solve tasks.
    """

    SUPPORTS_ATIF: bool = True
    _OUTPUT_FILENAME = "codex.txt"
    _REMOTE_CODEX_HOME = PurePosixPath("/tmp/codex-home")
    _REMOTE_CODEX_SECRETS_DIR = PurePosixPath("/tmp/codex-secrets")
    #: The brief the agent reads, in its WORKING DIRECTORY -- same contract as the
    #: claude-code adapter, so both harnesses see the task the same way. Never argv: a
    #: brief that names server.py would make the agent's own `pkill -f server.py` match
    #: the harness process.
    _WORKSPACE_DIR = "/workspace"
    _WORKSPACE_BRIEF_PATH = "/workspace/BRIEF.md"
    #: Persisted copy (a mount -> lands in the rollout record), produced by `cp` so the
    #: recorded bytes are the bytes the agent was given.
    _BRIEF_PATH = "/logs/agent/brief.md"
    #: Prompt for the resume that drives the run. Deliberately minimal: the thread
    #: already carries the brief, the goal and everything done so far, so this only has
    #: to hand control back. The GOAL is what keeps the agent working -- codex will not
    #: signal task_complete while one is outstanding -- not this text.
    _RESUME_PROMPT = "Continue working toward the goal."
    #: Mode of codex's hosted web search tool for every run. Codex offers the tool
    #: unless told otherwise (codex 0.146.0 under the sandbox bypass below sends it with
    #: live web access). The search runs on the model provider's side, so no egress
    #: rule in the sandbox sees it, and an agent whose brief says it is offline could
    #: still search the web. Set twice: as a key of the config.toml harbor writes, so
    #: the recorded config shows it, and as a `-c` override on both codex calls,
    #: because a `.codex/config.toml` in the agent's working directory takes precedence
    #: over $CODEX_HOME's and the agent can write one.
    _WEB_SEARCH_MODE = "disabled"
    #: The model provider block written into config.toml when a base URL is set. The
    #: shell puts the URL where ``{base_url}`` is (see _build_provider_config_command).
    _PROVIDER_TOML = (
        '\nmodel_provider = "custom"\n'
        "\n"
        "[model_providers.custom]\n"
        'name = "custom"\n'
        'base_url = "{base_url}"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = true\n"
    )

    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="-c",
            type="str",
            default="high",
            format="-c model_reasoning_effort={value}",
        ),
        CliFlag(
            "reasoning_summary",
            cli="-c",
            type="enum",
            choices=["auto", "concise", "detailed", "none"],
            format="-c model_reasoning_summary={value}",
        ),
    ]

    @staticmethod
    def name() -> str:
        return AgentName.CODEX.value

    def __init__(
        self, logs_dir: Path, *args, goal_kickoff: str | None = None, **kwargs
    ):
        # The instruction that asks the model to call its `create_goal` tool. Codex's
        # goal machinery is what keeps a long run alive: with a goal outstanding the
        # agent cannot signal task_complete (it emits turn_aborted instead), and
        # `codex exec resume` is driven turn after turn until the goal is met. Without
        # one, `codex exec` runs a single turn and exits -- measured.
        #
        # It must be phrased as a REQUEST, not as a slash command: `/goal ...` is
        # interpreted by interactive clients (TUI/app) and never reaches the model, so
        # passing it as prompt text is silently inert.
        self._goal_kickoff = goal_kickoff
        super().__init__(logs_dir, *args, **kwargs)

    @property
    def _trajectory_path(self) -> PurePosixPath:
        return PurePosixPath(EnvironmentPaths.agent_dir / "trajectory.json")

    def get_version_command(self) -> str | None:
        return "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex --version"

    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line.removeprefix("codex-cli").strip()
        return text

    async def install(self, environment: BaseEnvironment) -> None:
        # Already baked into the image? Then do NOTHING. An air-gapped sandbox bakes both
        # CLIs precisely because its build seal blocks the npm registry and
        # raw.githubusercontent.com; attempting the install below turns a working
        # environment into a hard failure at agent setup.
        #
        # Same shape as ClaudeCode's probe: ``exec_as_agent`` RAISES on a nonzero exit
        # rather than returning a status, so "not found" lands in the except and falls
        # through to install. Skipped when a version is pinned -- an explicit pin means
        # install THAT version rather than accept whatever is present.
        if self._version is None:
            try:
                await self.exec_as_agent(
                    environment,
                    command='export PATH="$HOME/.local/bin:$PATH"; command -v codex',
                )
                self.logger.debug("codex already present (baked); skipping install")
                return
            except RuntimeError:
                pass  # codex not present -> install it below
        # Install system packages (root)
        await self.exec_as_root(
            environment,
            command=(
                "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
                "  apk add --no-cache curl bash nodejs npm ripgrep;"
                " elif command -v apt-get &>/dev/null; then"
                "  apt-get update && apt-get install -y curl ripgrep;"
                " elif command -v yum &>/dev/null; then"
                "  yum install -y curl ripgrep;"
                " else"
                '  echo "Warning: No known package manager found, assuming curl is available" >&2;'
                " fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        # Install codex (as default user)
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
                f"  npm install -g @openai/codex{version_spec};"
                " else"
                "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash &&"
                '  export NVM_DIR="$HOME/.nvm" &&'
                '  \\. "$NVM_DIR/nvm.sh" || true &&'
                "  command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } &&"
                "  nvm install 22 && nvm alias default 22 && npm -v &&"
                f"  npm install -g @openai/codex{version_spec};"
                " fi && "
                "codex --version"
            ),
        )
        # Symlink node/codex to /usr/local/bin (root)
        await self.exec_as_root(
            environment,
            command=(
                "for bin in node codex; do"
                '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
                '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
                '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
                "  fi;"
                " done"
            ),
        )

    def _get_session_dir(self) -> Path | None:
        """Get the single session directory."""
        sessions_dir = self.logs_dir / "sessions"
        if not sessions_dir.exists():
            return None

        session_dirs = [d for d in sessions_dir.rglob("*") if d.is_dir()]
        if not session_dirs:
            return None
        max_depth = max(len(d.parts) for d in session_dirs)
        session_dirs = [d for d in session_dirs if len(d.parts) == max_depth]
        if not session_dirs:
            return None

        # Sanity check: there should be exactly one session
        if len(session_dirs) != 1:
            raise ValueError(f"Expected exactly 1 session, found {len(session_dirs)}")
        return session_dirs[0]

    @staticmethod
    def _extract_message_text(content: list[Any]) -> str:
        """Extract joined text from Codex content blocks."""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _parse_output_blob(raw: Any) -> tuple[str | None, dict[str, Any] | None]:
        """Extract textual output and metadata from Codex tool outputs."""
        if raw is None:
            return None, None

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:  # noqa: BLE001 -- the agent's commands wrote the output
                return raw, None
        else:
            parsed = raw

        if isinstance(parsed, dict):
            output = parsed.get("output")
            if output is None and parsed:
                # dumping remaining structure if output missing
                output = json.dumps(parsed, ensure_ascii=False)
            metadata = parsed.get("metadata")
            return output, metadata if isinstance(metadata, dict) else None

        return str(parsed), None

    def _convert_event_to_step(self, event: dict[str, Any], step_id: int) -> Step:
        """Convert a normalized Codex event dictionary into an ATIF step."""
        kind = event.get("kind")
        timestamp = event.get("timestamp")

        if kind == "message":
            role = event.get("role", "user")
            text = event.get("text", "")
            reasoning = event.get("reasoning")
            source: Literal["system", "user", "agent"]
            if role == "assistant":
                source = "agent"
            elif role == "user":
                source = "user"
            else:
                source = "system"

            extra = event.get("extra")

            return Step(
                step_id=step_id,
                timestamp=timestamp,
                source=source,
                message=text,
                reasoning_content=reasoning
                if source == "agent" and reasoning
                else None,
                model_name=self.model_name
                if source == "agent" and self.model_name
                else None,
                extra=extra if extra else None,
            )

        if kind == "tool_call":
            call_id = event.get("call_id", "")
            tool_name = event.get("tool_name", "")
            reasoning = event.get("reasoning")
            arguments = event.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}

            tool_call = ToolCall(
                tool_call_id=call_id,
                function_name=tool_name,
                arguments=arguments,
            )

            observation: Observation | None = None
            output_text = event.get("output")
            if output_text is not None:
                observation = Observation(
                    results=[
                        ObservationResult(
                            source_call_id=call_id or None,
                            content=output_text,
                        )
                    ]
                )

            metrics_payload = event.get("metrics")
            metrics: Metrics | None = None
            if isinstance(metrics_payload, dict):
                metrics = Metrics(**metrics_payload)

            extra: dict[str, Any] | None = None
            metadata = event.get("metadata")
            if metadata:
                extra = {"tool_metadata": metadata}
            raw_arguments = event.get("raw_arguments")
            if raw_arguments:
                extra = extra or {}
                extra["raw_arguments"] = raw_arguments
            status = event.get("status")
            if status:
                extra = extra or {}
                extra["status"] = status

            message = event.get("message")
            if not message:
                summary_parts = [part for part in [tool_name, call_id] if part]
                summary = " ".join(summary_parts) if summary_parts else "Tool call"
                message = f"Executed {summary}"

            return Step(
                step_id=step_id,
                timestamp=timestamp,
                source="agent",
                message=message,
                tool_calls=[tool_call],
                observation=observation,
                model_name=self.model_name if self.model_name else None,
                reasoning_content=reasoning if reasoning else None,
                metrics=metrics,
                extra=extra,
            )

        raise ValueError(f"Unsupported event kind '{kind}'")

    def _compute_cost_from_pricing(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
    ) -> float | None:
        """Compute total cost in USD from token counts via LiteLLM's pricing table.

        Codex CLI's session JSONL never includes a cost field, so we apply
        LiteLLM's per-token rates to the aggregated counts ourselves. Returns
        None when the model is missing from the table — caller should leave
        cost_usd unset rather than report a misleading $0.
        """
        if not self.model_name:
            return None

        try:
            import litellm
        except ImportError:
            self.logger.warning("litellm not available; leaving codex cost_usd as None")
            return None

        pricing: dict[str, Any] | None = None
        for key in (self.model_name, self.model_name.split("/", 1)[-1]):
            entry = litellm.model_cost.get(key)
            if entry:
                pricing = entry
                break

        if pricing is None:
            self.logger.warning(
                "No LiteLLM pricing entry for model '%s'; leaving codex "
                "cost_usd as None",
                self.model_name,
            )
            return None

        input_rate = pricing.get("input_cost_per_token") or 0.0
        output_rate = pricing.get("output_cost_per_token") or 0.0
        cache_read_rate = pricing.get("cache_read_input_token_cost", input_rate)
        if cache_read_rate is None:
            cache_read_rate = input_rate

        uncached_input = max(0, (prompt_tokens or 0) - (cached_tokens or 0))
        cached = cached_tokens or 0
        output = completion_tokens or 0

        return (
            uncached_input * input_rate
            + cached * cache_read_rate
            + output * output_rate
        )

    @staticmethod
    def _is_subagent_session(session_file: Path) -> bool:
        """Whether a session file records a spawned subagent rather than the root run.

        Codex writes one file per thread into the same directory, so a run that spawns
        subagents leaves several. A spawned thread's ``session_meta`` carries the parent
        it forked from; the root run's does not.
        """
        for event in json_objects(session_file, max_line_bytes=_SESSION_LINE_BYTES):
            if event.get("type") != "session_meta":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                return False
            return bool(
                payload.get("forked_from_id")
                or payload.get("parent_thread_id")
                or payload.get("thread_source") == "subagent"
            )
        return False

    def _select_session_file(self, session_files: list[Path]) -> Path:
        """Pick the root run's session file from every thread the run wrote.

        Selecting by directory order or by name would both be wrong: subagent files sort
        arbitrarily under the former and, being created later, sort last under the latter.
        Reading a subagent's file silently reports that thread's steps and token usage as
        the whole run's.
        """
        roots = [f for f in session_files if not self._is_subagent_session(f)]
        if len(roots) == 1:
            return roots[0]

        fallback = max(session_files, key=lambda f: f.stat().st_size)
        self.logger.warning(
            f"Expected exactly 1 root Codex session in {session_files[0].parent}, "
            f"found {len(roots)} among {len(session_files)} files; "
            f"falling back to the largest, {fallback.name}. "
            "Reported steps and token usage may cover only part of the run."
        )
        return fallback

    def _convert_events_to_trajectory(self, session_dir: Path) -> Trajectory | None:
        """Convert Codex session JSONL events into an ATIF trajectory."""
        session_files = list(session_dir.glob("*.jsonl"))

        if not session_files:
            self.logger.debug(f"No Codex session files found in {session_dir}")
            return None

        session_file = self._select_session_file(session_files)

        raw_events = list(
            json_objects(
                session_file, max_line_bytes=_SESSION_LINE_BYTES, logger=self.logger
            )
        )

        if not raw_events:
            return None

        session_meta = next(
            (e for e in raw_events if e.get("type") == "session_meta"), None
        )
        session_id = (
            session_meta.get("payload", {}).get("id")
            if session_meta and isinstance(session_meta, dict)
            else session_dir.name
        )

        agent_version = "unknown"
        agent_extra: dict[str, Any] | None = None
        default_model_name: str | None = None

        if session_meta:
            payload = session_meta.get("payload", {})
            agent_version = payload.get("cli_version") or agent_version
            extra: dict[str, Any] = {}
            for key in ("originator", "cwd", "git", "instructions"):
                value = payload.get(key)
                if value is not None:
                    extra[key] = value
            agent_extra = extra or None

        if agent_version == "unknown" and self._version:
            agent_version = self._version

        for event in raw_events:
            if event.get("type") == "turn_context":
                model_name = event.get("payload", {}).get("model")
                if isinstance(model_name, str):
                    default_model_name = model_name
                    break

        if default_model_name is None:
            default_model_name = self.model_name

        # normalize events to a structure suitable for conversion into Steps
        normalized_events: list[dict[str, Any]] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        pending_reasoning: str | None = None

        for event in raw_events:
            etype = event.get("type")
            payload = event.get("payload", {})
            timestamp = event.get("timestamp")

            if etype != "response_item":
                if etype == "event_msg":
                    # token counts handled later for metrics
                    continue
                continue

            payload_type = payload.get("type")
            if payload_type == "reasoning":
                summary = payload.get("summary")
                if isinstance(summary, list) and summary:
                    pending_reasoning = "\n".join(
                        str(item) for item in summary if isinstance(item, str)
                    )
                else:
                    pending_reasoning = None
                continue

            if payload_type == "message":
                content = payload.get("content", [])
                text = (
                    self._extract_message_text(content)
                    if isinstance(content, list)
                    else ""
                )
                normalized_events.append(
                    {
                        "kind": "message",
                        "timestamp": timestamp,
                        "role": payload.get("role", "user"),
                        "text": text,
                        "reasoning": pending_reasoning
                        if payload.get("role") == "assistant"
                        else None,
                    }
                )
                pending_reasoning = None
                continue

            if payload_type == "web_search_call":
                action = payload.get("action") or {}
                action_type = action.get("type", "")
                arguments: dict[str, Any] = {"action_type": action_type}
                if "query" in action:
                    arguments["query"] = action["query"]
                if "queries" in action:
                    arguments["queries"] = action["queries"]
                if "url" in action:
                    arguments["url"] = action["url"]

                normalized_events.append(
                    {
                        "kind": "tool_call",
                        "timestamp": timestamp,
                        "call_id": "",
                        "tool_name": "web_search_call",
                        "arguments": arguments,
                        "raw_arguments": None,
                        "reasoning": pending_reasoning,
                        "status": payload.get("status"),
                        "message": None,
                    }
                )
                pending_reasoning = None
                continue

            if payload_type in {"function_call", "custom_tool_call"}:
                call_id = payload.get("call_id")
                if not call_id:
                    continue

                raw_args_key = (
                    "arguments" if payload_type == "function_call" else "input"
                )
                raw_arguments = payload.get(raw_args_key)
                try:
                    parsed_args = json.loads(raw_arguments)
                except Exception:  # noqa: BLE001 -- the model wrote the arguments
                    if isinstance(raw_arguments, str):
                        parsed_args = {"input": raw_arguments}
                    elif raw_arguments is None:
                        parsed_args = {}
                    else:
                        parsed_args = {"value": raw_arguments}

                pending_calls[call_id] = {
                    "kind": "tool_call",
                    "timestamp": timestamp,
                    "call_id": call_id,
                    "tool_name": payload.get("name") or "",
                    "arguments": parsed_args,
                    "raw_arguments": raw_arguments,
                    "reasoning": pending_reasoning,
                    "status": payload.get("status"),
                    "message": None,
                }
                pending_reasoning = None
                continue

            if payload_type in {"function_call_output", "custom_tool_call_output"}:
                call_id = payload.get("call_id")
                output_text, metadata = self._parse_output_blob(payload.get("output"))

                call_info = pending_calls.pop(call_id, None) if call_id else None

                if call_info is None:
                    call_info = {
                        "kind": "tool_call",
                        "timestamp": timestamp,
                        "call_id": call_id or "",
                        "tool_name": payload.get("name", "") or "",
                        "arguments": {},
                        "raw_arguments": None,
                        "reasoning": pending_reasoning,
                        "status": None,
                        "message": None,
                    }

                call_info["output"] = output_text
                call_info["metadata"] = metadata
                call_info["timestamp"] = call_info.get("timestamp") or timestamp
                normalized_events.append(call_info)
                pending_reasoning = None
                continue

        steps: list[Step] = []
        for idx, norm_event in enumerate(normalized_events, start=1):
            try:
                step = self._convert_event_to_step(norm_event, idx)
            except ValueError as exc:
                self.logger.debug(f"Skipping event during step conversion: {exc}")
                continue

            # Provide default model name if not set for agent steps
            if step.source == "agent" and not step.model_name and default_model_name:
                step.model_name = default_model_name

            steps.append(step)

        if not steps:
            self.logger.debug("No valid steps produced from Codex session")
            return None

        # Extract final metrics from the last token_count event with totals
        total_metrics: FinalMetrics | None = None
        for event in reversed(raw_events):
            if event.get("type") != "event_msg":
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue

            info = payload.get("info")
            if not isinstance(info, dict):
                continue

            total_usage = info.get("total_token_usage")
            if not isinstance(total_usage, dict):
                continue

            prompt_tokens = total_usage.get("input_tokens")
            completion_tokens = total_usage.get("output_tokens")
            reasoning_tokens = total_usage.get("reasoning_output_tokens")
            cached_tokens = total_usage.get("cached_input_tokens")
            overall_tokens = total_usage.get("total_tokens")

            # Codex CLI does not include cost in token_count events, so fall
            # back to LiteLLM's pricing table when the field is absent. Use
            # explicit `is None` checks so a legitimate $0 cost reported by
            # the CLI isn't mistaken for a missing field.
            total_cost_usd = info.get("total_cost")
            if total_cost_usd is None:
                total_cost_usd = info.get("cost_usd")
            if total_cost_usd is None:
                total_cost_usd = self._compute_cost_from_pricing(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                )

            total_metrics = FinalMetrics(
                total_prompt_tokens=prompt_tokens if prompt_tokens else None,
                total_completion_tokens=completion_tokens or None,
                total_cached_tokens=cached_tokens or None,
                total_cost_usd=total_cost_usd,
                total_steps=len(steps),
                extra={
                    "reasoning_output_tokens": reasoning_tokens,
                    "total_tokens": overall_tokens,
                    "last_token_usage": info.get("last_token_usage"),
                },
            )
            break

        trajectory = Trajectory(
            schema_version="ATIF-v1.5",
            session_id=session_id,
            agent=Agent(
                name="codex",
                version=agent_version,
                model_name=default_model_name,
                extra=agent_extra,
            ),
            steps=steps,
            final_metrics=total_metrics,
        )

        return trajectory

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the agent context after Codex finishes executing.

        Converts the Codex session JSONL file into an ATIF trajectory, persists it,
        and propagates usage metrics back to the Harbor context.
        """
        try:
            session_dir = self._get_session_dir()
        except ValueError as exc:
            # The agent's commands can make directories among its sessions.
            self.logger.warning(f"No single Codex session directory: {exc}")
            return
        if not session_dir:
            self.logger.debug("No Codex session directory found")
            return

        try:
            trajectory = self._convert_events_to_trajectory(session_dir)
        except Exception:
            self.logger.exception("Failed to convert Codex events to trajectory")
            return

        if not trajectory:
            self.logger.debug("Failed to convert Codex session to trajectory")
            return

        trajectory_path = (self.trajectory_dir or self.logs_dir) / "trajectory.json"
        try:
            with open(trajectory_path, "w") as handle:
                handle.write(format_trajectory_json(trajectory.to_json_dict()))
            self.logger.debug(f"Wrote Codex trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(
                f"Failed to write trajectory file {trajectory_path}: {exc}"
            )

        if trajectory.final_metrics:
            metrics = trajectory.final_metrics
            context.cost_usd = metrics.total_cost_usd
            context.n_input_tokens = metrics.total_prompt_tokens or 0
            context.n_cache_tokens = metrics.total_cached_tokens or 0
            context.n_output_tokens = metrics.total_completion_tokens or 0

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to Codex's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            f"$HOME/.agents/skills/ 2>/dev/null || true"
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        """Return a shell command that writes MCP config to $CODEX_HOME/config.toml."""
        if not self.mcp_servers:
            return None
        lines: list[str] = []
        for server in self.mcp_servers:
            lines.append(f"[mcp_servers.{server.name}]")
            if server.transport == "stdio":
                cmd_parts = [server.command] + server.args if server.command else []
                lines.append(f'command = "{shlex.join(cmd_parts)}"')
            else:
                lines.append(f'url = "{server.url}"')
            lines.append("")
        escaped_config = shlex.quote("\n".join(lines))
        return f'echo {escaped_config} >> "$CODEX_HOME/config.toml"'

    def _build_provider_config_command(self) -> str | None:
        """Return a shell command that appends the model provider block to
        $CODEX_HOME/config.toml, or None when no base URL is set.

        A custom endpoint must be declared as a full model PROVIDER, not via the
        legacy `openai_base_url` key. That key was enough on codex 0.118.0, but by
        0.146 the Responses transport is a websocket (`codex_api::endpoint::
        responses_websocket`) which ignores it and dials wss://api.openai.com
        regardless -- inside a sealed sandbox that surfaces as "Connection reset by
        peer" against a host the egress allowlist never admitted, which reads like a
        network fault rather than a misconfiguration.

        `requires_openai_auth = true` makes the provider use the OpenAI credential
        (auth.json / OPENAI_API_KEY). `wire_api = "responses"` is the only supported
        value and is also the default; stated explicitly so a future default change
        cannot silently switch the transport.

        The base URL is a secret like the credential, so it is never part of the
        command: the shell expands ``$OPENAI_BASE_URL`` from the exec's environment,
        as the auth file's credential is, and the command fails when that holds none.
        A command's text is logged (at debug here, by an exec server, in a failed
        command's error) where its environment is not.

        Run as its own exec with `printf`, NOT as a heredoc concatenated onto the auth
        command. The heredoc form is fragile here -- it is appended after another
        `<<EOF` block and terminated by a `TOML` line with no trailing newline -- and
        when it silently fails to land the only symptom is codex dialling
        api.openai.com and dying on the egress seal, which reads as a network fault.
        The `grep -q` makes a failed write loud instead: exec_as_agent raises on a
        nonzero exit.
        """
        if not self._get_env("OPENAI_BASE_URL"):
            return None
        head, tail = self._PROVIDER_TOML.split("{base_url}")
        # The block is printf's argument, never its format, so a % or a \ in it stays.
        return (
            f"printf '%s' {shlex.quote(head)} "
            f'"${{OPENAI_BASE_URL:?}}" {shlex.quote(tail)} '
            '>> "$CODEX_HOME/config.toml" '
            '&& grep -q "^\\[model_providers.custom\\]" "$CODEX_HOME/config.toml"'
        )

    def _resolve_auth_json_path(self) -> Path | None:
        """Resolve which auth.json to inject, if any.

        Defaults to None (OPENAI_API_KEY auth). Opt into auth.json auth via:
          - CODEX_AUTH_JSON_PATH=<path> → use that specific file
          - CODEX_FORCE_AUTH_JSON=<truthy> → use ~/.codex/auth.json
        """
        explicit = self._get_env("CODEX_AUTH_JSON_PATH")
        if explicit:
            p = Path(explicit)
            if not p.is_file():
                raise ValueError(
                    f"CODEX_AUTH_JSON_PATH points to non-existent file: {explicit}"
                )
            return p

        if parse_bool_env_value(
            self._get_env("CODEX_FORCE_AUTH_JSON"),
            name="CODEX_FORCE_AUTH_JSON",
            default=False,
        ):
            default = Path.home() / ".codex" / "auth.json"
            if not default.is_file():
                raise ValueError(
                    f"CODEX_FORCE_AUTH_JSON is set but {default} does not exist"
                )
            return default

        return None

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name:
            raise ValueError("Model name is required")

        model = self.model_name.split("/")[-1]

        # Build command with optional CLI config flags from descriptors.
        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        # Auth resolution:
        #   1. CODEX_AUTH_JSON_PATH=<path> → use that specific auth.json file
        #   2. CODEX_FORCE_AUTH_JSON=<truthy> → use ~/.codex/auth.json
        #   3. Default: use OPENAI_API_KEY
        auth_json_path = self._resolve_auth_json_path()

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (self._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()

        env: dict[str, str] = {
            "CODEX_HOME": remote_codex_home,
        }

        # $CODEX_HOME/sessions is a SYMLINK into the agent log dir, so codex writes its
        # session rollouts straight into the mount the host live-syncs -- that is what
        # makes an in-progress build followable. Kept in $CODEX_HOME (not moved) so the
        # teardown below still unlinks it; `rm -rf` removes the link, never the target.
        #
        # Without this the rollouts sit in /tmp for the whole run and are copied out
        # only at teardown, so a 12h build renders nothing until it is over -- the
        # window in which watching it is worth anything.
        sessions_dest = (EnvironmentPaths.agent_dir / "sessions").as_posix()
        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)} '
                f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())} "
                f"{shlex.quote(sessions_dest)} "
                f'&& ln -sfn {shlex.quote(sessions_dest)} "$CODEX_HOME/sessions"'
            ),
            env=env,
        )

        if auth_json_path:
            self.logger.debug("Codex auth: using auth.json from %s", auth_json_path)
            await environment.upload_file(auth_json_path, remote_auth_path)
            # upload_file copies as root; fix ownership so the agent user can read it
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=f"chown {environment.default_user} {remote_auth_path}",
                )
            setup_command = (
                f'ln -sf {shlex.quote(remote_auth_path)} "$CODEX_HOME/auth.json"\n'
            )
        else:
            self.logger.debug("Codex auth: using OPENAI_API_KEY")
            env["OPENAI_API_KEY"] = self._get_env("OPENAI_API_KEY") or ""
            setup_command = (
                f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                f"ln -sf {shlex.quote(remote_auth_path)} "
                '"$CODEX_HOME/auth.json"\n'
            )

        # The base config is the first write, and it truncates the file. A later block
        # appends, and TOML reads a bare key that follows a table header as a key of
        # that table, so top-level keys have to come before the provider and MCP
        # tables. Truncating also means a setup that runs again into the same
        # $CODEX_HOME rebuilds the same file instead of repeating keys, which codex
        # refuses to load. Read back like the other blocks, so a write that does not
        # land fails setup.
        web_search = f"web_search = {json.dumps(self._WEB_SEARCH_MODE)}"
        await self.exec_as_agent(
            environment,
            command=(
                f"printf '%s\\n' {shlex.quote(web_search)} "
                '> "$CODEX_HOME/config.toml" '
                f'&& grep -qxF {shlex.quote(web_search)} "$CODEX_HOME/config.toml"'
            ),
            env=env,
        )

        if openai_base_url := self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = openai_base_url

        provider_command = self._build_provider_config_command()
        if provider_command:
            await self.exec_as_agent(environment, command=provider_command, env=env)
            # Snapshot the EFFECTIVE config into the rollout record. Which endpoint codex
            # actually dials is otherwise invisible after the sandbox is gone, and a
            # provider block that fails to take effect looks identical to a network
            # fault ("Connection reset by peer" against a host the seal never admitted).
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p {shlex.quote(EnvironmentPaths.agent_dir.as_posix())} && "
                    f'cp "$CODEX_HOME/config.toml" '
                    f"{shlex.quote((EnvironmentPaths.agent_dir / 'codex-config.toml').as_posix())}"
                ),
                env=env,
            )

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"

        if setup_command.strip():
            await self.exec_as_agent(
                environment,
                command=setup_command,
                env=env,
            )
        # The brief goes to the agent's WORKING DIRECTORY as a file (plus a byte-identical
        # `cp` to the persisted path), never argv -- same contract as the claude-code
        # adapter. What rides the command line is only the short kickoff that names it.
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p /logs/agent {self._WORKSPACE_DIR} "
                f"&& printf '%s' {escaped_instruction} > {self._WORKSPACE_BRIEF_PATH} "
                f"&& cp {self._WORKSPACE_BRIEF_PATH} {self._BRIEF_PATH}"
            ),
            env=env,
        )

        nvm = "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
        out = EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME
        # Codex's own sandboxing buys nothing inside a container and costs correctness: its
        # bubblewrap helper cannot pivot_root there, so every shell command and apply_patch
        # fails with "bwrap: Can't bind mount /oldroot/ on /newroot/: Invalid argument".
        # Both calls therefore bypass it: bypassing on call 1 alone leaves the drive
        # sandboxed and failing. Disabling the sandbox by feature flag instead does NOT
        # work, and `use_legacy_landlock` is deprecated in 0.146.0 and injects an error
        # item into every turn.
        #
        # `--ignore-rules` drops user and project execpolicy `.rules` files on top. Codex's
        # built-in rules are compiled into the binary and stay in force regardless --
        # `rm -f` is refused by one of them.
        unsandboxed = (
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            "--ignore-rules "
        )
        # Both calls carry it; see _WEB_SEARCH_MODE for why the config key is not enough.
        no_web_search = (
            f"-c {shlex.quote(f'web_search={json.dumps(self._WEB_SEARCH_MODE)}')} "
        )
        # Without a kickoff the prompt IS the brief text.
        opening = (
            shlex.quote(self._goal_kickoff)
            if self._goal_kickoff
            else escaped_instruction
        )
        try:
            # Call 1 -- establish the goal and take a first pass. With a goal outstanding
            # codex will NOT emit task_complete; it ends the turn as turn_aborted and the
            # process exits, which is why the second call is required rather than optional.
            await self.exec_as_agent(
                environment,
                command=(
                    f"{nvm}codex exec "
                    f"{unsandboxed}"
                    f"--model {model} "
                    "--json "
                    "--enable unified_exec "
                    f"{cli_flags_arg}"
                    f"{no_web_search}"
                    "-- "  # end of flags
                    f"{opening} "
                    f"2>&1 </dev/null | tee -a {out}"
                ),
                env=env,
            )
            if self._goal_kickoff is None:
                return
            # Call 2 -- the drive. `codex exec resume` on a thread with an OUTSTANDING
            # goal runs turn after turn inside ONE process, and returns only when the
            # goal is met or the orchestrator's wall-clock cap terminates it. This is
            # not a retry loop: it is a single long-lived invocation, the codex analogue
            # of claude-code's one `claude` process driven by its /goal Stop hook. If the
            # agent judges the goal met it exits here, and that is a legitimate end.
            #
            # Resumed by the id call 1 opened, never `--last`. `--last` is the newest
            # session in the directory, and a model that spawns sub-agents forks a
            # thread per sub-agent, each newer than the one that spawned it. Resuming a
            # fork either fails outright ("direct app-server input is not allowed for
            # multi-agent v2 sub-agents") or lands in a sub-agent's environment, and
            # both cost the whole build.
            #
            # `resume` takes a slightly different flag set than `exec` -- it rejects
            # `--sandbox` -- so its command is built separately. It does accept the bypass
            # and git-check flags, and must be given them: without the bypass it falls back
            # to codex's default sandbox mode and bubblewraps every command it runs.
            # The id is the first `thread.started` in call 1's own JSON stream, which a
            # session emits before it can fork anything. Found in two stages -- the first
            # line announcing a thread start, then `thread_id` anywhere on it -- because
            # JSON field order and spacing are not guaranteed and a pattern that assumed
            # them would fail into the `--last` fallback.
            # A stream carrying none still falls back, and says so on stderr.
            # The model is passed again. A resumed thread does not keep the model it was
            # recorded with: without `--model` codex resumes on its own default and says
            # so ("recorded with model <model> but is resuming with <default>").
            thread_id = (
                f"tid=$(grep -m1 'thread\\.started' {out} "
                '| grep -o \'"thread_id"[[:space:]]*:[[:space:]]*"[^"]*"\' '
                "| head -1 | cut -d'\"' -f4); "
                '[ -n "$tid" ] || echo \'harbor: no thread.started in the kickoff stream; '
                "resuming --last, which may select a sub-agent fork' >&2; "
            )
            await self.exec_as_agent(
                environment,
                command=(
                    f"{nvm}{thread_id}"
                    'codex exec resume "${tid:---last}" '
                    f"--model {model} "
                    "--json "
                    f"{unsandboxed}"
                    f"{cli_flags_arg}"
                    f"{no_web_search}"
                    "-- "
                    f"{shlex.quote(self._RESUME_PROMPT)} "
                    f"2>&1 </dev/null | tee -a {out}"
                ),
                env=env,
            )
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    # sessions are written through the symlink directly into the
                    # agent dir, so there is nothing to copy -- just make sure the
                    # directory exists for a run that produced no session at all.
                    command=f"mkdir -p {shlex.quote(sessions_dest)}",
                    env=env,
                )
            except Exception:
                pass
            # cleanup - best effort
            try:
                await self.exec_as_agent(
                    environment,
                    command=f'rm -rf {shlex.quote(remote_secrets_dir)} "$CODEX_HOME"',
                    env=env,
                )
            except Exception:
                pass
