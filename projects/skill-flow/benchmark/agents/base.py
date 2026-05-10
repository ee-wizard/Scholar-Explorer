"""
Shared base class and utilities for evaluation agents.
"""

import json
import os
import shlex
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

# Separator used in trial directory names: "task-name__hash"
TASK_NAME_SEPARATOR = "__"


@dataclass(frozen=True)
class McpServer:
    """MCP server configuration for Codex agents."""

    name: str
    url: str


def get_project_root() -> Path:
    """Get the project root directory (skill-flow/)."""
    return Path(__file__).parent.parent.parent


def _pick_env(*names: str) -> str | None:
    """Return the first non-empty environment value from candidate names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _resolve_model_service_auth() -> tuple[str | None, str | None]:
    """Resolve API key and base URL for Codex model access.

    Priority:
    1) Explicit OpenAI-compatible vars (OPENAI_API_KEY / OPENAI_BASE_URL)
    2) GitHub/Copilot vars (GITHUB_TOKEN/GH_TOKEN + GITHUB_MODELS_ENDPOINT)
    """
    api_key = _pick_env(
        "OPENAI_API_KEY",
        "GITHUB_COPILOT_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    )

    base_url = _pick_env(
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "COPILOT_OPENAI_BASE_URL",
        "GITHUB_MODELS_ENDPOINT",
    )

    # When using GitHub token without explicit endpoint, default to GitHub Models.
    if not base_url and _pick_env("GITHUB_COPILOT_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        base_url = "https://models.inference.ai.azure.com"

    return api_key, base_url


def _split_model_name(model_name: str) -> tuple[str, str]:
    """Split a model name into provider/model components."""
    model_parts = model_name.split("/", 1)
    provider = model_parts[0].lower() if len(model_parts) > 1 else ""
    model = model_parts[-1]
    return provider, model


class BaseCodexAgent(Codex):
    """
    Shared base class for evaluation Codex agents.

    Provides common utilities for template rendering and AGENTS.md upload.
    """

    _COPILOT_OUTPUT_FILENAME = "copilot_result.json"
    _COPILOT_HOME_ARCHIVE_FILENAME = "copilot-home.tar.gz"
    _COPILOT_RUNNER_FILENAME = "copilot_runner.py"

    def __init__(
        self,
        *args: Any,
        reasoning_effort: str | None = None,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort
        self._mcp_servers = mcp_servers or []

    @property
    def _uses_copilot_sdk(self) -> bool:
        """Whether this agent should execute through Copilot SDK."""
        return bool(self.model_name) and _split_model_name(self.model_name)[0] == "github"

    @property
    def _copilot_runner_source_path(self) -> Path:
        return get_project_root() / "benchmark" / "agents" / self._COPILOT_RUNNER_FILENAME

    @property
    def _copilot_install_template_path(self) -> Path:
        return get_project_root() / "benchmark" / "agents" / "install-copilot-sdk.sh.j2"

    @property
    def _copilot_container_home(self) -> Path:
        return EnvironmentPaths.agent_dir / "copilot-home"

    @property
    def _copilot_result_path(self) -> Path:
        return self.logs_dir / self._COPILOT_OUTPUT_FILENAME

    @property
    def _install_agent_template_path(self) -> Path:
        if self._uses_copilot_sdk:
            return self._copilot_install_template_path
        return super()._install_agent_template_path

    def _get_host_copilot_home(self) -> Path | None:
        """Resolve the host Copilot home to upload into the Harbor container."""
        candidates: list[Path] = []

        copilot_home = _pick_env("COPILOT_HOME")
        if copilot_home:
            candidates.append(Path(copilot_home).expanduser())

        candidates.append(Path.home() / ".copilot")

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    async def _upload_copilot_runner(self, environment: BaseEnvironment) -> None:
        """Upload the Copilot SDK runner into the Harbor container."""
        await environment.upload_file(
            source_path=self._copilot_runner_source_path,
            target_path=f"/installed-agent/{self._COPILOT_RUNNER_FILENAME}",
        )

    async def _upload_copilot_home(self, environment: BaseEnvironment) -> None:
        """Upload host Copilot state so the SDK can use logged-in user auth."""
        container_home = shlex.quote(self._copilot_container_home.as_posix())
        result = await environment.exec(command=f"mkdir -p {container_home}")
        if result.return_code != 0:
            raise RuntimeError("Failed to create Copilot home directory in environment")

        host_copilot_home = self._get_host_copilot_home()
        if host_copilot_home is None:
            print("No host Copilot home found; relying on external Copilot CLI if configured")
            return

        archive_path = self.logs_dir / self._COPILOT_HOME_ARCHIVE_FILENAME
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(host_copilot_home, arcname=".")

        archive_target = f"/installed-agent/{self._COPILOT_HOME_ARCHIVE_FILENAME}"
        await environment.upload_file(source_path=archive_path, target_path=archive_target)

        extract_result = await environment.exec(
            command=(
                f"mkdir -p {container_home} && "
                f"tar -xzf {shlex.quote(archive_target)} -C {container_home}"
            )
        )
        if extract_result.return_code != 0:
            raise RuntimeError("Failed to extract Copilot home archive in environment")

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)

        if not self._uses_copilot_sdk:
            return

        await self._upload_copilot_runner(environment)
        await self._upload_copilot_home(environment)

    def _build_codex_command(self, model: str, escaped_instruction: str) -> str:
        """Build the codex exec command string."""
        parts = [
            "codex exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            f"--model {model}",
        ]

        if self.reasoning_effort:
            parts.append(f'--config model_reasoning_effort="{self.reasoning_effort}"')

        parts.extend(
            [
                "--json",
                "--",
                escaped_instruction,
                "2>&1 </dev/null | tee",
                str(EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME),
                "&& rm -rf $CODEX_HOME/auth.json",
            ]
        )

        return " ".join(parts)

    def _build_mcp_registration_commands(self) -> str:
        """Build shell commands to register MCP servers with Codex."""
        lines: list[str] = []
        for server in self._mcp_servers:
            lines.append(f"codex mcp add {server.name} --url {server.url}")
        return "\n".join(lines)

    def _build_copilot_mcp_servers(self) -> dict[str, dict[str, Any]]:
        """Build MCP server config expected by the Copilot SDK."""
        servers: dict[str, dict[str, Any]] = {}
        for server in self._mcp_servers:
            servers[server.name] = {
                "type": "http",
                "url": server.url,
                "tools": ["*"],
            }
        return servers

    def _build_copilot_command(self, model: str, escaped_instruction: str) -> str:
        """Build the Copilot SDK runner command string."""
        parts = [
            "set -o pipefail &&",
            "/opt/copilot-sdk-venv/bin/python",
            f"/installed-agent/{self._COPILOT_RUNNER_FILENAME}",
            f"--model {shlex.quote(model)}",
            f"--output-file {shlex.quote((EnvironmentPaths.agent_dir / self._COPILOT_OUTPUT_FILENAME).as_posix())}",
            f"--copilot-home {shlex.quote(self._copilot_container_home.as_posix())}",
            f"--prompt {escaped_instruction}",
        ]

        if self.reasoning_effort:
            parts.append(f"--reasoning-effort {shlex.quote(self.reasoning_effort)}")

        mcp_servers = self._build_copilot_mcp_servers()
        if mcp_servers:
            parts.append(f"--mcp-servers-json {shlex.quote(json.dumps(mcp_servers))}")

        parts.extend(
            [
                "2>&1 | tee",
                shlex.quote(str(EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME)),
            ]
        )

        return " ".join(parts)

    def _create_copilot_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """Create commands to run the agent through the Copilot SDK."""
        if not self.model_name:
            raise ValueError("Model name is required")

        _, model = _split_model_name(self.model_name)
        escaped_instruction = shlex.quote(instruction)

        env = {
            "COPILOT_HOME": self._copilot_container_home.as_posix(),
            "PYTHONUNBUFFERED": "1",
        }

        copilot_cli_url = _pick_env("COPILOT_CLI_URL")
        if copilot_cli_url:
            env["COPILOT_CLI_URL"] = copilot_cli_url

        return [
            ExecInput(
                command=self._build_copilot_command(model, escaped_instruction),
                env=env,
            )
        ]

    def _populate_copilot_context_post_run(self, context: AgentContext) -> None:
        """Populate Harbor context from Copilot runner output when available."""
        if not self._copilot_result_path.exists():
            print(f"No Copilot runner output found at {self._copilot_result_path}")
            return

        try:
            payload = json.loads(self._copilot_result_path.read_text())
        except json.JSONDecodeError:
            self.logger.exception("Failed to parse Copilot runner output")
            return

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        cache_tokens = usage.get("cached_tokens") or usage.get("cache_tokens") or 0
        total_cost = usage.get("total_cost_usd") or usage.get("cost_usd")

        context.n_input_tokens = int(prompt_tokens)
        context.n_output_tokens = int(completion_tokens)
        context.n_cache_tokens = int(cache_tokens)
        if total_cost is not None:
            context.cost_usd = float(total_cost)

    def populate_context_post_run(self, context: AgentContext) -> None:
        if self._uses_copilot_sdk:
            self._populate_copilot_context_post_run(context)
            return
        super().populate_context_post_run(context)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """
        Create commands to run the Codex agent.

        Optionally includes reasoning_effort config if specified.
        """
        if not self.model_name:
            raise ValueError("Model name is required")

        if self._uses_copilot_sdk:
            return self._create_copilot_run_agent_commands(instruction)

        escaped_instruction = shlex.quote(instruction)

        provider, model = _split_model_name(self.model_name)

        api_key, base_url = _resolve_model_service_auth()

        if provider == "github":
            github_token = _pick_env("GITHUB_COPILOT_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
            if not github_token:
                msg = (
                    "Model provider 'github' requires one of GITHUB_COPILOT_TOKEN, "
                    "GITHUB_TOKEN, or GH_TOKEN."
                )
                raise ValueError(msg)
            api_key = github_token
            if not base_url:
                base_url = "https://models.inference.ai.azure.com"

        if not api_key:
            msg = (
                "No model service token found. Set one of OPENAI_API_KEY, "
                "GITHUB_COPILOT_TOKEN, GITHUB_TOKEN, or GH_TOKEN."
            )
            raise ValueError(msg)

        env = {
            "OPENAI_API_KEY": api_key,
            "CODEX_HOME": (EnvironmentPaths.agent_dir).as_posix(),
        }

        if base_url:
            # Keep both names for compatibility with OpenAI-compatible clients.
            env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_BASE"] = base_url

        setup_command = """
cat >"$CODEX_HOME/auth.json" <<EOF
{
  "OPENAI_API_KEY": "${OPENAI_API_KEY}",
  "OPENAI_BASE_URL": "${OPENAI_BASE_URL}"
}
EOF
                """

        if self._mcp_servers:
            setup_command += "\n" + self._build_mcp_registration_commands()

        return [
            ExecInput(command=setup_command, env=env),
            ExecInput(
                command=self._build_codex_command(model, escaped_instruction),
                env=env,
            ),
        ]
