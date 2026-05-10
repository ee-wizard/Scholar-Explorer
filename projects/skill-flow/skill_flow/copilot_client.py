"""GitHub Copilot API client — OpenAI-compatible via GitHub Models endpoint.

Reads authentication from the GitHub CLI (``gh auth token``) or from the
Copilot ``hosts.json`` file written by VS Code / GitHub Copilot extension.
Uses the GitHub Models inference endpoint which is OpenAI SDK-compatible.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# GitHub Models inference endpoint (OpenAI-compatible)
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

# Copilot hosts.json location on Windows
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")
_COPILOT_HOSTS_PATH = Path(_LOCALAPPDATA) / "GitHub Copilot" / "hosts.json"


def _token_from_gh_cli() -> str | None:
    """Try to obtain a GitHub token via the ``gh`` CLI."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        token = result.stdout.strip()
        if token:
            logger.debug("GitHub token obtained from gh CLI")
            return token
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _token_from_copilot_hosts() -> str | None:
    """Read the OAuth token from the Copilot hosts.json config file."""
    if not _COPILOT_HOSTS_PATH.exists():
        return None
    try:
        data = json.loads(_COPILOT_HOSTS_PATH.read_text(encoding="utf-8"))
        token: str | None = data.get("github.com", {}).get("oauth_token")
        if token:
            logger.debug("GitHub token obtained from Copilot hosts.json")
        return token
    except (json.JSONDecodeError, OSError, AttributeError):
        return None


def _resolve_token() -> str:
    """Resolve a GitHub token from available sources.

    Priority:
    1. ``GITHUB_TOKEN`` environment variable
    2. ``gh auth token`` CLI
    3. Copilot ``hosts.json``

    Raises ``RuntimeError`` if no token is found.
    """
    # 1. Env var (allows CI / explicit override)
    env_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if env_token:
        logger.debug("GitHub token obtained from GITHUB_TOKEN env var")
        return env_token

    # 2. gh CLI
    token = _token_from_gh_cli()
    if token:
        return token

    # 3. Copilot hosts.json
    token = _token_from_copilot_hosts()
    if token:
        return token

    raise RuntimeError(
        "No GitHub token found. Provide one via:\n"
        "  • GITHUB_TOKEN environment variable\n"
        "  • GitHub CLI: gh auth login\n"
        "  • GitHub Copilot extension (VS Code)"
    )


def create_copilot_client():
    """Return an OpenAI client configured for GitHub Models / Copilot API.

    The returned client is a drop-in replacement for ``openai.OpenAI()``
    and supports the same chat completions interface.  Model names from the
    paper (e.g. ``gpt-4o-mini``) are available on the GitHub Models endpoint.
    """
    from openai import OpenAI  # noqa: PLC0415 (deferred import — optional dep)

    token = _resolve_token()
    return OpenAI(
        base_url=GITHUB_MODELS_BASE_URL,
        api_key=token,
    )
