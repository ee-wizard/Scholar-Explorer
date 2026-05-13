"""LLM backend factory for SkillFlow.

Set environment variable ``SKILL_FLOW_LLM_BACKEND`` to choose the backend:
  - ``openai``  (default): uses openai.OpenAI() — needs OPENAI_API_KEY
  - ``copilot``           : uses GitHub Copilot SDK — needs GitHub Copilot subscription
                            Auth via COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN
                            or logged-in ``gh`` / ``copilot`` CLI user.
"""

from __future__ import annotations

import os


def get_llm_client() -> object:
    """Return a chat-completions client based on SKILL_FLOW_LLM_BACKEND."""
    backend = os.environ.get("SKILL_FLOW_LLM_BACKEND", "openai").lower()

    if backend == "copilot":
        from skill_flow.llm.copilot_backend import CopilotClient

        return CopilotClient()

    # Default: standard OpenAI client (also works with any OPENAI_BASE_URL override)
    from openai import OpenAI

    return OpenAI()
