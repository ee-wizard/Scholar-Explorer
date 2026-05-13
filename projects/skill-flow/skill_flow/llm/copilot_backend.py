"""Synchronous wrapper around GitHub Copilot SDK for simple LLM completions.

The GitHub Copilot SDK is async/agent-oriented (JSON-RPC to Copilot CLI).
This module bridges it to the synchronous chat.completions.create() interface
used by QueryGenerator and Selector.

Authentication (in priority order):
  1. COPILOT_GITHUB_TOKEN env var (PAT with Copilot access)
  2. GH_TOKEN / GITHUB_TOKEN env var
  3. Logged-in `gh` / `copilot` CLI user (if available)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fake response objects matching the openai SDK shape used by callers
# ---------------------------------------------------------------------------


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _CompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    """Mimic openai.resources.chat.completions.Completions."""

    def __init__(self, backend: CopilotCompletionBackend) -> None:
        self._backend = backend

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 200,
        max_completion_tokens: int | None = None,
        temperature: float = 0.0,
        **_kwargs: Any,
    ) -> _CompletionResponse:
        effective_max = max_completion_tokens or max_tokens
        result = self._backend.complete(
            model=model,
            messages=messages,
            max_tokens=effective_max,
        )
        return _CompletionResponse(result)


class _Chat:
    def __init__(self, backend: CopilotCompletionBackend) -> None:
        self.completions = _Completions(backend)


class CopilotClient:
    """Drop-in replacement for ``openai.OpenAI()`` using GitHub Copilot SDK."""

    def __init__(self) -> None:
        self._backend = CopilotCompletionBackend()
        self.chat = _Chat(self._backend)


# ---------------------------------------------------------------------------
# The actual async-to-sync bridge
# ---------------------------------------------------------------------------


class CopilotCompletionBackend:
    """Manages a background event loop and a persistent Copilot CLI client.

    Thread-safe: multiple threads may call ``complete()`` concurrently.
    Each completion call opens a new lightweight session (no tools, no
    infinite context) and returns the first AssistantMessage content.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="copilot-backend",
            daemon=True,
        )
        self._thread.start()
        self._copilot_client: Any = None  # copilot.CopilotClient
        self._init_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Sync public API
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 200,
    ) -> str:
        """Blocking completion call.  Returns the assistant message text."""
        self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(
            self._async_complete(model=model, messages=messages, max_tokens=max_tokens),
            self._loop,
        )
        return future.result(timeout=120)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._init_lock:
            if self._started:
                return
            future = asyncio.run_coroutine_threadsafe(
                self._async_start(), self._loop
            )
            future.result(timeout=60)
            self._started = True

    async def _async_start(self) -> None:
        from copilot import CopilotClient as _CopilotClient, SubprocessConfig  # type: ignore[import-untyped]

        github_token = (
            os.environ.get("COPILOT_GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
        )

        # If a token is set use it; otherwise fall back to the stored OAuth
        # token from `copilot login` (use_logged_in_user=True).
        self._copilot_client = _CopilotClient(
            SubprocessConfig(
                github_token=github_token or None,
                use_logged_in_user=True if not github_token else None,
            ),
        )
        await self._copilot_client.start()
        logger.info("GitHub Copilot CLI client started")

    async def _async_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        from copilot.generated.session_events import (  # type: ignore[import-untyped]
            AssistantMessageData,
            SessionIdleData,
        )
        from copilot.session import PermissionHandler  # type: ignore[import-untyped]

        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), None
        )
        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )

        response_content: list[str] = []
        done = asyncio.Event()

        session_kwargs: dict[str, Any] = {
            "on_permission_request": PermissionHandler.approve_all,
            "model": model,
            "tools": [],  # No tools — pure completions mode
            "infinite_sessions": {"enabled": False},
        }
        if system_msg:
            session_kwargs["system_message"] = {"text": system_msg}

        async with await self._copilot_client.create_session(
            **session_kwargs
        ) as session:
            def on_event(event: Any) -> None:
                match event.data:
                    case AssistantMessageData() as data:
                        response_content.append(data.content)
                    case SessionIdleData():
                        done.set()

            session.on(on_event)
            await session.send(user_msg)
            await asyncio.wait_for(done.wait(), timeout=90)

        return response_content[0] if response_content else ""

    def __del__(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
