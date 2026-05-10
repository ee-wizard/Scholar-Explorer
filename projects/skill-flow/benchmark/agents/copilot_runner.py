"""Run a Harbor agent prompt through the GitHub Copilot SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from copilot import CopilotClient, ExternalServerConfig
from copilot.session import PermissionHandler


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--copilot-home")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--mcp-servers-json")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    return parser.parse_args()


def _build_client(args: argparse.Namespace) -> CopilotClient:
    cli_url = os.environ.get("COPILOT_CLI_URL")
    if cli_url:
        return CopilotClient(ExternalServerConfig(url=cli_url))

    if args.copilot_home:
        os.environ["COPILOT_HOME"] = args.copilot_home

    return CopilotClient()


async def _run(args: argparse.Namespace) -> int:
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mcp_servers = json.loads(args.mcp_servers_json) if args.mcp_servers_json else None
    payload: dict[str, Any] = {
        "model": args.model,
        "copilot_home": args.copilot_home,
        "mcp_servers": mcp_servers,
    }

    client = _build_client(args)

    try:
        await client.start()

        try:
            session_kwargs: dict[str, Any] = {
                "model": args.model,
                "on_permission_request": PermissionHandler.approve_all,
                "streaming": False,
                "infinite_sessions": {"enabled": False},
            }
            if args.reasoning_effort:
                session_kwargs["reasoning_effort"] = args.reasoning_effort
            if mcp_servers:
                session_kwargs["mcp_servers"] = mcp_servers

            try:
                session = await client.create_session(**session_kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    "reasoning effort" in msg
                    and "does not support" in msg
                    and "reasoning_effort" in session_kwargs
                ):
                    session_kwargs.pop("reasoning_effort", None)
                    payload["reasoning_effort_fallback"] = True
                    session = await client.create_session(**session_kwargs)
                else:
                    raise
            try:
                response = await session.send_and_wait(
                    args.prompt,
                    timeout=args.timeout_sec,
                )
                response_data = getattr(response, "data", None)
                content = getattr(response_data, "content", None)
                if isinstance(content, str):
                    print(content)

                payload.update(
                    {
                        "response": content,
                        "response_event": _to_jsonable(response),
                        "response_data": _to_jsonable(response_data),
                        "session_id": getattr(session, "session_id", None),
                        "workspace_path": getattr(session, "workspace_path", None),
                        "capabilities": _to_jsonable(getattr(session, "capabilities", None)),
                    }
                )
            finally:
                await session.disconnect()
        finally:
            await client.stop()
    except Exception as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        output_path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2))
        raise

    output_path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())