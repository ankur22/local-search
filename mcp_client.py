"""Minimal MCP client wrapper for the demo agent.

Connects to the toolbox MCP server (streamable-http), discovers its tools at
runtime (`list_tools`), and invokes them (`call_tool`). The async MCP calls are
wrapped in ``asyncio.run`` so they can be used from the synchronous Flask agent.

The server is stateless-http, so a fresh session per call is fine for the demo.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8200/mcp")


def _content_to_text(result: Any) -> str:
    """Flattens a CallToolResult's content blocks into plain text."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


async def _alist_tools() -> list[dict[str, Any]]:
    async with streamablehttp_client(MCP_URL) as (read, write, *_rest):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "schema": t.inputSchema or {"type": "object", "properties": {}},
                }
                for t in resp.tools
            ]


async def _acall_tool(name: str, arguments: dict[str, Any]) -> str:
    async with streamablehttp_client(MCP_URL) as (read, write, *_rest):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return _content_to_text(result)


def list_tools() -> list[dict[str, Any]]:
    """Returns discovered MCP tools: [{name, description, schema}, ...]."""
    return asyncio.run(_alist_tools())


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Calls an MCP tool and returns its text result."""
    return asyncio.run(_acall_tool(name, arguments))
