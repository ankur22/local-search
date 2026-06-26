#!/usr/bin/env python3
"""MCP 'toolbox' server for the k6 demo agent (real MCP protocol via FastMCP).

Exposes two cross-boundary tools the local search agent can call as an MCP
*client*:

  - web_fetch(url)        : fetch text from a public http(s) URL (SSRF-guarded)
  - grep_repo(pattern,..) : regex search over the demo docs (safe, in-process)

Run it over streamable-http:

    MCP_PORT=8200 python mcp_toolbox.py
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse
from urllib.request import urlopen

from mcp.server.fastmcp import FastMCP

HOST = os.environ.get("MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("MCP_PORT", "8200"))

# Roots grep_repo is allowed to search (colon-separated). Defaults to ./demo_docs.
_DEFAULT_ROOT = os.path.join(os.getcwd(), "demo_docs")
ALLOWED_ROOTS = [os.path.realpath(os.path.expanduser(p)) for p in
                 os.environ.get("MCP_ROOTS", _DEFAULT_ROOT).split(":") if p.strip()]

mcp = FastMCP("toolbox", host=HOST, port=PORT, stateless_http=True)


def _host_is_blocked(host: str) -> bool:
    """True if the host resolves to a private/loopback/link-local/metadata address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True  # unresolvable -> block
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return True
        if str(addr) == "169.254.169.254":  # cloud metadata
            return True
    return False


@mcp.tool()
def web_fetch(url: str) -> str:
    """Fetch the text content at a public http(s) URL. Blocks internal/metadata hosts."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"ERROR: blocked scheme '{parsed.scheme}' (only http/https allowed)"
    host = parsed.hostname or ""
    if not host or _host_is_blocked(host):
        return f"ERROR: blocked host '{host}' (SSRF protection)"
    try:
        with urlopen(url, timeout=8) as resp:  # noqa: S310 (guarded above)
            body = resp.read(200_000).decode("utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", body)[:4000]
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: fetch failed: {exc}"


@mcp.tool()
def grep_repo(pattern: str, path: str = "") -> str:
    """Search the demo docs for a regex pattern. Returns up to 25 'file:line: text' hits."""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"ERROR: bad regex: {exc}"

    search_roots = ALLOWED_ROOTS
    if path:
        target = os.path.realpath(os.path.expanduser(path))
        if not any(target == r or target.startswith(r + os.sep) for r in ALLOWED_ROOTS):
            return f"ERROR: path '{path}' is outside the allowed roots (path-traversal protection)"
        search_roots = [target]

    hits: list[str] = []
    for root in search_roots:
        walk = [root] if os.path.isfile(root) else _walk_files(root)
        for fpath in walk:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(f"{os.path.basename(fpath)}:{lineno}: {line.strip()[:200]}")
                            if len(hits) >= 25:
                                return "\n".join(hits)
            except Exception:
                continue
    return "\n".join(hits) if hits else "(no matches)"


def _walk_files(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            yield os.path.join(dirpath, name)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
