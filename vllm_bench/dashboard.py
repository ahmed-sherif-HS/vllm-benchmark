"""Local dashboard for benchmark runs.

The browser only receives this public view. API keys, URL userinfo, and
query strings are dropped before a response is written.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_PART = re.compile(r"^[A-Za-z0-9._-]+$")
_LOCALHOST = {"127.0.0.1", "localhost", "::1"}


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def public_endpoint(url: str | None) -> str:
    """Scheme, host, and path. Userinfo and query strings are removed."""
    if not url:
        return ""
    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        return ""
    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path or "", "", ""))


def _device(raw: dict[str, Any]) -> str | None:
    server = raw.get("server") or {}
    if server.get("device") in ("gpu", "cpu"):
        return server["device"]
    cfg = raw.get("config") or {}
    if cfg.get("device") in ("gpu", "cpu"):
        return cfg["device"]
    label = str(raw.get("label") or "")
    if label.endswith("-gpu"):
        return "gpu"
    if label.endswith("-cpu"):
        return "cpu"
    return None


def _ms(block: Any, key: str) -> int | None:
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if value is None:
        return None
    return round(float(value) * 1000)


def _round(value: Any, places: int) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


def _cell(summary: dict[str, Any]) -> dict[str, Any]:
    prefill = summary.get("prefill_tok_per_s")
    prefill_p50 = prefill.get("p50") if isinstance(prefill, dict) else None
    return {
        "concurrency": summary.get("concurrency"),
        "input_tokens": summary.get("target_input_tokens"),
        "output_tokens": summary.get("target_output_tokens"),
        "ok": summary.get("ok") or 0,
        "requests": summary.get("requests") or 0,
        "failed": summary.get("failed") or 0,
        "ttft_p50_ms": _ms(summary.get("ttft"), "p50"),
        "ttft_p99_ms": _ms(summary.get("ttft"), "p99"),
        "tpot_p50_ms": _ms(summary.get("tpot"), "p50"),
        "e2e_p50_ms": _ms(summary.get("e2e"), "p50"),
        "e2e_p99_ms": _ms(summary.get("e2e"), "p99"),
        "itl_p99_ms": _ms(summary.get("itl"), "p99"),
        "output_tok_per_s": _round(summary.get("output_tok_per_s"), 1),
        "req_per_s": _round(summary.get("req_per_s"), 2),
        "prefill_tok_per_s_p50": _round(prefill_p50, 1),
        "goodput": _round(summary.get("goodput"), 4),
        "fail_timeout": summary.get("fail_timeout") or 0,
        "fail_context_length": summary.get("fail_context_length") or 0,
        "fail_oom": summary.get("fail_oom") or 0,
        "fail_other": summary.get("fail_other") or 0,
    }


def first_break(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First cell in sweep order whose goodput is under 1."""
    for cell in cells:
        goodput = cell.get("goodput")
        if goodput is not None and goodput < 1:
            return {
                "concurrency": cell["concurrency"],
                "input_tokens": cell["input_tokens"],
                "output_tokens": cell["output_tokens"],
                "goodput": cell["goodput"],
                "ttft_p50_ms": cell["ttft_p50_ms"],
                "ttft_p99_ms": cell["ttft_p99_ms"],
                "tpot_p50_ms": cell["tpot_p50_ms"],
            }
    return None


def to_public(raw: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Fields the dashboard is allowed to show."""
    cfg = raw.get("config") or {}
    server = raw.get("server") or {}
    endpoint = cfg.get("endpoint") or {}
    bars = cfg.get("bars") or {}
    cells = [_cell(s) for s in raw.get("summaries") or [] if isinstance(s, dict)]
    started = raw.get("started_at")
    finished = raw.get("finished_at")
    duration = None
    if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
        duration = round(finished - started)
    peak = None
    for cell in cells:
        rate = cell.get("output_tok_per_s")
        if rate is None:
            continue
        if peak is None or rate > peak["output_tok_per_s"]:
            peak = {
                "output_tok_per_s": rate,
                "concurrency": cell["concurrency"],
                "input_tokens": cell["input_tokens"],
                "output_tokens": cell["output_tokens"],
            }
    host = raw.get("host") or {}
    max_len = server.get("max_model_len")
    if max_len is None:
        max_len = cfg.get("max_model_len")
    return {
        "id": run_id,
        "label": raw.get("label") or run_id,
        "model": server.get("model") or raw.get("model") or endpoint.get("model") or "",
        "device": _device(raw),
        "dtype": server.get("dtype") or cfg.get("dtype") or None,
        "max_model_len": max_len,
        "launch_args": server.get("launch_args") or cfg.get("launch_args") or "",
        "endpoint": public_endpoint(endpoint.get("base_url")),
        "started_at": started if isinstance(started, (int, float)) else None,
        "duration_s": duration,
        "bars": {"ttft_ms": bars.get("ttft_ms"), "tpot_ms": bars.get("tpot_ms")},
        "client_cpus": host.get("cpu_count"),
        "cells": cells,
        "requests_ok": sum(c["ok"] for c in cells),
        "requests": sum(c["requests"] for c in cells),
        "break": first_break(cells),
        "peak": peak,
    }


def _brief(pub: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "id", "label", "model", "device", "dtype", "max_model_len",
        "started_at", "duration_s", "bars", "requests_ok", "requests", "break", "peak",
    )
    return {key: pub[key] for key in keep}


def iter_summaries(roots: list[Path]):
    """One summary per directory name. Earlier roots win, so results/ shadows examples/."""
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir() or not _PART.match(root.name):
            continue
        for path in sorted(root.glob("*/summary.json")):
            name = path.parent.name
            if name in seen or not _PART.match(name):
                continue
            seen.add(name)
            yield root, path, f"{root.name}/{name}"


def list_runs(roots: list[Path]) -> list[dict[str, Any]]:
    items = []
    for _root, path, run_id in iter_summaries(roots):
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        items.append(_brief(to_public(raw, run_id)))
    items.sort(key=lambda item: item.get("started_at") or 0, reverse=True)
    return items


def load_run(roots: list[Path], root_name: str, name: str) -> dict[str, Any] | None:
    if not _PART.match(root_name) or not _PART.match(name):
        return None
    for root, path, run_id in iter_summaries(roots):
        if root.name == root_name and path.parent.name == name:
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                return None
            return to_public(raw, run_id)
    return None


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], roots: list[Path]):
        self.roots = roots
        self.static = static_dir()
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._static("index.html", "text/html; charset=utf-8")
            return
        if path == "/app.css":
            self._static("app.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._static("app.js", "text/javascript; charset=utf-8")
            return
        if path == "/api/runs":
            self._json(list_runs(self.server.roots))
            return
        parts = [part for part in path.split("/") if part]
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs":
            run = load_run(self.server.roots, parts[2], parts[3])
            if run is None:
                self.send_error(404)
                return
            self._json(run)
            return
        self.send_error(404)

    def _static(self, name: str, content_type: str) -> None:
        file = self.server.static / name
        if not file.is_file():
            self.send_error(404)
            return
        body = file.read_bytes()
        self._send(200, content_type, body)

    def _json(self, payload: Any) -> None:
        self._send(200, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def serve(roots: list[Path], host: str = "127.0.0.1", port: int = 8765) -> None:
    if host not in _LOCALHOST:
        raise ValueError("the dashboard binds to localhost only")
    httpd = DashboardServer((host, port), roots)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return
    finally:
        httpd.server_close()
