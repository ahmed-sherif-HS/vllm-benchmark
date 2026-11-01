"""Async streaming client for OpenAI-compatible /v1/chat/completions.

Measures per request:
  ttft_s      time to first generated token (includes queueing + prefill)
  itl_s       list of inter-token latencies between streamed chunks
  e2e_s       request start -> final chunk received
  prompt_tokens / completion_tokens  from `usage` if the server sends it,
              otherwise completion_tokens falls back to chunk count.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import EndpointConfig


@dataclass
class RequestResult:
    ok: bool
    start_ts: float
    e2e_s: float
    ttft_s: float | None
    itl_s: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    chunks: int = 0
    usage_from_server: bool = False
    error: str | None = None
    status_code: int | None = None

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first."""
        if not self.itl_s:
            return None
        return sum(self.itl_s) / len(self.itl_s)

    @property
    def output_tok_per_s(self) -> float | None:
        if not self.completion_tokens or self.e2e_s <= 0:
            return None
        return self.completion_tokens / self.e2e_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "start_ts": self.start_ts,
            "e2e_s": self.e2e_s,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "itl_s": self.itl_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "chunks": self.chunks,
            "usage_from_server": self.usage_from_server,
            "error": self.error,
            "status_code": self.status_code,
        }


def _image_part(path: str, detail: str) -> dict[str, Any]:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail},
    }


class OpenAIClient:
    def __init__(self, cfg: EndpointConfig, model: str):
        self.cfg = cfg
        self.model = model
        self._image_part = (
            _image_part(cfg.image_path, cfg.image_detail) if cfg.image_path else None
        )
        limits = httpx.Limits(max_connections=4096, max_keepalive_connections=4096)
        self._http = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_s, connect=30.0),
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            limits=limits,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @staticmethod
    async def list_models(cfg: EndpointConfig) -> list[str]:
        async with httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=30.0,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        ) as http:
            r = await http.get("/models")
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]

    def _messages(self, prompt: str) -> list[dict[str, Any]]:
        if self._image_part is None:
            return [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": [{"type": "text", "text": prompt}, self._image_part]}]

    def _body(self, prompt: str, max_tokens: int, stream: bool, temperature: float) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        body.update(self.cfg.extra_body or {})
        if stream and body.get("ignore_eos"):
            # make the output length exact where the server supports it
            body.setdefault("min_tokens", max_tokens)
        return body

    async def count_prompt_tokens(self, prompt: str) -> int:
        """Non-streaming probe with max_tokens=1; returns usage.prompt_tokens."""
        body = self._body(prompt, max_tokens=1, stream=False, temperature=0.0)
        body.pop("ignore_eos", None)
        body.pop("min_tokens", None)
        r = await self._http.post("/chat/completions", json=body)
        r.raise_for_status()
        usage = r.json().get("usage") or {}
        if "prompt_tokens" not in usage:
            raise RuntimeError("Server did not return usage.prompt_tokens; cannot calibrate.")
        return int(usage["prompt_tokens"])

    async def stream_chat(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> RequestResult:
        body = self._body(prompt, max_tokens, stream=True, temperature=temperature)
        start = time.perf_counter()
        start_ts = time.time()
        ttft: float | None = None
        last = start
        itl: list[float] = []
        chunks = 0
        prompt_tokens = completion_tokens = None
        usage_seen = False
        try:
            async with self._http.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode(errors="replace")[:500]
                    return RequestResult(
                        ok=False, start_ts=start_ts, e2e_s=time.perf_counter() - start,
                        ttft_s=None, error=text, status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = obj.get("usage")
                    if usage:
                        usage_seen = True
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content is None and "reasoning_content" in delta:
                        content = delta.get("reasoning_content")
                    if not content:
                        continue
                    now = time.perf_counter()
                    chunks += 1
                    if ttft is None:
                        ttft = now - start
                    else:
                        itl.append(now - last)
                    last = now
        except (httpx.HTTPError, OSError) as e:
            return RequestResult(
                ok=False, start_ts=start_ts, e2e_s=time.perf_counter() - start,
                ttft_s=ttft, itl_s=itl, chunks=chunks, error=f"{type(e).__name__}: {e}",
            )
        e2e = time.perf_counter() - start
        if completion_tokens is None:
            completion_tokens = chunks
        return RequestResult(
            ok=chunks > 0, start_ts=start_ts, e2e_s=e2e, ttft_s=ttft, itl_s=itl,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            chunks=chunks, usage_from_server=usage_seen, status_code=200,
            error=None if chunks > 0 else "no content chunks received",
        )
