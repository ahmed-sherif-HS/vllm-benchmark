"""Minimal OpenAI-compatible mock server (stdlib only) for tests and dry runs.

Tokenisation: whitespace words * 1.3 + 10 template tokens. Streams `max_tokens`
chunks with a configurable delay, after a prefill delay proportional to the
prompt length. Honours stream_options.include_usage and ignore_eos.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def fake_token_count(messages: list[dict]) -> int:
    n_words = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n_words += len(c.split())
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    n_words += len(part["text"].split())
                elif part.get("type") == "image_url":
                    n_words += 200  # pretend an image is ~260 tokens
    return int(n_words * 1.3) + 10


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    token_delay_s = 0.005
    prefill_s_per_token = 0.00005
    model_id = "mock-model"

    def log_message(self, *_a) -> None:  # silence
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{
                "id": self.model_id, "object": "model", "max_model_len": 4096,
            }]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt_tokens = fake_token_count(req.get("messages", []))
        max_tokens = int(req.get("max_tokens", 16))
        time.sleep(prompt_tokens * self.prefill_s_per_token)
        rid = f"chatcmpl-{int(time.time()*1000)}"
        if not req.get("stream"):
            self._json(200, {
                "id": rid, "object": "chat.completion", "model": self.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "x " * max_tokens},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                          "total_tokens": prompt_tokens + max_tokens},
            })
            return
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj: dict) -> None:
            data = b"data: " + json.dumps(obj).encode() + b"\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        for i in range(max_tokens):
            send({"id": rid, "object": "chat.completion.chunk", "model": self.model_id,
                  "choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}]})
            time.sleep(self.token_delay_s)
        send({"id": rid, "object": "chat.completion.chunk", "model": self.model_id,
              "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        if include_usage:
            send({"id": rid, "object": "chat.completion.chunk", "model": self.model_id, "choices": [],
                  "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                            "total_tokens": prompt_tokens + max_tokens}})
        done = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
        self.wfile.flush()


def make_server(host: str, port: int, token_delay_ms: float = 5.0, prefill_ms_per_1k: float = 50.0) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {
        "token_delay_s": token_delay_ms / 1000.0,
        "prefill_s_per_token": prefill_ms_per_1k / 1000.0 / 1000.0,
    })
    ThreadingHTTPServer.request_queue_size = 1024
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def start_background(host: str = "127.0.0.1", port: int = 0, **kw) -> tuple[ThreadingHTTPServer, str]:
    srv = make_server(host, port, **kw)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://{srv.server_address[0]}:{srv.server_address[1]}/v1"


def serve(host: str, port: int, **kw) -> None:
    srv = make_server(host, port, **kw)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
