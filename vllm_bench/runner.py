"""Closed-loop load runner.

For each cell (concurrency C, input I, output O) we spawn C virtual users.
Each user sends requests back-to-back (no think time) until the cell's
request budget or duration is exhausted. This models C concurrent users
hammering the server and is what vLLM's own benchmark calls
"max concurrency" mode.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .client import OpenAIClient, RequestResult
from .config import BenchConfig
from .metrics import CellSummary, summarize
from .prompts import PromptBuilder

ProgressCb = Callable[[str], None]


@dataclass
class CellRun:
    summary: CellSummary
    results: list[RequestResult] = field(default_factory=list)


async def calibrate(client: OpenAIClient, pb: PromptBuilder, log: ProgressCb) -> dict[str, Any]:
    """Fit tokens-per-word and template overhead using two probe prompts."""
    w1, w2 = 50, 400
    p1 = f"[{pb.nonce()}] " + "Continue the following text with as much detail as you can:\n\n" + pb.body_words(w1)
    p2 = f"[{pb.nonce()}] " + "Continue the following text with as much detail as you can:\n\n" + pb.body_words(w2)
    t1 = await client.count_prompt_tokens(p1)
    t2 = await client.count_prompt_tokens(p2)
    pb.update_from_probe(w1, t1, w2, t2)
    # verification probe at a realistic size
    target = 512
    t3 = await client.count_prompt_tokens(pb.build(target))
    log(f"calibration: tokens/word={pb.tokens_per_word:.3f} overhead={pb.overhead_tokens} "
        f"(target {target} -> measured {t3})")
    return {
        "tokens_per_word": pb.tokens_per_word,
        "overhead_tokens": pb.overhead_tokens,
        "verify_target": target,
        "verify_measured": t3,
    }


async def run_cell(
    client: OpenAIClient,
    cfg: BenchConfig,
    pb: PromptBuilder,
    concurrency: int,
    in_tok: int,
    out_tok: int,
    log: ProgressCb,
) -> CellRun:
    n_requests = cfg.requests_for(concurrency)
    duration = cfg.sweep.duration_s
    temperature = cfg.sweep.temperature

    # Warmup: fire a few requests at the same shape, discard results.
    if cfg.sweep.warmup_requests > 0:
        warm = [client.stream_chat(pb.build(in_tok), out_tok, temperature)
                for _ in range(min(cfg.sweep.warmup_requests, max(1, concurrency)))]
        await asyncio.gather(*warm)

    results: list[RequestResult] = []
    issued = 0
    lock = asyncio.Lock()
    t0 = time.perf_counter()

    async def should_continue() -> bool:
        nonlocal issued
        async with lock:
            if duration is not None:
                if time.perf_counter() - t0 >= duration:
                    return False
            elif issued >= (n_requests or 0):
                return False
            issued += 1
            return True

    async def user() -> None:
        while await should_continue():
            r = await client.stream_chat(pb.build(in_tok), out_tok, temperature)
            results.append(r)

    await asyncio.gather(*(user() for _ in range(concurrency)))
    wall = time.perf_counter() - t0
    summary = summarize(results, concurrency, in_tok, out_tok, wall)
    log(_fmt_cell(summary))
    return CellRun(summary=summary, results=results)


def _ms(v: float | None) -> str:
    return "   -  " if v is None else f"{v*1000:7.0f}"


def _fmt_cell(s: CellSummary) -> str:
    return (f"C={s.concurrency:<3d} in={s.target_input_tokens:<5d} out={s.target_output_tokens:<5d} "
            f"ok={s.ok}/{s.requests}  TTFT p50/p99={_ms(s.ttft['p50'])}/{_ms(s.ttft['p99'])}ms  "
            f"TPOT p50={_ms(s.tpot['p50'])}ms  E2E p50={_ms(s.e2e['p50'])}ms  "
            f"{s.output_tok_per_s:7.1f} out tok/s  {s.req_per_s:5.2f} req/s"
            + (f"  ERR: {s.errors[0]}" if s.errors else ""))


async def run_sweep(
    cfg: BenchConfig,
    log: ProgressCb = print,
    on_cell: Callable[[CellRun], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    model = cfg.endpoint.model
    if not model:
        models = await OpenAIClient.list_models(cfg.endpoint)
        if not models:
            raise RuntimeError("No models returned by /v1/models; pass --model.")
        model = models[0]
        log(f"using model: {model}")

    client = OpenAIClient(cfg.endpoint, model)
    pb = PromptBuilder(cfg.sweep.seed)
    started = time.time()
    cells: list[CellRun] = []
    try:
        calib = await calibrate(client, pb, log)
        for (c, i, o) in cfg.cells():
            cell = await run_cell(client, cfg, pb, c, i, o, log)
            cells.append(cell)
            if on_cell:
                r = on_cell(cell)
                if asyncio.iscoroutine(r):
                    await r
    finally:
        await client.aclose()
    return {
        "model": model,
        "label": cfg.label,
        "started_at": started,
        "finished_at": time.time(),
        "config": cfg.to_dict(),
        "calibration": calib,
        "cells": cells,
    }
