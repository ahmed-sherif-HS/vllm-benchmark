"""Aggregation of per-request results into per-cell summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from .client import RequestResult


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p99": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


@dataclass
class CellSummary:
    concurrency: int
    target_input_tokens: int
    target_output_tokens: int
    requests: int
    ok: int
    failed: int
    wall_s: float
    # achieved token counts
    actual_input_tokens_mean: float | None
    actual_output_tokens_mean: float | None
    # latency stats (seconds)
    ttft: dict[str, float | None]
    tpot: dict[str, float | None]
    itl: dict[str, float | None]
    e2e: dict[str, float | None]
    # throughput
    req_per_s: float
    output_tok_per_s: float
    total_tok_per_s: float
    per_user_output_tok_per_s: float | None
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat(self) -> dict[str, Any]:
        """Single-level dict for CSV."""
        d: dict[str, Any] = {
            "concurrency": self.concurrency,
            "input_tokens": self.target_input_tokens,
            "output_tokens": self.target_output_tokens,
            "requests": self.requests,
            "ok": self.ok,
            "failed": self.failed,
            "wall_s": round(self.wall_s, 3),
            "actual_input_tokens_mean": _r(self.actual_input_tokens_mean, 1),
            "actual_output_tokens_mean": _r(self.actual_output_tokens_mean, 1),
            "req_per_s": round(self.req_per_s, 3),
            "output_tok_per_s": round(self.output_tok_per_s, 1),
            "total_tok_per_s": round(self.total_tok_per_s, 1),
            "per_user_output_tok_per_s": _r(self.per_user_output_tok_per_s, 2),
        }
        for name, st in (("ttft", self.ttft), ("tpot", self.tpot), ("itl", self.itl), ("e2e", self.e2e)):
            for k in ("mean", "p50", "p90", "p99"):
                v = st.get(k)
                d[f"{name}_{k}_ms"] = _r(None if v is None else v * 1000, 1)
        return d


def _r(v: float | None, nd: int) -> float | None:
    return None if v is None else round(v, nd)


def summarize(
    results: Iterable[RequestResult],
    concurrency: int,
    target_in: int,
    target_out: int,
    wall_s: float,
) -> CellSummary:
    rs = list(results)
    ok = [r for r in rs if r.ok]
    failed = [r for r in rs if not r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    itl = [x for r in ok for x in r.itl_s]
    e2e = [r.e2e_s for r in ok]
    in_tok = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    out_tok = [r.completion_tokens for r in ok if r.completion_tokens is not None]
    total_out = sum(out_tok)
    total_in = sum(in_tok)
    wall = max(wall_s, 1e-9)
    per_user = [r.output_tok_per_s for r in ok if r.output_tok_per_s]
    errs = sorted({(r.error or "unknown")[:200] for r in failed})
    return CellSummary(
        concurrency=concurrency,
        target_input_tokens=target_in,
        target_output_tokens=target_out,
        requests=len(rs),
        ok=len(ok),
        failed=len(failed),
        wall_s=wall_s,
        actual_input_tokens_mean=(total_in / len(in_tok)) if in_tok else None,
        actual_output_tokens_mean=(total_out / len(out_tok)) if out_tok else None,
        ttft=_stats(ttft),
        tpot=_stats(tpot),
        itl=_stats(itl),
        e2e=_stats(e2e),
        req_per_s=len(ok) / wall,
        output_tok_per_s=total_out / wall,
        total_tok_per_s=(total_out + total_in) / wall,
        per_user_output_tok_per_s=(sum(per_user) / len(per_user)) if per_user else None,
        errors=errs[:5],
    )
