"""Aggregation of per-request results into per-cell summaries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from .client import RequestResult
from .config import BarConfig


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def failure_class(error: str | None) -> str:
    """Map an error string to timeout, context_length, oom, or other."""
    text = (error or "").lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "out of memory" in text or re.search(r"\boom\b", text):
        return "oom"
    if "context" in text and any(s in text for s in ("length", "longer", "maximum", "exceed", "window")):
        return "context_length"
    if "max_model_len" in text or "maximum model length" in text:
        return "context_length"
    return "other"


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
    # prompt tokens / time to first token, same shape as the latency stats
    prefill_tok_per_s: dict[str, float | None] = field(default_factory=lambda: _stats([]))
    # share of all requests that succeeded under both bars; None when bars are unset
    goodput: float | None = None
    fail_timeout: int = 0
    fail_context_length: int = 0
    fail_oom: int = 0
    fail_other: int = 0

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
        for k in ("mean", "p50", "p90", "p99"):
            d[f"prefill_tok_per_s_{k}"] = _r(self.prefill_tok_per_s.get(k), 1)
        d["goodput"] = _r(self.goodput, 3)
        d["error_rate"] = None if not self.requests else round(self.failed / self.requests, 3)
        d["fail_timeout"] = self.fail_timeout
        d["fail_context_length"] = self.fail_context_length
        d["fail_oom"] = self.fail_oom
        d["fail_other"] = self.fail_other
        return d


def _r(v: float | None, nd: int) -> float | None:
    return None if v is None else round(v, nd)


def _request_tag(r: RequestResult) -> str | None:
    if r.ok:
        return None
    tag = r.failure_class
    if tag in ("timeout", "context_length", "oom", "other"):
        return tag
    return failure_class(r.error)


def _meets_bars(r: RequestResult, bars: BarConfig | None) -> bool:
    if bars is None or (bars.ttft_ms is None and bars.tpot_ms is None):
        return False
    if not r.ok:
        return False
    if bars.ttft_ms is not None and (r.ttft_s is None or r.ttft_s * 1000 > bars.ttft_ms):
        return False
    if bars.tpot_ms is not None and (r.tpot_s is None or r.tpot_s * 1000 > bars.tpot_ms):
        return False
    return True


def summarize(
    results: Iterable[RequestResult],
    concurrency: int,
    target_in: int,
    target_out: int,
    wall_s: float,
    bars: BarConfig | None = None,
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
    prefill = [
        r.prompt_tokens / r.ttft_s
        for r in ok
        if r.prompt_tokens is not None and r.ttft_s is not None and r.ttft_s > 0
    ]
    errs = sorted({(r.error or "unknown")[:200] for r in failed})
    tags = [_request_tag(r) for r in failed]
    bars_set = bars is not None and (bars.ttft_ms is not None or bars.tpot_ms is not None)
    goodput = None
    if bars_set and rs:
        goodput = sum(1 for r in rs if _meets_bars(r, bars)) / len(rs)
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
        prefill_tok_per_s=_stats(prefill),
        goodput=goodput,
        fail_timeout=tags.count("timeout"),
        fail_context_length=tags.count("context_length"),
        fail_oom=tags.count("oom"),
        fail_other=tags.count("other"),
    )
