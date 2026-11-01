"""Compare several runs (e.g. CPU vs GPU) on one metric, cell by cell."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import CellSummary


def _load(run_dir: Path) -> tuple[str, list[dict]]:
    meta = json.loads((run_dir / "summary.json").read_text())
    label = meta.get("label") or run_dir.name
    rows = []
    for s in meta["summaries"]:
        cs = CellSummary(**s)
        rows.append(cs.flat())
    return label, rows


def compare_runs(run_dirs: list[Path], metric: str = "ttft_p50_ms") -> str:
    runs = [_load(d) for d in run_dirs]
    labels = [l for l, _ in runs]
    keys = sorted({(r["concurrency"], r["input_tokens"], r["output_tokens"]) for _, rows in runs for r in rows})
    lookup = [{(r["concurrency"], r["input_tokens"], r["output_tokens"]): r.get(metric) for r in rows} for _, rows in runs]
    hdr = "| C | in | out | " + " | ".join(labels) + " |"
    out = [f"metric: {metric}", "", hdr, "|" + "---|" * (3 + len(labels))]
    for k in keys:
        vals = [lk.get(k) for lk in lookup]
        cells = ["-" if v is None else f"{v:g}" for v in vals]
        if len(vals) == 2 and all(v for v in vals):
            cells[1] += f" ({vals[0] / vals[1]:.1f}x)" if "tok_per_s" not in metric and "req_per_s" not in metric else f" ({vals[1] / vals[0]:.1f}x)"
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")
    return "\n".join(out)
