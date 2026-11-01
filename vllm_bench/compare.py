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


# Latency ratio is CPU/GPU. Throughput and goodput ratio is GPU/CPU.
ALL_METRICS: list[tuple[str, str, str]] = [
    ("ttft_p50_ms", "TTFT p50 (ms)", "cpu/gpu"),
    ("tpot_p50_ms", "TPOT p50 (ms)", "cpu/gpu"),
    ("itl_p50_ms", "ITL p50 (ms)", "cpu/gpu"),
    ("e2e_p50_ms", "E2E p50 (ms)", "cpu/gpu"),
    ("output_tok_per_s", "output tok/s", "gpu/cpu"),
    ("per_user_output_tok_per_s", "per-user tok/s", "gpu/cpu"),
    ("req_per_s", "req/s", "gpu/cpu"),
    ("goodput", "goodput", "gpu/cpu"),
    ("error_rate", "error rate", "cpu/gpu"),
]


def _meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def _device_of(meta: dict, label: str) -> str | None:
    device = (meta.get("server") or {}).get("device")
    if device in ("gpu", "cpu"):
        return device
    if label.endswith("-gpu") or label == "gpu":
        return "gpu"
    if label.endswith("-cpu") or label == "cpu":
        return "cpu"
    return None


def compare_devices(run_dirs: list[Path]) -> str:
    """Every metric for one GPU run and one CPU run, with the ratio direction labeled."""
    loaded = []
    for d in run_dirs:
        meta = _meta(d)
        label = meta.get("label") or d.name
        _, rows = _load(d)
        device = _device_of(meta, label)
        loaded.append((device, label, rows))
    by_device = {device: (label, rows) for device, label, rows in loaded if device in ("gpu", "cpu")}
    if "gpu" not in by_device or "cpu" not in by_device:
        raise ValueError("compare --all needs one gpu run and one cpu run (server.device, or a -gpu/-cpu label)")
    gpu_label, gpu_rows = by_device["gpu"]
    cpu_label, cpu_rows = by_device["cpu"]
    gpu = {(r["concurrency"], r["input_tokens"], r["output_tokens"]): r for r in gpu_rows}
    cpu = {(r["concurrency"], r["input_tokens"], r["output_tokens"]): r for r in cpu_rows}
    keys = sorted(set(gpu) | set(cpu))
    sections = [
        f"GPU: {gpu_label}",
        f"CPU: {cpu_label}",
        "",
        "Latency ratio is CPU/GPU (how many times slower). Throughput and goodput ratio is GPU/CPU.",
        "A cell missing on one side is absent, not dropped.",
        "",
    ]
    for metric, title, direction in ALL_METRICS:
        ratio_name = "CPU/GPU" if direction == "cpu/gpu" else "GPU/CPU"
        sections.append(f"## {title}")
        sections.append("")
        sections.append(f"| C | in | out | GPU | CPU | {ratio_name} |")
        sections.append("|---|---|---|---|---|---|")
        for k in keys:
            gv = gpu.get(k, {}).get(metric) if k in gpu else None
            cv = cpu.get(k, {}).get(metric) if k in cpu else None
            gcell = "absent" if k not in gpu else ("-" if gv is None else f"{gv:g}")
            ccell = "absent" if k not in cpu else ("-" if cv is None else f"{cv:g}")
            if k in gpu and k in cpu and gv is not None and cv is not None:
                if direction == "cpu/gpu" and gv:
                    ratio = f"{cv / gv:.1f}x"
                elif direction == "gpu/cpu" and cv:
                    ratio = f"{gv / cv:.1f}x"
                else:
                    ratio = "-"
            else:
                ratio = "-"
            sections.append(f"| {k[0]} | {k[1]} | {k[2]} | {gcell} | {ccell} | {ratio} |")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


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
