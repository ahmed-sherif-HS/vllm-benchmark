"""Write run artefacts: raw JSONL, summary CSV/JSON, Markdown table, plots."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .metrics import CellSummary
from .runner import CellRun


def describe_server(cfg: Any, model: str, card: dict[str, Any] | None) -> dict[str, Any]:
    """Identity of the server under test.

    Device, dtype, and launch flags come from the campaign. max_model_len
    prefers the value /v1/models returns. Nothing here reads nvidia-smi:
    a client with no GPU is not a CPU run.
    """
    card = card or {}
    device = cfg.device if getattr(cfg, "device", "") in ("gpu", "cpu") else None
    dtype = None
    for key in ("dtype", "torch_dtype"):
        val = card.get(key)
        if isinstance(val, str) and val:
            dtype = val
            break
    if dtype is None and getattr(cfg, "dtype", ""):
        dtype = cfg.dtype
    remote_len = card.get("max_model_len")
    if remote_len is None:
        max_model_len = getattr(cfg, "max_model_len", None)
    else:
        max_model_len = int(remote_len)
    launch = getattr(cfg, "launch_args", "") or None
    return {
        "device": device,
        "model": model,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "launch_args": launch,
    }


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
    }
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            info["gpus"] = [l.strip() for l in out.splitlines() if l.strip()]
        except Exception:  # noqa: BLE001
            pass
    return info


def _fmt_ms(v: float | None) -> str:
    return "-" if v is None else f"{v*1000:.0f}"


def markdown_table(summaries: list[CellSummary]) -> str:
    hdr = ("| C | in | out | ok | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | "
           "E2E p50 (ms) | E2E p99 (ms) | out tok/s | req/s | goodput | timeout | context | oom | other |")
    sep = "|" + "---|" * 17
    rows = [hdr, sep]
    for s in summaries:
        goodput = "-" if s.goodput is None else f"{s.goodput:.2f}"
        rows.append(
            f"| {s.concurrency} | {s.target_input_tokens} | {s.target_output_tokens} | {s.ok}/{s.requests} | "
            f"{_fmt_ms(s.ttft['p50'])} | {_fmt_ms(s.ttft['p99'])} | {_fmt_ms(s.tpot['p50'])} | "
            f"{_fmt_ms(s.itl['p99'])} | {_fmt_ms(s.e2e['p50'])} | {_fmt_ms(s.e2e['p99'])} | "
            f"{s.output_tok_per_s:.1f} | {s.req_per_s:.2f} | {goodput} | "
            f"{s.fail_timeout} | {s.fail_context_length} | {s.fail_oom} | {s.fail_other} |"
        )
    return "\n".join(rows)


def write_report(run: dict[str, Any], out_dir: Path, plots: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: list[CellRun] = run["cells"]
    summaries = [c.summary for c in cells]
    paths: dict[str, Path] = {}

    # raw per-request results
    p = out_dir / "requests.jsonl"
    with p.open("w") as f:
        for c in cells:
            s = c.summary
            for r in c.results:
                d = r.to_dict()
                d.update(concurrency=s.concurrency, input_tokens=s.target_input_tokens,
                         output_tokens=s.target_output_tokens)
                f.write(json.dumps(d) + "\n")
    paths["requests"] = p

    # summary JSON (everything except raw results)
    meta = {k: v for k, v in run.items() if k != "cells"}
    meta["host"] = host_info()
    server = dict(run.get("server") or {})
    meta["server"] = server
    meta["summaries"] = [s.to_dict() for s in summaries]
    p = out_dir / "summary.json"
    p.write_text(json.dumps(meta, indent=2))
    paths["summary_json"] = p

    # summary CSV
    p = out_dir / "summary.csv"
    flat = [s.flat() for s in summaries]
    if flat:
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
    paths["summary_csv"] = p

    # Markdown
    md = [
        f"# Benchmark: {run.get('label') or run['model']}",
        "",
        f"- model: `{run['model']}`",
        f"- device: `{server.get('device') or 'unset'}`",
        f"- dtype: `{server.get('dtype') or 'unset'}`",
        f"- max_model_len: `{server.get('max_model_len') if server.get('max_model_len') is not None else 'unset'}`",
        f"- launch_args: `{server.get('launch_args') or 'unset'}`",
        f"- endpoint: `{run['config']['endpoint']['base_url']}`",
        f"- started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run['started_at']))}",
        f"- duration: {run['finished_at'] - run['started_at']:.0f}s",
        f"- host: {meta['host'].get('hostname')} / {meta['host'].get('platform')} / "
        f"{meta['host'].get('cpu_count')} CPUs / GPUs: {', '.join(meta['host'].get('gpus', [])) or 'none visible'}",
        f"- calibration: {run['calibration']}",
        "",
        "TTFT = time to first token, TPOT = mean time per output token (per request), "
        "ITL = inter-token latency (all gaps), E2E = full request latency. C = concurrent users.",
        "",
        markdown_table(summaries),
        "",
    ]
    p = out_dir / "REPORT.md"
    p.write_text("\n".join(md))
    paths["report_md"] = p

    if plots:
        try:
            paths.update(write_plots(summaries, out_dir, run.get("label") or run["model"]))
        except ImportError:
            pass
    return paths


def _plot_metrics() -> list[tuple[str, str, Any, float]]:
    return [
        ("ttft_p50_ms", "TTFT p50 (ms)", lambda s: s.ttft["p50"], 1000),
        ("tpot_p50_ms", "TPOT p50 (ms)", lambda s: s.tpot["p50"], 1000),
        ("e2e_p50_ms", "E2E latency p50 (ms)", lambda s: s.e2e["p50"], 1000),
        ("output_tok_per_s", "Output throughput (tok/s)", lambda s: s.output_tok_per_s, 1),
        ("goodput", "Goodput (share of requests)", lambda s: s.goodput, 1),
    ]


def _draw(ax: Any, pts_by_line: list[tuple[str, list[tuple[float, float]]]], xlabel: str, ylabel: str, title: str, log_x: bool) -> None:
    for label, pts in pts_by_line:
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xs = sorted({x for _, pts in pts_by_line for x, _ in pts})
    if log_x and xs and min(xs) > 0:
        ax.set_xscale("log", base=2)
    if xs:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x) if x == int(x) else x) for x in xs])
    ax.minorticks_off()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def write_plots(summaries: list[CellSummary], out_dir: Path, title: str) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    user_shapes = sorted({(s.target_input_tokens, s.target_output_tokens) for s in summaries})
    contexts = {s.target_input_tokens for s in summaries}
    for key, label, getter, scale in _plot_metrics():
        fig, ax = plt.subplots(figsize=(8, 5))
        lines = []
        for (i, o) in user_shapes:
            pts = sorted(
                (s.concurrency, getter(s) * scale) for s in summaries
                if s.target_input_tokens == i and s.target_output_tokens == o and getter(s) is not None
            )
            lines.append((f"in={i} out={o}", pts))
        _draw(ax, lines, "concurrent users", label, f"{title}: {label} vs concurrency", log_x=True)
        fig.tight_layout()
        p = out_dir / f"{key}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[key] = p

        # One context length has no vs-context chart. The five metrics above all get one.
        if len(contexts) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        lines = []
        series = sorted({(s.concurrency, s.target_output_tokens) for s in summaries})
        for (c, o) in series:
            pts = sorted(
                (s.target_input_tokens, getter(s) * scale) for s in summaries
                if s.concurrency == c and s.target_output_tokens == o and getter(s) is not None
            )
            lines.append((f"C={c} out={o}", pts))
        _draw(ax, lines, "context length (input tokens)", label, f"{title}: {label} vs context length", log_x=True)
        fig.tight_layout()
        p = out_dir / f"{key}_vs_context.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[f"{key}_vs_context"] = p
    return paths
