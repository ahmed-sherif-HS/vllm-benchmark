"""Command line entry point: `vllm-bench run ...`, `vllm-bench mock-server`, `vllm-bench compare`."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from rich.console import Console

from .config import BenchConfig, load_config, parse_int_list, parse_json_arg

console = Console()


def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="run a benchmark sweep against an OpenAI-compatible endpoint")
    p.add_argument("-c", "--config", help="YAML config file (CLI flags override it)")
    p.add_argument("--base-url", help="e.g. http://localhost:8000/v1")
    p.add_argument("--model", help="model id; defaults to first entry of /v1/models")
    p.add_argument("--api-key", help="bearer token (default EMPTY)")
    p.add_argument("--label", help="free-text label for this run, e.g. cpu-12c or rtx3060")
    p.add_argument("--concurrency", help="comma list, e.g. 1,2,4,8,16")
    p.add_argument("--input-tokens", help="comma list, e.g. 128,512,2048")
    p.add_argument("--output-tokens", help="comma list, e.g. 64,256")
    p.add_argument("--requests-per-cell", type=int, help="fixed request count per cell")
    p.add_argument("--requests-per-user", type=int, help="requests each virtual user sends (default 4)")
    p.add_argument("--duration", type=float, help="seconds per cell instead of a request count")
    p.add_argument("--warmup", type=int, help="warmup requests per cell (default 2)")
    p.add_argument("--timeout", type=float, help="per-request timeout seconds")
    p.add_argument("--extra-body", help='JSON merged into every request body, e.g. \'{"ignore_eos": true}\'')
    p.add_argument("--no-ignore-eos", action="store_true", help="drop ignore_eos (for non-vLLM servers)")
    p.add_argument("--image", help="image file attached to every request (VLM benchmarking)")
    p.add_argument("--out-dir", help="results root (default results/)")
    p.add_argument("--run-name", help="sub-directory name (default <label>-<timestamp>)")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_run)


def _apply_overrides(cfg: BenchConfig, a: argparse.Namespace) -> BenchConfig:
    ep, sw, out = cfg.endpoint, cfg.sweep, cfg.output
    if a.base_url: ep.base_url = a.base_url
    if a.model: ep.model = a.model
    if a.api_key: ep.api_key = a.api_key
    if a.timeout: ep.timeout_s = a.timeout
    if a.image: ep.image_path = a.image
    if a.extra_body: ep.extra_body = {**(ep.extra_body or {}), **parse_json_arg(a.extra_body)}
    if a.no_ignore_eos: ep.extra_body.pop("ignore_eos", None)
    if a.label: cfg.label = a.label
    if a.concurrency: sw.concurrency = parse_int_list(a.concurrency)
    if a.input_tokens: sw.input_tokens = parse_int_list(a.input_tokens)
    if a.output_tokens: sw.output_tokens = parse_int_list(a.output_tokens)
    if a.requests_per_cell is not None: sw.requests_per_cell = a.requests_per_cell
    if a.requests_per_user is not None: sw.requests_per_user = a.requests_per_user
    if a.duration is not None: sw.duration_s = a.duration
    if a.warmup is not None: sw.warmup_requests = a.warmup
    if a.out_dir: out.dir = a.out_dir
    if a.run_name: out.run_name = a.run_name
    if a.no_plots: out.plots = False
    return cfg


def cmd_run(a: argparse.Namespace) -> int:
    from .report import write_report
    from .runner import run_sweep

    cfg = _apply_overrides(load_config(a.config), a)
    n_cells = len(cfg.cells())
    console.print(f"[bold]vllm-bench[/] -> {cfg.endpoint.base_url}  cells={n_cells}  "
                  f"C={cfg.sweep.concurrency} in={cfg.sweep.input_tokens} out={cfg.sweep.output_tokens}")

    def log(msg: str) -> None:
        console.print(f"[dim]{time.strftime('%H:%M:%S')}[/] {msg}", highlight=False)

    try:
        run = asyncio.run(run_sweep(cfg, log=log))
    except KeyboardInterrupt:
        console.print("[red]interrupted[/]")
        return 130
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]error:[/] {e}")
        return 1

    name = cfg.output.run_name or f"{(cfg.label or 'run').replace(' ', '_')}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(cfg.output.dir) / name
    paths = write_report(run, out_dir, plots=cfg.output.plots)
    console.print(f"\n[green]wrote[/] {out_dir}/")
    for k, p in paths.items():
        console.print(f"  {k:14s} {p.name}")
    console.print()
    console.print(Path(paths["report_md"]).read_text().split("\n\n", 3)[-1])
    return 0


def cmd_mock(a: argparse.Namespace) -> int:
    from .mock_server import serve
    console.print(f"mock OpenAI server on http://{a.host}:{a.port}/v1  (tok delay {a.token_delay_ms}ms, "
                  f"prefill {a.prefill_ms_per_1k}ms/1k tokens)")
    serve(a.host, a.port, token_delay_ms=a.token_delay_ms, prefill_ms_per_1k=a.prefill_ms_per_1k)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from .compare import compare_runs
    print(compare_runs([Path(p) for p in a.runs], metric=a.metric))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vllm-bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_run_parser(sub)

    m = sub.add_parser("mock-server", help="start a fake OpenAI-compatible server for testing")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8321)
    m.add_argument("--token-delay-ms", type=float, default=5.0)
    m.add_argument("--prefill-ms-per-1k", type=float, default=50.0)
    m.set_defaults(func=cmd_mock)

    c = sub.add_parser("compare", help="side-by-side table of several runs (e.g. cpu vs gpu)")
    c.add_argument("runs", nargs="+", help="run directories containing summary.json")
    c.add_argument("--metric", default="ttft_p50_ms",
                   help="column from summary.csv, e.g. ttft_p50_ms, tpot_p50_ms, e2e_p99_ms, output_tok_per_s")
    c.set_defaults(func=cmd_compare)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
