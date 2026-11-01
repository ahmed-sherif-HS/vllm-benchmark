from pathlib import Path

from vllm_bench.metrics import CellSummary
from vllm_bench.report import markdown_table, write_plots


def _stats(v: float) -> dict:
    return {"mean": v, "p50": v, "p90": v, "p99": v, "min": v, "max": v}


def _cell(c: int, i: int, o: int, ttft: float, goodput: float) -> CellSummary:
    return CellSummary(
        concurrency=c, target_input_tokens=i, target_output_tokens=o,
        requests=4, ok=4, failed=0, wall_s=1.0,
        actual_input_tokens_mean=float(i), actual_output_tokens_mean=float(o),
        ttft=_stats(ttft), tpot=_stats(0.01), itl=_stats(0.01), e2e=_stats(ttft + 0.3),
        req_per_s=4.0, output_tok_per_s=40.0, total_tok_per_s=80.0,
        per_user_output_tok_per_s=10.0, errors=[], goodput=goodput, fail_timeout=1,
    )


def test_two_contexts_write_user_and_context_charts(tmp_path: Path):
    cells = [
        _cell(1, 128, 64, 0.4, 1.0),
        _cell(1, 512, 64, 0.8, 0.5),
        _cell(4, 128, 64, 0.6, 0.75),
        _cell(4, 512, 64, 1.2, 0.25),
    ]
    paths = write_plots(cells, tmp_path, "qwen")
    assert (tmp_path / "ttft_p50_ms.png").exists()
    assert (tmp_path / "ttft_p50_ms_vs_context.png").exists()
    assert (tmp_path / "goodput_vs_context.png").exists()
    assert (tmp_path / "e2e_p50_ms_vs_context.png").exists()
    text = markdown_table(cells)
    assert "TTFT p50" in text and "ITL p99" in text and "E2E p50" in text and "out tok/s" in text
    assert "goodput" in text and "| 1.00 | 1 | 0 | 0 | 0 |" in text
