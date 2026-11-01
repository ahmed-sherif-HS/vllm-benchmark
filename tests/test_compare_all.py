import json
from pathlib import Path

from vllm_bench.cli import main
from vllm_bench.metrics import CellSummary


def _stats(v: float) -> dict:
    return {"mean": v, "p50": v, "p90": v, "p99": v, "min": v, "max": v}


def _cell(c: int, i: int, ttft: float, out_tok: float, goodput: float, failed: int) -> dict:
    s = CellSummary(
        concurrency=c, target_input_tokens=i, target_output_tokens=64,
        requests=10, ok=10 - failed, failed=failed, wall_s=1.0,
        actual_input_tokens_mean=float(i), actual_output_tokens_mean=64,
        ttft=_stats(ttft / 1000), tpot=_stats(0.01), itl=_stats(0.01), e2e=_stats(0.5),
        req_per_s=out_tok / 10, output_tok_per_s=out_tok, total_tok_per_s=out_tok,
        per_user_output_tok_per_s=out_tok / c, errors=[], goodput=goodput,
    )
    return s.to_dict()


def _write(path: Path, device: str, cells: list[dict]) -> None:
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({
        "label": f"qwen-{device}",
        "server": {"device": device, "model": "Qwen/Qwen3.5-9B"},
        "summaries": cells,
    }))


def test_compare_all_labels_ratio_direction_and_keeps_absent_cells(tmp_path: Path, capsys):
    gpu = tmp_path / "gpu"
    cpu = tmp_path / "cpu"
    _write(gpu, "gpu", [
        _cell(1, 128, ttft=100, out_tok=50, goodput=0.9, failed=0),
        _cell(1, 512, ttft=200, out_tok=40, goodput=0.8, failed=0),
    ])
    _write(cpu, "cpu", [
        _cell(1, 128, ttft=400, out_tok=10, goodput=0.4, failed=1),
    ])
    rc = main(["compare", str(cpu), str(gpu), "--all"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "CPU/GPU" in text and "GPU/CPU" in text
    assert "| 1 | 128 | 64 | 100 | 400 | 4.0x |" in text
    assert "| 1 | 128 | 64 | 50 | 10 | 5.0x |" in text
    assert "| 1 | 512 | 64 | 200 | absent | - |" in text
    assert "error rate" in text
