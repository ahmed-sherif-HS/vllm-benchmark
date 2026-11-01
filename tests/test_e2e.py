import json
from pathlib import Path

from vllm_bench.cli import main
from vllm_bench.client import OpenAIClient
from vllm_bench.config import BenchConfig, EndpointConfig
from vllm_bench.runner import run_sweep


async def test_stream_metrics(mock_url):
    client = OpenAIClient(EndpointConfig(base_url=mock_url), "mock-model")
    try:
        r = await client.stream_chat("hello " * 20, max_tokens=12)
    finally:
        await client.aclose()
    assert r.ok and r.completion_tokens == 12 and r.chunks == 12
    assert r.usage_from_server and r.prompt_tokens == int(20 * 1.3) + 10
    assert r.ttft_s is not None and len(r.itl_s) == 11


async def test_sweep_and_report(mock_url, tmp_path: Path):
    from vllm_bench.report import write_report

    cfg = BenchConfig(endpoint=EndpointConfig(base_url=mock_url))
    cfg.sweep.concurrency = [1, 2]
    cfg.sweep.input_tokens = [64, 256]
    cfg.sweep.output_tokens = [8]
    cfg.sweep.requests_per_user = 2
    cfg.sweep.warmup_requests = 1
    run = await run_sweep(cfg, log=lambda _m: None)
    assert run["model"] == "mock-model"
    assert len(run["cells"]) == 4
    for cell in run["cells"]:
        s = cell.summary
        assert s.failed == 0
        assert s.requests == s.concurrency * 2
        # calibrated input length within ~5% of target
        assert abs(s.actual_input_tokens_mean - s.target_input_tokens) <= max(4, 0.05 * s.target_input_tokens)
        assert s.actual_output_tokens_mean == 8
    paths = write_report(run, tmp_path / "r", plots=False)
    assert (tmp_path / "r" / "summary.csv").exists()
    meta = json.loads(paths["summary_json"].read_text())
    assert len(meta["summaries"]) == 4
    assert sum(1 for _ in paths["requests"].open()) == sum(s.summary.requests for s in run["cells"])


def test_cli_run_and_compare(mock_url, tmp_path: Path):
    for label in ("a", "b"):
        rc = main(["run", "--base-url", mock_url, "--concurrency", "1,2", "--input-tokens", "64",
                   "--output-tokens", "4", "--requests-per-user", "1", "--warmup", "0",
                   "--out-dir", str(tmp_path), "--run-name", label, "--label", label, "--no-plots"])
        assert rc == 0
    rc = main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--metric", "e2e_p50_ms"])
    assert rc == 0
