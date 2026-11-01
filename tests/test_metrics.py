from vllm_bench.client import RequestResult
from vllm_bench.metrics import percentile, summarize


def test_percentile_interpolates():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 50) == 2.5
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile([], 50) is None


def _res(e2e, ttft, itl, out):
    return RequestResult(ok=True, start_ts=0, e2e_s=e2e, ttft_s=ttft, itl_s=itl,
                         prompt_tokens=100, completion_tokens=out)


def test_summarize_basic():
    rs = [_res(1.0, 0.2, [0.1] * 8, 9), _res(2.0, 0.4, [0.2] * 8, 9),
          RequestResult(ok=False, start_ts=0, e2e_s=0.1, ttft_s=None, error="boom")]
    s = summarize(rs, concurrency=2, target_in=100, target_out=9, wall_s=2.0)
    assert s.requests == 3 and s.ok == 2 and s.failed == 1
    assert abs(s.ttft["p50"] - 0.3) < 1e-9
    assert abs(s.tpot["mean"] - 0.15) < 1e-9
    assert s.output_tok_per_s == 9.0  # 18 tokens / 2 s
    assert s.req_per_s == 1.0
    assert s.errors == ["boom"]
    assert s.fail_other == 1 and s.fail_timeout == 0
    assert s.prefill_tok_per_s["p50"] == 375.0
    flat = s.flat()
    assert flat["ttft_p50_ms"] == 300.0
    assert flat["prefill_tok_per_s_p50"] == 375.0
    assert "goodput" in flat and "fail_timeout" in flat
