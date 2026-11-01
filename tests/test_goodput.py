from vllm_bench.client import OpenAIClient
from vllm_bench.config import BarConfig, EndpointConfig
from vllm_bench.metrics import failure_class, summarize
from vllm_bench.mock_server import start_background


def test_failure_class_tags():
    assert failure_class("ReadTimeout: timed out") == "timeout"
    assert failure_class("The input exceeds the maximum context length") == "context_length"
    assert failure_class("CUDA out of memory") == "oom"
    assert failure_class("boom") == "other"


async def test_mock_timeout_and_success_fix_goodput():
    fast, fast_url = start_background(token_delay_ms=1.0, prefill_ms_per_1k=1.0)
    slow, slow_url = start_background(token_delay_ms=500.0, prefill_ms_per_1k=500.0)
    bars = BarConfig(ttft_ms=5000, tpot_ms=1000)
    try:
        ok_client = OpenAIClient(EndpointConfig(base_url=fast_url, timeout_s=5), "mock-model")
        slow_client = OpenAIClient(EndpointConfig(base_url=slow_url, timeout_s=0.05), "mock-model")
        try:
            success = await ok_client.stream_chat("hello " * 8, max_tokens=8)
            timed_out = await slow_client.stream_chat("hello " * 8, max_tokens=8)
        finally:
            await ok_client.aclose()
            await slow_client.aclose()
    finally:
        fast.shutdown()
        fast.server_close()
        slow.shutdown()
        slow.server_close()

    assert success.ok and success.failure_class is None
    assert not timed_out.ok and timed_out.failure_class == "timeout"
    summary = summarize([success, timed_out], concurrency=1, target_in=16, target_out=8, wall_s=1.0, bars=bars)
    assert summary.fail_timeout == 1
    assert summary.fail_context_length == summary.fail_oom == summary.fail_other == 0
    assert summary.goodput == 0.5
    assert summary.prefill_tok_per_s["p50"] is not None
    flat = summary.flat()
    assert flat["goodput"] == 0.5
    assert flat["fail_timeout"] == 1
    assert flat["prefill_tok_per_s_p50"] == round(summary.prefill_tok_per_s["p50"], 1)
