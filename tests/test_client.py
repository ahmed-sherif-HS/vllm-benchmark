from vllm_bench.client import _delta_text


def test_delta_text_reads_reasoning_when_content_is_blank():
    assert _delta_text({"role": "assistant", "content": ""}) == ""
    assert _delta_text({"reasoning": "Thinking"}) == "Thinking"
    assert _delta_text({"content": "", "reasoning_content": "step"}) == "step"
    assert _delta_text({"content": "Hi", "reasoning": "done"}) == "Hidone"
