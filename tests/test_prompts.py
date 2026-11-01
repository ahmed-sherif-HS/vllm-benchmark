from vllm_bench.prompts import PromptBuilder


def test_calibration_fit():
    pb = PromptBuilder(seed=1)
    # server: tokens = 10 + 1.3 * words
    pb.update_from_probe(50, 10 + 65, 400, 10 + 520)
    assert abs(pb.tokens_per_word - 1.3) < 1e-6
    assert pb.overhead_tokens == 10
    n = pb.words_for_tokens(1000)
    assert abs((10 + 1.3 * n) - 1000) < 2


def test_prompts_are_unique():
    pb = PromptBuilder(seed=1)
    a, b = pb.build(200), pb.build(200)
    assert a != b and a[:14] != b[:14]
