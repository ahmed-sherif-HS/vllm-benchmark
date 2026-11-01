"""Prompt generation with server-calibrated token lengths.

We don't know the server's tokenizer, so we calibrate: generate text with a
guessed words-per-token ratio, ask the server (max_tokens=1) how many prompt
tokens it saw, and adjust. Two or three rounds get within a few tokens.

Each prompt starts with a unique nonce so prefix caching cannot hide the
prefill cost between requests.
"""

from __future__ import annotations

import random
import string

_WORDS = (
    "system data model request response latency token server network memory "
    "compute vector matrix kernel batch stream buffer cache thread process "
    "signal packet query index table column schema value result error status "
    "client service cluster node shard replica queue worker task schedule "
    "metric sample window percentile average median throughput capacity load "
    "image pixel frame region object label caption describe detect classify "
    "analysis report summary detail context history prompt answer question"
).split()

# Fixed overhead the chat template adds (role tokens, BOS, etc.). Updated by
# calibration; initial guess is typical for Llama/Qwen style templates.
_INSTRUCTION = "Continue the following text with as much detail as you can:\n\n"


class PromptBuilder:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        # tokens per word, refined by calibrate()
        self.tokens_per_word: float = 1.3
        # tokens contributed by chat template + instruction, refined by calibrate()
        self.overhead_tokens: int = 20

    def nonce(self) -> str:
        return "".join(self._rng.choices(string.ascii_lowercase + string.digits, k=12))

    def body_words(self, n_words: int) -> str:
        return " ".join(self._rng.choices(_WORDS, k=max(0, n_words)))

    def words_for_tokens(self, target_tokens: int) -> int:
        avail = max(1, target_tokens - self.overhead_tokens)
        return max(1, round(avail / self.tokens_per_word))

    def build(self, target_tokens: int) -> str:
        """Prompt text expected to tokenise to ~target_tokens (with template)."""
        n = self.words_for_tokens(target_tokens)
        return f"[{self.nonce()}] {_INSTRUCTION}{self.body_words(n)}"

    def update_from_probe(self, n_words: int, measured_tokens: int, n_words_2: int, measured_tokens_2: int) -> None:
        """Fit tokens = overhead + tpw * words from two probes of different size."""
        dw = n_words_2 - n_words
        if dw <= 0:
            return
        tpw = (measured_tokens_2 - measured_tokens) / dw
        if tpw <= 0.05:
            return
        self.tokens_per_word = tpw
        self.overhead_tokens = max(0, round(measured_tokens - tpw * n_words))
