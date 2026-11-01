"""Benchmark configuration: loaded from YAML and/or CLI overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EndpointConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str | None = None  # None -> first model from /v1/models
    api_key: str = "EMPTY"
    timeout_s: float = 600.0
    # Extra JSON fields merged into every chat/completions body.
    # vLLM honours ignore_eos/min_tokens so output length is exact.
    extra_body: dict[str, Any] = field(
        default_factory=lambda: {"ignore_eos": True}
    )
    # Optional image attached to every request (for vision-language models).
    image_path: str | None = None
    image_detail: str = "auto"


@dataclass
class SweepConfig:
    concurrency: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    input_tokens: list[int] = field(default_factory=lambda: [128, 512, 1024])
    output_tokens: list[int] = field(default_factory=lambda: [64, 256])
    # Requests per (concurrency, input, output) cell. Each virtual user runs
    # requests back-to-back until the cell total is reached.
    requests_per_cell: int | None = None
    # If set, each cell runs for this many seconds instead of a fixed count.
    duration_s: float | None = None
    # Minimum requests per user when requests_per_cell is None and no duration.
    requests_per_user: int = 4
    warmup_requests: int = 2
    temperature: float = 0.0
    seed: int = 1234


@dataclass
class OutputConfig:
    dir: str = "results"
    run_name: str | None = None  # default: timestamp
    plots: bool = True


@dataclass
class BenchConfig:
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    label: str = ""  # free-text label, e.g. "cpu-12core" or "rtx3060"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cells(self) -> list[tuple[int, int, int]]:
        """All (concurrency, input_tokens, output_tokens) combinations."""
        return [
            (c, i, o)
            for c in self.sweep.concurrency
            for i in self.sweep.input_tokens
            for o in self.sweep.output_tokens
        ]

    def requests_for(self, concurrency: int) -> int | None:
        """Request count for a cell, or None when duration-based."""
        if self.sweep.duration_s:
            return None
        if self.sweep.requests_per_cell:
            return max(self.sweep.requests_per_cell, concurrency)
        return concurrency * self.sweep.requests_per_user


def _merge(dc, data: dict[str, Any]):
    for k, v in data.items():
        if not hasattr(dc, k):
            raise ValueError(f"Unknown config key: {k}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None) -> BenchConfig:
    cfg = BenchConfig()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    return cfg


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def parse_json_arg(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    v = json.loads(s)
    if not isinstance(v, dict):
        raise ValueError("--extra-body must be a JSON object")
    return v
