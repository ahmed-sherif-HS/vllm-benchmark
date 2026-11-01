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
class BarConfig:
    """Latency bars for goodput. Shared by GPU and CPU."""

    ttft_ms: float | None = None
    tpot_ms: float | None = None


@dataclass
class BenchConfig:
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    label: str = ""  # free-text label, e.g. "cpu-12core" or "rtx3060"
    # Set only by the campaign. Empty means the run does not name a device.
    device: str = ""
    dtype: str = ""
    max_model_len: int | None = None
    launch_args: str = ""
    bars: BarConfig = field(default_factory=BarConfig)

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


@dataclass
class DeviceConfig:
    """One server. GPU is a vast.ai instance; you start vLLM there yourself."""

    base_url: str = ""
    api_key: str = "EMPTY"
    timeout_s: float = 600.0
    # Offer you rented, e.g. "RTX 4090". Empty on the CPU device.
    gpu_type: str = ""
    provider: str = ""
    # Written onto the run. Not inferred from the client machine.
    dtype: str = ""
    max_model_len: int | None = None
    launch_args: str = ""


@dataclass
class Campaign:
    """Models, both devices, and the one sweep every run must use."""

    sweep: SweepConfig = field(default_factory=SweepConfig)
    gpu: DeviceConfig = field(default_factory=DeviceConfig)
    cpu: DeviceConfig = field(default_factory=DeviceConfig)
    models: list[str] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    bars: BarConfig = field(default_factory=BarConfig)

    def grid(self) -> tuple[list[int], list[int], list[int]]:
        s = self.sweep
        return list(s.concurrency), list(s.input_tokens), list(s.output_tokens)


def load_campaign(path: str | Path) -> Campaign:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = set(data) - {"sweep", "gpu", "cpu", "models", "output", "bars"}
    if unknown:
        raise ValueError(f"Unknown campaign keys: {', '.join(sorted(unknown))}")
    camp = Campaign()
    if "sweep" in data:
        _merge(camp.sweep, data["sweep"])
    if "gpu" in data:
        _merge(camp.gpu, data["gpu"])
    if "cpu" in data:
        _merge(camp.cpu, data["cpu"])
    if "output" in data:
        _merge(camp.output, data["output"])
    if "bars" in data:
        _merge(camp.bars, data["bars"])
    models = data.get("models") or []
    if not isinstance(models, list) or not models:
        raise ValueError("campaign.models must list at least one model id")
    ids: list[str] = []
    for item in models:
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            extra = set(item) - {"id"}
            if extra:
                raise ValueError(f"Unknown model keys: {', '.join(sorted(extra))}")
            model_id = item["id"]
        else:
            raise ValueError("each campaign model must be an id string or {id: ...}")
        if model_id in ids:
            raise ValueError(f"duplicate model id: {model_id}")
        ids.append(model_id)
    camp.models = ids
    return camp


def model_slug(model_id: str) -> str:
    name = model_id.split("/")[-1].lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")
    return slug or "model"


def config_from_campaign(camp: Campaign, device: str, model: str) -> BenchConfig:
    """Bench config for one model on one device. Both devices share camp.sweep."""
    if device not in ("gpu", "cpu"):
        raise ValueError("device must be gpu or cpu")
    if model not in camp.models:
        raise ValueError(f"{model!r} is not in the campaign. Listed: {', '.join(camp.models)}")
    dev = camp.gpu if device == "gpu" else camp.cpu
    sweep = SweepConfig(**asdict(camp.sweep))
    return BenchConfig(
        endpoint=EndpointConfig(
            base_url=dev.base_url,
            model=model,
            api_key=dev.api_key,
            timeout_s=dev.timeout_s,
        ),
        sweep=sweep,
        output=OutputConfig(**asdict(camp.output)),
        label=f"{model_slug(model)}-{device}",
        device=device,
        dtype=dev.dtype,
        max_model_len=dev.max_model_len,
        launch_args=dev.launch_args,
        bars=BarConfig(ttft_ms=camp.bars.ttft_ms, tpot_ms=camp.bars.tpot_ms),
    )
