import json
from pathlib import Path

from vllm_bench.config import BenchConfig, config_from_campaign, load_campaign
from vllm_bench.report import describe_server, write_report

CAMPAIGN = Path(__file__).resolve().parents[1] / "configs" / "campaign.yaml"


def _run(cfg: BenchConfig, server: dict) -> dict:
    return {
        "model": cfg.endpoint.model,
        "label": cfg.label,
        "started_at": 0,
        "finished_at": 1,
        "config": cfg.to_dict(),
        "calibration": {},
        "server": server,
        "cells": [],
    }


def test_same_model_two_devices_are_distinct_without_a_client_gpu(tmp_path: Path):
    camp = load_campaign(CAMPAIGN)
    model = camp.models[0]
    # The client that writes the report has no GPU. Device still follows the campaign.
    card = {"id": model, "max_model_len": 32000}
    gpu_cfg = config_from_campaign(camp, "gpu", model)
    cpu_cfg = config_from_campaign(camp, "cpu", model)
    gpu = describe_server(gpu_cfg, model, card)
    cpu = describe_server(cpu_cfg, model, card)
    assert gpu["device"] == "gpu"
    assert cpu["device"] == "cpu"
    assert gpu["model"] == cpu["model"] == model
    assert gpu["max_model_len"] == cpu["max_model_len"] == 32000
    assert gpu["dtype"] == "auto"
    assert cpu["dtype"] is None
    assert gpu["launch_args"] != cpu["launch_args"]

    for cfg, server in ((gpu_cfg, gpu), (cpu_cfg, cpu)):
        out = tmp_path / server["device"]
        paths = write_report(_run(cfg, server), out, plots=False)
        meta = json.loads(paths["summary_json"].read_text())
        report = paths["report_md"].read_text()
        assert meta["server"]["device"] == server["device"]
        assert f"device: `{server['device']}`" in report
        assert meta["server"]["model"] == model
        assert "max_model_len: `32000`" in report
        # Host GPUs stay a separate fact. They do not decide the device label.
        assert "gpus" not in meta["server"]


def test_server_max_model_len_overrides_the_campaign_value():
    camp = load_campaign(CAMPAIGN)
    cfg = config_from_campaign(camp, "gpu", camp.models[0])
    assert cfg.max_model_len == 32000
    server = describe_server(cfg, camp.models[0], {"max_model_len": 8192, "dtype": "bfloat16"})
    assert server["max_model_len"] == 8192
    assert server["dtype"] == "bfloat16"
    assert server["device"] == "gpu"


def test_missing_gpu_does_not_label_a_plain_run_as_cpu():
    cfg = BenchConfig()
    server = describe_server(cfg, "some-model", {})
    assert server["device"] is None
    assert server["dtype"] is None
    assert server["max_model_len"] is None
    assert server["launch_args"] is None
