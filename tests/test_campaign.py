from pathlib import Path

from vllm_bench.cli import main
from vllm_bench.config import config_from_campaign, load_campaign, load_config

CAMPAIGN = Path(__file__).resolve().parents[1] / "configs" / "campaign.yaml"
GRID = ([1, 2, 4, 8, 16], [128, 512, 1024, 2048, 4096], [64, 256])


def test_checked_in_campaign_is_one_shared_grid():
    camp = load_campaign(CAMPAIGN)
    assert camp.grid() == GRID
    assert camp.gpu.provider == "vast.ai"
    assert camp.models == ["Qwen/Qwen3.5-9B"]
    gpu = config_from_campaign(camp, "gpu", camp.models[0])
    cpu = config_from_campaign(camp, "cpu", camp.models[0])
    assert gpu.sweep.concurrency == cpu.sweep.concurrency == GRID[0]
    assert gpu.sweep.input_tokens == cpu.sweep.input_tokens == GRID[1]
    assert gpu.sweep.output_tokens == cpu.sweep.output_tokens == GRID[2]
    assert gpu.cells() == cpu.cells()
    assert gpu.label.endswith("-gpu")
    assert cpu.label.endswith("-cpu")
    assert gpu.endpoint.model == camp.models[0]
    assert cpu.endpoint.timeout_s == 900
    assert gpu.device == "gpu" and cpu.device == "cpu"
    assert gpu.dtype == "auto" and gpu.max_model_len == 32000
    assert gpu.bars.ttft_ms == cpu.bars.ttft_ms == 2000
    assert gpu.bars.tpot_ms == 50
    assert "max-num-seqs 8" in gpu.launch_args
    assert cpu.dtype == "" and cpu.launch_args == "" and cpu.max_model_len is None


def test_gpu_and_cpu_configs_list_the_campaign_grid():
    root = CAMPAIGN.parent
    camp = load_campaign(CAMPAIGN)
    gpu = load_config(root / "gpu.yaml")
    cpu = load_config(root / "cpu.yaml")
    smoke = load_config(root / "smoke.yaml")
    assert gpu.sweep.concurrency == cpu.sweep.concurrency == list(camp.sweep.concurrency)
    assert gpu.sweep.input_tokens == cpu.sweep.input_tokens == list(camp.sweep.input_tokens)
    assert gpu.sweep.output_tokens == cpu.sweep.output_tokens == list(camp.sweep.output_tokens)
    assert cpu.endpoint.timeout_s == 900
    note = (root / "cpu.yaml").read_text().lower()
    assert "timeout is recorded" in note
    assert smoke.sweep.concurrency == [1, 4]
    assert smoke.sweep.input_tokens == [128]
    assert smoke.sweep.output_tokens == [32]


def test_campaign_run_uses_the_file_grid(mock_url, tmp_path: Path):
    path = tmp_path / "campaign.yaml"
    path.write_text(
        f"""
sweep:
  concurrency: [1]
  input_tokens: [32]
  output_tokens: [4]
  requests_per_user: 1
  warmup_requests: 0
gpu:
  base_url: {mock_url}
cpu:
  base_url: http://cpu.example:8000/v1
models:
  - id: mock-model
"""
    )
    rc = main([
        "run", "--campaign", str(path), "--device", "gpu",
        "--out-dir", str(tmp_path / "out"), "--run-name", "gpu", "--no-plots",
    ])
    assert rc == 0
    assert (tmp_path / "out" / "gpu" / "summary.json").exists()


def test_campaign_rejects_grid_overrides_and_a_missing_url(tmp_path: Path):
    path = tmp_path / "campaign.yaml"
    path.write_text(
        """
sweep:
  concurrency: [1, 2]
  input_tokens: [128]
  output_tokens: [64]
gpu:
  base_url: ""
cpu:
  base_url: http://cpu.example:8000/v1
models:
  - id: Qwen/Qwen2.5-1.5B-Instruct
  - id: Qwen3.6-35B-A3B
"""
    )
    rc = main(["run", "--campaign", str(path), "--device", "gpu", "--concurrency", "1"])
    assert rc == 2
    rc = main(["run", "--campaign", str(path), "--device", "gpu", "--model", "Qwen/Qwen2.5-1.5B-Instruct"])
    assert rc == 2
    rc = main(["run", "--campaign", str(path), "--device", "cpu"])
    assert rc == 2
