import json
import re
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from vllm_bench.dashboard import DashboardServer, list_runs, load_run, serve, static_dir, to_public


def _raw(api_key: str = "super-secret-token") -> dict:
    return {
        "label": "demo-gpu",
        "model": "demo",
        "started_at": 10,
        "finished_at": 70,
        "config": {
            "device": "gpu",
            "bars": {"ttft_ms": 2000, "tpot_ms": 50},
            "endpoint": {
                "base_url": f"http://user:{api_key}@10.1.2.3:8000/v1?token={api_key}",
                "api_key": api_key,
            },
        },
        "host": {"hostname": "secret-host", "cpu_count": 8},
        "summaries": [
            {
                "concurrency": 1,
                "target_input_tokens": 128,
                "target_output_tokens": 64,
                "requests": 4,
                "ok": 4,
                "failed": 0,
                "ttft": {"p50": 0.5, "p99": 0.6},
                "tpot": {"p50": 0.012},
                "e2e": {"p50": 1.2},
                "itl": {"p99": 0.02},
                "output_tok_per_s": 40.0,
                "req_per_s": 1.0,
                "goodput": 1.0,
            },
            {
                "concurrency": 4,
                "target_input_tokens": 4096,
                "target_output_tokens": 64,
                "requests": 16,
                "ok": 16,
                "failed": 0,
                "ttft": {"p50": 1.829, "p99": 2.220},
                "tpot": {"p50": 0.017},
                "e2e": {"p50": 2.9},
                "itl": {"p99": 0.16},
                "output_tok_per_s": 87.4,
                "req_per_s": 1.37,
                "goodput": 0.6875,
            },
        ],
    }


def test_public_view_drops_secrets_and_finds_the_break():
    pub = to_public(_raw(), "results/demo")
    blob = json.dumps(pub)
    assert "super-secret-token" not in blob
    assert "secret-host" not in blob
    assert pub["endpoint"] == "http://10.1.2.3:8000/v1"
    assert pub["break"]["concurrency"] == 4
    assert pub["break"]["input_tokens"] == 4096
    assert pub["break"]["goodput"] == 0.6875
    assert pub["client_cpus"] == 8
    assert pub["requests_ok"] == 20


def test_results_shadow_examples_and_reject_odd_names(tmp_path: Path):
    results = tmp_path / "results"
    examples = tmp_path / "examples"
    for root in (results, examples):
        run = root / "demo"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps(_raw()))
    odd = results / "has space"
    odd.mkdir()
    (odd / "summary.json").write_text(json.dumps(_raw()))
    listed = list_runs([results, examples])
    assert [item["id"] for item in listed] == ["results/demo"]
    assert load_run([results, examples], "examples", "demo") is None
    assert load_run([results, examples], "..", "demo") is None
    assert load_run([results, examples], "results", "..") is None
    assert "super-secret-token" not in json.dumps(load_run([results, examples], "results", "demo"))


def test_dashboard_http_serves_html_and_hides_the_key(tmp_path: Path):
    root = tmp_path / "results"
    run = root / "demo"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(_raw()))
    httpd = DashboardServer(("127.0.0.1", 0), [root])
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        page = urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "vllm-bench" in page
        assert (static_dir() / "app.js").is_file()
        body = urlopen(f"http://127.0.0.1:{port}/api/runs/results/demo").read().decode()
        assert "super-secret-token" not in body
        assert json.loads(body)["break"]["input_tokens"] == 4096
        try:
            urlopen(f"http://127.0.0.1:{port}/api/runs/results/nope")
            missing = None
        except HTTPError as exc:
            missing = exc.code
        assert missing == 404
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_serve_refuses_a_public_bind():
    try:
        serve([Path(".")], host="0.0.0.0", port=9)
    except ValueError as exc:
        assert "localhost" in str(exc)
    else:
        raise AssertionError("expected localhost-only refusal")


def test_published_gpu_grid_is_sanitized():
    path = Path("examples/qwen35-9b-gpu-grid/summary.json")
    text = path.read_text()
    assert not re.search(r"\b[0-9a-f]{32,}\b", text)
    raw = json.loads(text)
    pub = to_public(raw, "examples/qwen35-9b-gpu-grid")
    assert pub["device"] == "gpu"
    assert pub["model"] == "Qwen/Qwen3.5-9B"
    assert pub["requests_ok"] == pub["requests"]
    assert pub["break"]["concurrency"] == 4
    assert pub["break"]["input_tokens"] == 4096
    assert len(pub["cells"]) == 50
