# vllm-bench

Latency and throughput benchmark for any **OpenAI-compatible** LLM / VLM endpoint
(vLLM on CPU or GPU, llama.cpp server, TGI, Ollama, ...). It sweeps

- **users** (concurrent closed-loop clients),
- **input length** (prompt tokens),
- **output length** (generated tokens),

and records TTFT, inter-token latency, end-to-end latency and token throughput
for every combination, then writes CSV / JSON / Markdown / PNG reports that you
can compare across machines (e.g. CPU vs GPU).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[plots,dev]"
```

## Quick start

```bash
# 1. a server to test (examples below), e.g. vLLM on http://localhost:8000/v1
# 2. smoke test
vllm-bench run -c configs/smoke.yaml --base-url http://localhost:8000/v1

# 3. full sweep
vllm-bench run -c configs/gpu.yaml --label rtx3060
vllm-bench run -c configs/cpu.yaml --label cpu-12c --base-url http://cpu-box:8000/v1

# 4. compare two runs cell by cell
vllm-bench compare results/rtx3060-* results/cpu-12c-* --metric ttft_p50_ms
vllm-bench compare results/rtx3060-* results/cpu-12c-* --metric output_tok_per_s
```

One campaign file is the grid for every model and both devices. GPU means a vast.ai machine: rent the GPU, start vLLM with the model id from the file, then set `gpu.base_url` and `gpu.gpu_type` in `configs/campaign.yaml`. The CPU URL goes in the same file. Do not pass `--concurrency`, `--input-tokens`, or `--output-tokens` for a campaign run; those three lists live only there.

```bash
vllm-bench run --campaign configs/campaign.yaml --device gpu \
  --model Qwen/Qwen2.5-1.5B-Instruct
vllm-bench run --campaign configs/campaign.yaml --device cpu \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

The label is `<model>-gpu` or `<model>-cpu`, so the two runs pair up later.

Everything in a single-run YAML can be overridden from the CLI:

```bash
vllm-bench run --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-1.5B-Instruct \
  --concurrency 1,2,4,8,16 --input-tokens 128,512,2048 --output-tokens 64,256 \
  --requests-per-user 4 --warmup 2 --label gpu
```

Open the dashboard (localhost only; API keys stay in the summary file):

```bash
vllm-bench ui
# http://127.0.0.1:8765/
```

`results/` is listed first and shadows an example with the same folder name. `examples/` holds a sanitized copy of a published GPU grid so the page has something to show before a local sweep.

Try the tool without a model:

```bash
vllm-bench mock-server --port 8321 &        # fake OpenAI server, 5 ms/token
vllm-bench run --base-url http://127.0.0.1:8321/v1 --concurrency 1,4,16 \
  --input-tokens 128,1024 --output-tokens 32 --requests-per-user 2 --label mock
```

## How it works

1. **Model discovery**: if `--model` is omitted the first entry of `/v1/models` is used.
2. **Token calibration**: the server's tokenizer is unknown, so two probe requests
   (`max_tokens=1`) read back `usage.prompt_tokens` and fit
   `tokens = overhead + tokens_per_word * words`. Prompts are then sized to hit
   the target input length within a few tokens for *any* model. Each prompt
   starts with a random nonce so prefix caching cannot hide prefill cost.
3. **Exact output length**: requests send `max_tokens`, and for vLLM also
   `ignore_eos: true` + `min_tokens` so every request generates exactly the
   target number of tokens. Use `--no-ignore-eos` for servers that reject
   unknown fields.
4. **Closed-loop load**: for a cell with `C` users, `C` coroutines each send
   requests back-to-back (no think time) until the cell's request budget
   (`requests_per_user * C`, or `requests_per_cell`, or `duration_s`) is used up.
   Warmup requests are discarded.
5. **Streaming measurement** (`stream: true`, `stream_options.include_usage`):

   | metric | meaning |
   |---|---|
   | TTFT | request sent → first content chunk (queueing + prefill) |
   | TPOT | per request, mean gap between consecutive chunks (decode speed) |
   | ITL  | all inter-chunk gaps pooled, useful for p99 jitter |
   | E2E  | request sent → stream finished |
   | out tok/s | total completion tokens / cell wall time (server throughput) |
   | req/s | completed requests / cell wall time |
   | per_user_output_tok_per_s | what one user perceives |

   Percentiles reported: mean, p50, p90, p99.

6. **Outputs** in `results/<label>-<timestamp>/`:
   `requests.jsonl` (every request, incl. the raw ITL list), `summary.csv`,
   `summary.json` (config, calibration, host/GPU info, all stats),
   `REPORT.md` (device, dtype, max context, launch flags, goodput, and failure counts), and `*.png` charts of each metric vs concurrency and, when the run has more than one context length, vs context length.

## Vision-language models

Set `endpoint.image_path` (or `--image path.jpg`) and the same image is attached
to every request as a base64 `image_url` part. The text prompt is still
calibrated to the target length; the server adds the image tokens on top, and
the achieved `actual_input_tokens_mean` in the report shows the real total.
`configs/vlm.yaml` uses `assets/sample.jpg`.

## Launching vLLM to benchmark

GPU (RTX 3060, 12 GB):

```bash
docker run --gpus all -p 8000:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.9
```

CPU (vLLM CPU backend image):

```bash
docker run -p 8000:8000 --ipc=host \
  -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_OMP_THREADS_BIND=0-11 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct --dtype bfloat16 --max-model-len 4096
```

VLM example (GPU):

```bash
docker run --gpus all -p 8000:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2-VL-2B-Instruct --max-model-len 8192 --limit-mm-per-prompt image=1
vllm-bench run -c configs/vlm.yaml
```

Any other OpenAI-compatible server works the same way; just point `--base-url`
at its `/v1` and add `--no-ignore-eos` if it rejects vLLM-specific fields.

## Tips for fair CPU vs GPU comparisons

- Same model, same dtype where possible, same `max-model-len`.
- `configs/gpu.yaml` and `configs/cpu.yaml` list the same users, context lengths, and output lengths as `configs/campaign.yaml`. Edit that grid in the campaign file.
- A CPU cell may time out. That failure is recorded. It is not a reason to shrink the grid first.
- Watch the `ok/requests` column: timeouts or OOM show up as failures, and the
  first error message is kept in `summary.json`.
- Run the client on a different machine than a CPU server, or pin the server's
  threads, so the benchmark client does not steal cores.

## Development

```bash
pytest -q          # unit + end-to-end tests against the built-in mock server
```

Layout:

```
vllm_bench/
  cli.py          argparse entry point: run / mock-server / compare
  config.py       YAML + CLI config dataclasses
  prompts.py      calibrated prompt generation
  client.py       async streaming OpenAI client + per-request metrics
  runner.py       closed-loop sweep runner
  metrics.py      percentiles, per-cell summaries
  report.py       JSONL/CSV/JSON/Markdown/PNG writers
  compare.py      multi-run comparison table
  dashboard.py    localhost UI; strips API keys before the page sees a run
  mock_server.py  stdlib fake OpenAI server
  static/         dashboard HTML, CSS, and JS
configs/          campaign.yaml cpu.yaml gpu.yaml vlm.yaml smoke.yaml
examples/         sanitized summary.json files for the dashboard
```
