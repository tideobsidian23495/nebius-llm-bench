# LLM Inference Benchmark & Serving Lab

A small, practical benchmark harness for studying Hugging Face LLM inference across model sizes and quantization levels. It records throughput, per-token latency percentiles, and peak GPU memory, with a dark static dashboard and a one-command Nebius-ready container workflow.

> **Sample data disclaimer:** `results/benchmark.json` contains clearly labeled synthetic demonstration values. It is not measured performance and must not be presented as a real benchmark.

## Why GPU

LLM serving has to move large model weights and activation tensors every step. GPU memory bandwidth, tensor cores, and batched kernel execution materially determine serving throughput and tail latency. This lab lets an applicant quantify those effects on representative Nebius GPU instances rather than relying on CPU anecdotes or vendor claims. Credits enable comparative runs across model sizes and fp16/int8 quantization, where results can vary sharply by GPU class and batch size.

## Features

- Hugging Face `Transformers` model loading with an explicit tiny-model default
- Batch inference and warmup runs
- Median throughput and p50/p90/p99 per-token latency
- CUDA peak allocated memory
- `fp32`, `fp16`, and `int8` CLI modes
- Static Chart.js dashboard with no build step
- Docker and GitHub Pages deployment

## Requirements

- Python 3.11+
- PyTorch
- NVIDIA GPU optional; CPU works for small/tiny models
- Docker + NVIDIA Container Toolkit for `./run.sh --gpus all`

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default model is `sshleifer/tiny-gpt2`, a compact CPU-friendly Hugging Face model. A custom model that fails to load falls back to this tiny model. On GPU, use a larger model ID and quantization flags from the CLI.

## Run

```bash
# Tiny CPU-friendly verification
python inference/benchmark.py --device cpu --runs 3 --warmup 1 --batch-size 2

# GPU fp16
python inference/benchmark.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --model-size 1B \
  --quantization fp16 \
  --device cuda \
  --batch-size 8 \
  --runs 10

# GPU int8
python inference/benchmark.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --model-size 1B \
  --quantization int8 \
  --device cuda \
  --batch-size 8 \
  --runs 10
```

Results are written to `results/benchmark.json`. The dashboard can be served locally with:

```bash
python3 -m http.server 8080
# Open http://localhost:8080/dashboard/
```

## Docker on a Nebius GPU instance

```bash
chmod +x run.sh
./run.sh --model meta-llama/Llama-3.2-1B-Instruct \
  --model-size 1B --quantization fp16 --device cuda \
  --batch-size 8 --runs 10
```

`run.sh` builds the image, mounts `./results`, and requests all GPUs. Override the host port with `PORT=9090 ./run.sh`.

## Result format

The JSON contains metadata, median summary metrics, raw measured runs, and a `"benchmarks"` series when comparing configurations. Real CLI runs are stamped `"data_type": "measured"`; replace or remove the shipped sample before presenting a measured campaign.

## Repository layout

- `inference/benchmark.py` — benchmark harness
- `dashboard/` — static Chart.js dashboard
- `results/benchmark.json` — sample or generated benchmark data
- `docs/` — GitHub Pages landing page and embedded demo
- `.github/workflows/pages.yml` — Pages deployment
