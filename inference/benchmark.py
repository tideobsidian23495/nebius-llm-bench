#!/usr/bin/env python3
"""Run reproducible batched LLM inference benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Iterable

try:
    import torch
except ImportError:
    torch = None


DEFAULT_PROMPTS = [
    "Nebius provides accelerated infrastructure for",
    "Large language model inference at scale requires",
    "The best GPU benchmark measures",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face causal LLM inference")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Hugging Face model ID")
    parser.add_argument("--model-size", default="tiny", help="Human-readable model-size label")
    parser.add_argument("--quantization", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if torch is None:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id: str, quantization: str, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    kwargs = {"trust_remote_code": False}
    if quantization == "int8":
        kwargs.update(load_in_8bit=True, device_map="auto")
    elif quantization == "fp16" and device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        if model_id == "sshleifer/tiny-gpt2":
            raise
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", **kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if quantization != "int8":
        model.to(device)
    model.eval()
    return tokenizer, model


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.batch_size < 1 or args.runs < 1 or args.warmup < 0:
        raise ValueError("batch size and runs must be positive; warmup must be non-negative")

    device = choose_device(args.device)
    tokenizer, model = load_model(args.model, args.quantization, device)
    prompts = (DEFAULT_PROMPTS * ((args.batch_size // len(DEFAULT_PROMPTS)) + 1))[: args.batch_size]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(args.warmup):
        with torch.inference_mode():
            model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)

    latencies_ms = []
    throughputs = []
    for _ in range(args.runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        prompt_length = encoded["input_ids"].shape[1]
        generated_tokens = max(output.shape[1] - prompt_length, 1) * args.batch_size
        total_per_token_ms = elapsed * 1000.0 / generated_tokens
        latencies_ms.append(total_per_token_ms)
        throughputs.append(generated_tokens / elapsed)

    peak_gpu_mb = None
    if device.type == "cuda":
        peak_gpu_mb = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)

    return {
        "metadata": {
            "data_type": "measured",
            "model": args.model,
            "model_size": args.model_size,
            "quantization": args.quantization,
            "batch_size": args.batch_size,
            "runs": args.runs,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU",
            "torch_version": torch.__version__,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": {
            "throughput_tokens_per_sec": round(statistics.median(throughputs), 2),
            "latency_ms_per_token": round(statistics.median(latencies_ms), 3),
            "latency_p50_ms": round(percentile(latencies_ms, 50), 3),
            "latency_p90_ms": round(percentile(latencies_ms, 90), 3),
            "latency_p99_ms": round(percentile(latencies_ms, 99), 3),
            "peak_gpu_memory_mb": peak_gpu_mb,
        },
        "runs": [
            {
                "throughput_tokens_per_sec": round(throughput, 2),
                "latency_ms_per_token": round(latency, 3),
            }
            for throughput, latency in zip(throughputs, latencies_ms)
        ],
    }


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote benchmark result to {args.output}")


if __name__ == "__main__":
    main()
