from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
import json
import statistics
import timeit
from enum import StrEnum
from typing import Any

import torch

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
from cs336_systems.model_configs import DEFAULT_BATCH_SIZE, DEFAULT_CONTEXT_LENGTH, DEFAULT_VOCAB_SIZE, MODEL_CONFIGS


class BenchmarkMode(StrEnum):
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    TRAIN_STEP = "train_step"


def resolve_model_config(
    *,
    model_size: str,
    vocab_size: int | None = None,
    context_length: int | None = None,
    d_model: int | None = None,
    d_ff: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
) -> dict[str, int]:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}")

    config = dict(MODEL_CONFIGS[model_size])
    overrides = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }

    config.setdefault("vocab_size", DEFAULT_VOCAB_SIZE)
    config.setdefault("context_length", DEFAULT_CONTEXT_LENGTH)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nvtx_range(name: str, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def annotated_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    device = Q.device
    d_k = K.shape[-1]

    with nvtx_range("scaled dot product attention", device):
        with nvtx_range("computing attention scores", device):
            attention_scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx_range("computing softmax", device):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx_range("final matmul", device):
            return torch.einsum("...qk,...kd->...qd", attention_weights, V)


def run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: BenchmarkMode,
    mixed_precision: bool = False,
) -> None:
    device = inputs.device
    precision_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_precision else nullcontext()

    if mode == BenchmarkMode.FORWARD:
        with torch.no_grad(), nvtx_range("forward", device), precision_context:
            model(inputs)
        return

    optimizer.zero_grad(set_to_none=True)
    with nvtx_range("forward", device), precision_context:
        logits = model(inputs)

    loss = cross_entropy(logits, targets)
    with nvtx_range("backward", device):
        loss.backward()

    if mode == BenchmarkMode.TRAIN_STEP:
        with nvtx_range("optimizer_step", device):
            optimizer.step()


def benchmark_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: BenchmarkMode,
    warmup_steps: int,
    measurement_steps: int,
    mixed_precision: bool = False,
) -> dict[str, Any]:
    device = inputs.device

    with nvtx_range("warmup", device):
        for _ in range(warmup_steps):
            run_step(model, optimizer, inputs, targets, mode, mixed_precision=mixed_precision)
            synchronize(device)

    timings = []
    with nvtx_range("measurement", device):
        for _ in range(measurement_steps):
            start = timeit.default_timer()
            run_step(model, optimizer, inputs, targets, mode, mixed_precision=mixed_precision)
            synchronize(device)
            timings.append(timeit.default_timer() - start)

    return {
        "mode": mode.value,
        "mixed_precision": mixed_precision,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "mean_seconds": statistics.fmean(timings),
        "std_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "timings_seconds": timings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward/backward/training steps.")
    parser.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), default="small")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--mode", choices=[mode.value for mode in BenchmarkMode], default=BenchmarkMode.FORWARD.value)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true", help="Run model forward under BF16 autocast.")
    parser.add_argument("--annotate-attention", action="store_true", help="Add NVTX ranges inside scaled dot-product attention for Nsight profiling.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    if args.annotate_attention:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    device = torch.device(args.device)
    config = resolve_model_config(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    model = BasicsTransformerLM(**config).to(device)
    optimizer = AdamW(model.parameters())
    inputs = torch.randint(0, config["vocab_size"], (args.batch_size, config["context_length"]), device=device)
    targets = torch.randint(0, config["vocab_size"], (args.batch_size, config["context_length"]), device=device)

    results = benchmark_steps(model, optimizer, inputs, targets, BenchmarkMode(args.mode), args.warmup_steps, args.measurement_steps, mixed_precision=args.mixed_precision)
    results["model_config"] = config
    results["batch_size"] = args.batch_size
    results["device"] = str(device)

    print(json.dumps(results, indent=2))
