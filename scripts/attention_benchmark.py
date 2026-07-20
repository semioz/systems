from __future__ import annotations

import argparse
import gc
import json
import statistics
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention


D_K_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]


def benchmark_attention(
    *,
    batch_size: int,
    seq_len: int,
    d_k: int,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(0)
    Q = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    do = torch.randn(batch_size, seq_len, d_k, device=device)

    for _ in range(warmup_steps):
        Q.grad = K.grad = V.grad = None
        out = scaled_dot_product_attention(Q, K, V, mask=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        out.backward(do)
        if device.type == "cuda":
            torch.cuda.synchronize()

    forward_timings = []
    backward_timings = []
    memory_before_backward = []
    for _ in range(measurement_steps):
        Q.grad = K.grad = V.grad = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = timeit.default_timer()
        out = scaled_dot_product_attention(Q, K, V, mask=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_timings.append(timeit.default_timer() - start)

        memory_before_backward.append(torch.cuda.memory_allocated(device) if device.type == "cuda" else 0)

        start = timeit.default_timer()
        out.backward(do)
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_timings.append(timeit.default_timer() - start)

    return {
        "seq_len": seq_len,
        "d_k": d_k,
        "forward_mean_seconds": statistics.fmean(forward_timings),
        "backward_mean_seconds": statistics.fmean(backward_timings),
        "memory_before_backward_bytes": max(memory_before_backward),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark naive scaled dot product attention (no heads).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    results = []
    for d_k in D_K_VALUES:
        for seq_len in SEQ_LEN_VALUES:
            print(f"Running d_k={d_k}, seq_len={seq_len}...", flush=True)
            try:
                result = benchmark_attention(
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    d_k=d_k,
                    warmup_steps=args.warmup_steps,
                    measurement_steps=args.measurement_steps,
                    device=device,
                )
                result["status"] = "ok"
            except torch.OutOfMemoryError:
                result = {"seq_len": seq_len, "d_k": d_k, "status": "oom"}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            results.append(result)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output = {
        "batch_size": args.batch_size,
        "device": str(device),
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
