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
IMPLEMENTATIONS = ("eager", "compiled")


def cuda_device_index(device: torch.device) -> int:
    return 0 if device.index is None else device.index


def implementation_names(selection: str) -> tuple[str, ...]:
    return IMPLEMENTATIONS if selection == "both" else (selection,)


def is_oom_error(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.OutOfMemoryError) or "out of memory" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def benchmark_attention(
    *,
    batch_size: int,
    seq_len: int,
    d_k: int,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
    attention_fn=scaled_dot_product_attention,
) -> dict:
    torch.manual_seed(0)
    Q = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    do = torch.randn(batch_size, seq_len, d_k, device=device)

    for _ in range(warmup_steps):
        Q.grad = K.grad = V.grad = None
        out = attention_fn(Q, K, V, mask=None)
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
            torch.cuda.synchronize()

        start = timeit.default_timer()
        out = attention_fn(Q, K, V, mask=None)
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


def run_benchmark_sweep(
    *,
    batch_size: int,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
    implementation: str,
    d_k_values: tuple[int, ...] | list[int] = D_K_VALUES,
    seq_len_values: tuple[int, ...] | list[int] = SEQ_LEN_VALUES,
) -> list[dict]:
    results = []
    for d_k in d_k_values:
        for seq_len in seq_len_values:
            for implementation_name in implementation_names(implementation):
                print(f"Running {implementation_name}: d_k={d_k}, seq_len={seq_len}...", flush=True)
                compiled = implementation_name == "compiled"
                try:
                    if compiled:
                        torch.compiler.reset()
                        attention_fn = torch.compile(scaled_dot_product_attention, fullgraph=True)
                    else:
                        attention_fn = scaled_dot_product_attention

                    result = benchmark_attention(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        d_k=d_k,
                        warmup_steps=warmup_steps,
                        measurement_steps=measurement_steps,
                        device=device,
                        attention_fn=attention_fn,
                    )
                    result.update({"implementation": implementation_name, "status": "ok"})
                except Exception as error:
                    result = {
                        "seq_len": seq_len,
                        "d_k": d_k,
                        "implementation": implementation_name,
                        "status": "oom" if is_oom_error(error) else "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                finally:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if compiled:
                        torch.compiler.reset()
                results.append(result)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark naive scaled dot product attention (no heads).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--implementation", choices=("eager", "compiled", "both"), default="both")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(cuda_device_index(device))

    results = run_benchmark_sweep(
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=device,
        implementation=args.implementation,
    )

    output = {
        "batch_size": args.batch_size,
        "device": str(device),
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "implementation": args.implementation,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
