import torch

import scripts.attention_benchmark as attention_benchmark
from scripts.attention_benchmark import (
    benchmark_attention,
    cuda_device_index,
    implementation_names,
    run_benchmark_sweep,
)


def test_cuda_device_index_defaults_to_zero() -> None:
    assert cuda_device_index(torch.device("cuda")) == 0


def test_cuda_device_index_preserves_explicit_index() -> None:
    assert cuda_device_index(torch.device("cuda:2")) == 2


def test_implementation_names() -> None:
    assert implementation_names("eager") == ("eager",)
    assert implementation_names("compiled") == ("compiled",)
    assert implementation_names("both") == ("eager", "compiled")


def test_benchmark_uses_supplied_attention_function() -> None:
    calls = 0

    def attention_fn(Q, K, V, mask=None):
        nonlocal calls
        calls += 1
        return attention_benchmark.scaled_dot_product_attention(Q, K, V, mask)

    benchmark_attention(
        attention_fn=attention_fn,
        batch_size=2,
        seq_len=8,
        d_k=4,
        warmup_steps=1,
        measurement_steps=2,
        device=torch.device("cpu"),
    )

    assert calls == 3


def test_compiled_sweep_uses_fullgraph_and_resets_each_shape(monkeypatch) -> None:
    compile_calls = []
    reset_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return object()

    def fake_benchmark(**kwargs):
        return {
            "seq_len": kwargs["seq_len"],
            "d_k": kwargs["d_k"],
            "forward_mean_seconds": 1.0,
            "backward_mean_seconds": 2.0,
            "memory_before_backward_bytes": 0,
        }

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.compiler, "reset", lambda: reset_calls.append(None))
    monkeypatch.setattr(attention_benchmark, "benchmark_attention", fake_benchmark)

    results = run_benchmark_sweep(
        batch_size=8,
        warmup_steps=1,
        measurement_steps=1,
        device=torch.device("cpu"),
        implementation="compiled",
        d_k_values=(16, 32),
        seq_len_values=(256, 1024),
    )

    assert len(results) == 4
    assert all(result["implementation"] == "compiled" for result in results)
    assert all(kwargs == {"fullgraph": True} for _, kwargs in compile_calls)
    assert len(reset_calls) == 8


def test_both_sweep_emits_40_labeled_results(monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.compiler, "reset", lambda: None)
    monkeypatch.setattr(
        attention_benchmark,
        "benchmark_attention",
        lambda **kwargs: {
            "seq_len": kwargs["seq_len"],
            "d_k": kwargs["d_k"],
            "forward_mean_seconds": 1.0,
            "backward_mean_seconds": 2.0,
            "memory_before_backward_bytes": 0,
        },
    )

    results = run_benchmark_sweep(
        batch_size=8,
        warmup_steps=1,
        measurement_steps=1,
        device=torch.device("cpu"),
        implementation="both",
    )

    assert len(results) == 40
    assert {result["implementation"] for result in results} == {"eager", "compiled"}


def test_compiled_oom_does_not_stop_later_results(monkeypatch) -> None:
    compiled_fn = object()

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compiled_fn)
    monkeypatch.setattr(torch.compiler, "reset", lambda: None)

    def fake_benchmark(**kwargs):
        if kwargs["attention_fn"] is compiled_fn and kwargs["seq_len"] == 256:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return {
            "seq_len": kwargs["seq_len"],
            "d_k": kwargs["d_k"],
            "forward_mean_seconds": 1.0,
            "backward_mean_seconds": 2.0,
            "memory_before_backward_bytes": 0,
        }

    monkeypatch.setattr(attention_benchmark, "benchmark_attention", fake_benchmark)

    results = run_benchmark_sweep(
        batch_size=8,
        warmup_steps=1,
        measurement_steps=1,
        device=torch.device("cpu"),
        implementation="both",
        d_k_values=(16,),
        seq_len_values=(256, 1024),
    )

    assert [result["status"] for result in results] == ["ok", "oom", "ok", "ok"]
