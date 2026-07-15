# Assignment 2 Experiments

## 2.1.3 End-to-End Benchmarking

Setup:
- GPU: Modal B200
- Batch size: 4
- Context length: 512
- Vocab size: 10,000
- Measurement steps: 10
- Baseline warmup steps: 5
- Model: `cs336_basics.model.BasicsTransformerLM` from the provided `cs336-basics` package

Scripts/files:
- Benchmark script: `scripts/benchmark.py`
- Modal runner: `scripts/modal_benchmark.py`
- Raw outputs: `benchmark_small_to_xl.txt`, `benchmark_10b.txt`, `benchmark_no_warmup.txt`, `benchmark_1_warmup.txt`, `benchmark_2_warmup.txt`
- CSV summary: `benchmark_results.csv`

### Part (b): 5 warmup steps

Forward and backward timings with 5 warmup steps and 10 measured steps:

| size | forward mean (s) | forward std (s) | forward+backward mean (s) | backward estimate (s) | train step mean (s) | optimizer estimate (s) |
|---|---:|---:|---:|---:|---:|---:|
| small | 0.016467 | 0.000023 | 0.049401 | 0.032935 | 0.056596 | 0.007195 |
| medium | 0.047008 | 0.000033 | 0.140251 | 0.093243 | 0.160285 | 0.020034 |
| large | 0.106030 | 0.000031 | 0.314072 | 0.208042 | 0.351535 | 0.037463 |
| xl | 0.295591 | 0.000186 | 0.865023 | 0.569432 | 0.944706 | 0.079683 |
| 10B | 0.944138 | 0.000524 | 2.809717 | 1.865579 | OOM | OOM |

Writeup draft:

> On a B200 GPU with batch size 4 and context length 512, forward pass times for small/medium/large/xl/10B were 0.0165s, 0.0470s, 0.1060s, 0.2956s, and 0.9441s; estimated backward times were 0.0329s, 0.0932s, 0.2080s, 0.5694s, and 1.8656s. Standard deviations were small relative to the means, indicating low variability; the 10B full training step ran out of memory during the AdamW optimizer step.

Notes:
- Backward estimate is `forward_backward_mean - forward_mean`.
- Optimizer estimate is `train_step_mean - forward_backward_mean`.
- 10B `train_step` OOMed during AdamW optimizer state allocation/update in `cs336_basics/optimizer.py`.

### Part (c): warmup effect

No-warmup summary:

| size | forward mean (s) | forward std (s) | backward estimate (s) | train step mean (s) |
|---|---:|---:|---:|---:|
| small | 0.059935 | 0.137349 | 0.086425 | 0.152889 |
| medium | 0.087619 | 0.128535 | 0.141663 | 0.254866 |
| large | 0.149083 | 0.135891 | 0.250219 | 0.438702 |
| xl | 0.336971 | 0.124035 | 0.613277 | 1.032298 |

1-warmup summary:

| size | forward mean (s) | forward std (s) | backward estimate (s) | train step mean (s) |
|---|---:|---:|---:|---:|
| small | 0.016329 | 0.000064 | 0.032514 | 0.057444 |
| medium | 0.046895 | 0.000135 | 0.093323 | 0.157518 |
| large | 0.106062 | 0.000024 | 0.208141 | 0.344278 |
| xl | 0.293102 | 0.000643 | 0.569525 | 0.942397 |

2-warmup summary:

| size | forward mean (s) | forward std (s) | backward estimate (s) | train step mean (s) |
|---|---:|---:|---:|---:|
| small | 0.016327 | 0.000014 | 0.032492 | 0.056427 |
| medium | 0.046947 | 0.000064 | 0.093294 | 0.157019 |
| large | 0.106490 | 0.000045 | 0.208919 | 0.345619 |
| xl | 0.295964 | 0.000816 | 0.571503 | 0.946053 |

Writeup draft:

> Without warmup, the first measured iteration includes CUDA/PyTorch initialization, allocator setup, and optimizer-state allocation effects, so the mean timings become inflated and the standard deviations become much larger. With 1–2 warmup steps, the results become close to the 5-warmup steady-state numbers, but they can still differ slightly because not all kernels, memory allocator state, and optimizer/cache behavior are fully stabilized after only a small number of warmup iterations.

## 2.1.4 Nsight Systems Profiling

Setup:
- GPU: RunPod A100 SXM
- Batch size: 4
- Vocab size: 10,000
- Model sizes: `small`, `medium`
- Context lengths profiled: 256, 512, 1024; additionally 2048 for `small` forward/train-step and `medium` forward
- Nsight Systems: `nsys profile --sample=none --cpuctxsw=none --trace=cuda,cudnn,cublas,osrt,nvtx`
- Notes: `--gpu-metrics-devices=0` was unavailable due `ERR_NVGPUCTRPERM`; `--pytorch=...` was unavailable in Nsight Systems 2024.6.2. Manual NVTX ranges were used for `warmup`, `measurement`, `forward`, `backward`, `optimizer_step`, and attention sub-operations.

Scripts/files:
- Benchmark script: `scripts/benchmark.py`
- Profiles: `runpod_outputs/profiles/*.nsys-rep`
- Stats summaries: `runpod_outputs/profile_stats/*.txt`

Forward pass measured NVTX range times:

| size | ctx | forward time from nsys `measurement` range (ms) |
|---|---:|---:|
| small | 256 | 25.874 |
| small | 512 | 43.709 |
| small | 1024 | 98.361 |
| small | 2048 | 256.877 |
| medium | 256 | 64.073 |
| medium | 512 | 132.155 |
| medium | 1024 | 293.860 |
| medium | 2048 | 753.133 |

Top cumulative CUDA kernels in forward profiles. Instance counts below divide total profile counts by 6 because the profile includes 5 warmup forwards plus 1 measured forward.

| size | ctx | top forward CUDA kernel | invocations / forward |
|---|---:|---|---:|
| small | 256 | `ampere_sgemm_32x128_tn` | 60 |
| small | 512 | `ampere_sgemm_128x32_tn` | 24 |
| small | 1024 | `ampere_sgemm_128x64_tn` | 25 |
| small | 2048 | `ampere_sgemm_128x64_tn` | 25 |
| medium | 256 | `ampere_sgemm_128x64_tn` | 169 |
| medium | 512 | `ampere_sgemm_128x128_tn` | 144 |
| medium | 1024 | `ampere_sgemm_128x64_tn` | 145 |
| medium | 2048 | `ampere_sgemm_128x32_tn` | 120 |

Representative attention NVTX timings from forward profiles:

| size | ctx | attention-score matmul total (ms) | softmax total (ms) | final attention matmul total (ms) |
|---|---:|---:|---:|---:|
| small | 512 | 47.970 | 38.778 | 17.610 |
| medium | 512 | 56.041 | 44.001 | 24.646 |
| small | 2048 | 46.213 | 38.727 | 18.691 |
| medium | 2048 | 120.933 | 218.509 | 26.054 |

Train-step profiles:

| size | ctx | train-step `measurement` range (ms) | top cumulative CUDA kernel |
|---|---:|---:|---|
| small | 256 | 113.793 | `ampere_sgemm_64x32_sliced1x4_nt` |
| small | 512 | 149.625 | `ampere_sgemm_64x32_sliced1x4_nt` |
| small | 1024 | 326.685 | `ampere_sgemm_64x32_sliced1x4_nt` |
| small | 2048 | 833.748 | `ampere_sgemm_128x128_nt` |
| medium | 256 | 248.012 | `ampere_sgemm_128x64_tn` |
| medium | 512 | 455.607 | `ampere_sgemm_128x64_nn` |
| medium | 1024 | 965.841 | `ampere_sgemm_128x64_tn` |

Notes for writeup draft:
- For part (a), the nsys `measurement` range times match the benchmark script timings closely (e.g. small/ctx512 was about 43.7 ms in nsys vs. 43.2 ms from the script output; medium/ctx512 was about 132.2 ms vs. 130.9 ms).
- For part (b), the top forward kernels are SGEMM matrix-multiplication kernels; in full train-step profiles the top cumulative kernels often change to backward/gradient GEMM variants, so the top kernel is not always the same as forward-only.
- For part (c), non-matmul kernels with non-trivial runtime include PyTorch elementwise kernels, vectorized elementwise kernels, reductions for RMSNorm/softmax, `exp`, `sigmoid`, `rsqrt`, `pow`, cat/copy, gather/scatter, and arange/indexing kernels.
- For part (d), full train-step profiles contain many more non-matmul elementwise/reduction kernels from backward and AdamW, so the runtime is less purely dominated by forward GEMMs than inference-only.
- For part (e), softmax is far fewer FLOPs than the attention matrix multiplications but can take a comparable or even larger amount of wall-clock time at long contexts because it is memory/reduction heavy rather than dense-matmul compute heavy.

## 2.1.5 Mixed Precision

### Problem (mixed_precision_accumulation)

Script:
- `scripts/mixed_precision_accumulation.py`

Output:

```text
float32 += float32: tensor(10.0001) torch.float32
float16 += float16: tensor(9.9531, dtype=torch.float16) torch.float16
float32 += float16: tensor(10.0021) torch.float32
float32 += float16 cast to float32: tensor(10.0021) torch.float32
```

Writeup draft:

> The exact result should be 10.0. Accumulating entirely in FP32 gives 10.0001, while accumulating in FP16 gives 9.9531 because FP16 has much lower precision and repeatedly rounds the partial sum. Keeping the accumulator in FP32 improves stability, but when each added value is first represented as FP16 the quantization error of 0.01 remains, giving about 10.0021.
