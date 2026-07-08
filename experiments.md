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
