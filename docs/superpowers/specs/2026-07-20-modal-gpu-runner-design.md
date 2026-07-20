# Modal GPU Runner Design

## Goal

Provide one lightweight launcher that runs any Python script in this repository on a user-selected Modal GPU.

## Interface

The launcher lives at `scripts/modal_gpu.py` and is invoked with `modal run`. It accepts a GPU name (default `A100-80GB`), a timeout, an optional local output path, and then a required repository-relative Python script under `scripts/`. The script path is the CLI boundary: every token after it is forwarded unchanged to the target script, so launcher options must precede it.

Example:

```bash
uvx modal run scripts/modal_gpu.py \
  --gpu A100-80GB \
  --output attention_benchmark_results.txt \
  scripts/attention_benchmark.py \
  --device cuda --warmup-steps 5 --measurement-steps 100
```

## Architecture

The variadic local entrypoint parses its own arguments with `argparse`, validates the script path, and creates a Modal Sandbox using the requested GPU. A cached Python 3.12 image runs `uv_sync(frozen=True, extra_options="--no-install-package cs336-basics")` for third-party dependencies. Only `scripts/`, `cs336_systems/`, and `cs336-basics/cs336_basics/` are mounted into `/workspace`; the image sets `PYTHONPATH=/workspace:/workspace/cs336-basics` and executes with the environment's `python` from `/workspace`.

The launcher calls `Sandbox.exec("python", script, *args, workdir="/workspace", ...)` directly, with no shell interpolation. Stdout and stderr remain separate pipes and are drained concurrently by two local threads to prevent blocking. Each chunk is written immediately to its matching terminal stream and to a shared, locked output file when requested. Cross-stream ordering is best-effort; ordering within each stream is preserved. Partial output survives failures without unbounded in-memory buffering.

No Modal Volume is used. Source code is bundled into the image and small benchmark results return through process output.

## Failure Handling

Before GPU allocation, the launcher resolves the requested path strictly against the fixed repository root and requires it to remain within `scripts/`, be a regular file, and end in `.py`; this also rejects symlinks that escape the directory. The output path is opened before allocation.

Unsupported GPU names and image-build/allocation errors surface directly. Timeout is validated in the range 1 to 86,400 seconds and is applied to both the Sandbox lifetime and remote process. A remote nonzero status becomes the local launcher status, while timeout or interruption also fails locally after retaining partial output. Once created, the Sandbox is terminated with `terminate(wait=True)` in a `finally` block; cleanup errors do not replace an active execution error.

## Verification

Unit tests mock Modal's Sandbox/process interfaces and cover argument splitting, command construction, path rejection, creation failure, timeout, nonzero exit, partial output, and cleanup. A real Modal smoke run on a low-cost GPU verifies authentication, image creation, CUDA availability, output forwarding, and termination. The attention benchmark then runs on one consistent A100 configuration for comparable timings and OOM results.
