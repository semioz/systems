from __future__ import annotations

import subprocess

import modal

from cs336_systems.modal_runner import RunnerConfig, build_remote_command, parse_runner_args


app = modal.App("cs336-systems-gpu")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "einops>=0.8",
        "einx>=0.4",
        "humanfriendly",
        "jaxtyping>=0.3",
        "matplotlib",
        "numpy>=2",
        "pandas>=2",
        "pytest>=8",
        "regex",
        "tqdm",
        "wandb",
    )
    .workdir("/root")
    .add_local_dir("cs336_systems", "/root/cs336_systems")
    .add_local_dir("cs336-basics/cs336_basics", "/root/cs336_basics")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("tests", "/root/tests")
)


def run_remote(script: str, script_args: list[str]) -> int:
    command = ["python", f"/root/{script}", *script_args]
    print(f"running: {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd="/root", check=False).returncode


def make_gpu_function(config: RunnerConfig):
    return app.function(image=image, gpu=config.gpu, timeout=config.timeout)(run_remote)


def main() -> None:
    config = parse_runner_args()
    command = build_remote_command(config)
    print(f"using gpu={config.gpu}")
    print(f"running: {' '.join(command)}")

    remote_run = make_gpu_function(config)
    with modal.enable_output(), app.run():
        returncode = remote_run.remote(config.script.as_posix(), list(config.script_args))

    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
