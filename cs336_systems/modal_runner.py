from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunnerConfig:
    gpu: str
    timeout: int
    script: Path
    script_args: tuple[str, ...]


def parse_runner_args(argv: list[str] | None = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Run a repository script on a Modal GPU.")
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("script", type=Path, help="Python script under scripts/")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.script.suffix != ".py" or args.script.parent != Path("scripts"):
        parser.error("script must be a Python file directly under scripts/")

    return RunnerConfig(
        gpu=args.gpu,
        timeout=args.timeout,
        script=args.script,
        script_args=tuple(args.script_args),
    )


def build_remote_command(config: RunnerConfig) -> list[str]:
    return ["python", f"/root/{config.script.as_posix()}", *config.script_args]
