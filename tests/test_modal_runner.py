from pathlib import Path

import pytest

from cs336_systems.modal_runner import build_remote_command, parse_runner_args


def test_parse_runner_args_preserves_target_arguments() -> None:
    config = parse_runner_args(
        [
            "--gpu",
            "A100-80GB",
            "--timeout",
            "900",
            "scripts/attention_benchmark.py",
            "--device",
            "cuda",
            "--measurement-steps",
            "100",
        ]
    )

    assert config.gpu == "A100-80GB"
    assert config.timeout == 900
    assert config.script == Path("scripts/attention_benchmark.py")
    assert config.script_args == ("--device", "cuda", "--measurement-steps", "100")


def test_parse_runner_args_defaults_to_a100_80gb() -> None:
    config = parse_runner_args(["scripts/attention_benchmark.py"])

    assert config.gpu == "A100-80GB"
    assert config.timeout == 7200


def test_parse_runner_args_rejects_script_outside_scripts_directory() -> None:
    with pytest.raises(SystemExit):
        parse_runner_args(["cs336_systems/benchmarking.py"])


def test_build_remote_command_uses_root_script_path() -> None:
    config = parse_runner_args(
        ["scripts/attention_benchmark.py", "--device", "cuda"]
    )

    assert build_remote_command(config) == [
        "python",
        "/root/scripts/attention_benchmark.py",
        "--device",
        "cuda",
    ]


def test_modal_entrypoint_enables_remote_output_streaming() -> None:
    source = Path("scripts/modal_gpu.py").read_text()

    assert "with modal.enable_output(), app.run():" in source
