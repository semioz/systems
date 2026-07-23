from __future__ import annotations

import subprocess
import sys
import os


raise SystemExit(
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            "-s",
            "tests/test_attention.py",
            "-k",
            "test_flash_forward_pass_triton",
        ],
        cwd="/root",
        check=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ).returncode
)
