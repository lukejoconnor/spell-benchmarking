#!/usr/bin/env python3
"""Compatibility shim for the unified benchmark CLI.

Legacy usage is preserved by forwarding to:
  python benchmarking/bench.py swebench ...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    bench = Path(__file__).resolve().with_name("bench.py")
    cmd = [sys.executable, str(bench), "swebench", *sys.argv[1:]]
    print("[deprecated] run_swebench.py -> bench.py swebench")
    raise SystemExit(subprocess.call(cmd))
