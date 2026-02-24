#!/usr/bin/env python3
"""Compatibility shim for the unified benchmark CLI.

Legacy usage is preserved by forwarding to:
  python benchmarking/bench.py general ...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    bench = Path(__file__).resolve().with_name("bench.py")
    cmd = [sys.executable, str(bench), "general", *sys.argv[1:]]
    print("[deprecated] run_benchmark.py -> bench.py general")
    raise SystemExit(subprocess.call(cmd))
