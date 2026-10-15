from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

benchmark_root = str(Path(__file__).resolve().parents[1])

spell_root = os.environ.get("SPELL_ROOT")
if spell_root:
    root_path = Path(spell_root)
    if root_path.exists() and str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

sys.path.insert(0, benchmark_root)

if importlib.util.find_spec("spell_benchmark_client") is None and "spell_benchmark_client" not in sys.modules:
    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - import wiring only
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            raise NotImplementedError

    stub.SpellBenchmarkClient = _DummyClient
    sys.modules["spell_benchmark_client"] = stub
