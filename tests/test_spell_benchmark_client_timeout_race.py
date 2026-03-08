from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spell_benchmark_client import SpellBenchmarkClient


class _FakeProc:
    def __init__(self, responses):
        self._responses = list(responses)
        self.returncode = None

    def poll(self):
        return None if self.returncode is None else self.returncode

    def communicate(self, input=None, timeout=None):
        if not self._responses:
            self.returncode = 0
            return ("", "")
        next_response = self._responses.pop(0)
        if isinstance(next_response, BaseException):
            raise next_response
        if self.returncode is None:
            self.returncode = 0
        return next_response

    def terminate(self):
        raise ProcessLookupError("already exited")

    def kill(self):
        raise ProcessLookupError("already exited")


class SpellBenchmarkClientTimeoutRaceTest(unittest.TestCase):
    def test_returns_timeout_payload_when_terminate_races_with_exit(self):
        proc = _FakeProc(
            [
                subprocess.TimeoutExpired(["clj"], 1, output="partial-out", stderr="partial-err"),
                ("final-out", "final-err"),
            ]
        )

        with patch("spell_benchmark_client.subprocess.Popen", return_value=proc):
            client = SpellBenchmarkClient(project_root=Path.cwd(), clj_cmd=["clj"])
            response = client.run({"mode": "test"}, timeout=1, cwd=Path.cwd())

        self.assertFalse(response.ok)
        self.assertEqual(response.error_type, "timeout")
        self.assertIn("timed out after 1s", response.error or "")
        self.assertEqual(response.error_data["stdout"], "partial-out")
        self.assertEqual(response.error_data["stderr"], "partial-err")

    def test_returns_timeout_payload_when_kill_races_with_exit(self):
        proc = _FakeProc(
            [
                subprocess.TimeoutExpired(["clj"], 1, output="", stderr=""),
                subprocess.TimeoutExpired(["clj"], 5, output="", stderr=""),
                ("", ""),
            ]
        )

        with patch("spell_benchmark_client.subprocess.Popen", return_value=proc):
            client = SpellBenchmarkClient(project_root=Path.cwd(), clj_cmd=["clj"])
            response = client.run({"mode": "test"}, timeout=1, cwd=Path.cwd())

        self.assertFalse(response.ok)
        self.assertEqual(response.error_type, "timeout")
        self.assertIn("timed out after 1s", response.error or "")

    def test_recovers_usage_when_timeout_teardown_returns_final_json(self):
        final_payload = {
            "ok": True,
            "mode": "test",
            "result": "partial-result",
            "usage": {"total": {"input_tokens": 11, "output_tokens": 7, "cost": 0.42}},
            "latency_ms": 1234.0,
            "trace_dir": "traces/demo",
        }
        proc = _FakeProc(
            [
                subprocess.TimeoutExpired(["clj"], 1, output="", stderr=""),
                (json.dumps(final_payload), ""),
            ]
        )

        with patch("spell_benchmark_client.subprocess.Popen", return_value=proc):
            client = SpellBenchmarkClient(project_root=Path.cwd(), clj_cmd=["clj"])
            response = client.run({"mode": "test"}, timeout=1, cwd=Path.cwd())

        self.assertFalse(response.ok)
        self.assertEqual(response.error_type, "timeout")
        self.assertEqual(response.result, "partial-result")
        self.assertEqual(response.usage, final_payload["usage"])
        self.assertEqual(response.latency_ms, 1234.0)
        self.assertEqual(response.trace_dir, "traces/demo")
        self.assertTrue(response.error_data["recovered_payload"])


if __name__ == "__main__":
    unittest.main()
