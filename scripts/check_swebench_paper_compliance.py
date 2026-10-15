#!/usr/bin/env python3
"""Best-effort checks for SWE-bench paper-compliant benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PAPER_INSTRUCTION_PREFIX = "Paper-compliance rule: Do not use web browsing or external internet resources"

SUSPICIOUS_PATTERNS = [
    ("claude_webfetch", re.compile(r"\bWebFetch\b")),
    ("claude_websearch", re.compile(r"\bWebSearch\b")),
    ("codex_search", re.compile(r"\bcodex\b[^\n]*\s--search\b")),
    (
        "github_url_fetch",
        re.compile(r"\b(?:curl|wget|git\s+clone)\b[^\n]*https://github\.com/[^\s'\"`]+", re.IGNORECASE),
    ),
    (
        "patch_url",
        re.compile(
            r"(?:patch-diff\.githubusercontent\.com|github\.com/[^\s'\"`]+/files/[^\s'\"`]+|https://[^\s'\"`]+\.patch\b)",
            re.IGNORECASE,
        ),
    ),
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _selected_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str | None,
    condition: str | None,
    name: str | None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (run_id is None or row.get("run_id") == run_id)
        and (condition is None or row.get("condition") == condition)
        and (name is None or row.get("name") == name)
    ]


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file():
            try:
                yield path, path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue


def check_artifacts(
    jsonl_path: Path,
    *,
    run_id: str | None = None,
    condition: str | None = None,
    name: str | None = None,
    trace_root: Path | None = None,
    warnings_are_failures: bool = True,
) -> dict[str, Any]:
    rows = _selected_rows(_load_jsonl(jsonl_path), run_id=run_id, condition=condition, name=name)
    failures: list[str] = []
    warnings: list[dict[str, str]] = []

    if not rows:
        failures.append("no selected rows")

    for row in rows:
        label = f"{row.get('run_id')} {row.get('condition')} {row.get('item_id')}"
        metadata = row.get("metadata") or {}
        if metadata.get("paper_compliant") is not True:
            failures.append(f"{label}: metadata.paper_compliant is not true")
        if metadata.get("hints_included") is not False:
            failures.append(f"{label}: metadata.hints_included is not false")
        if metadata.get("web_search") is True:
            failures.append(f"{label}: metadata.web_search is true")
        if metadata.get("paper_compliance_instruction") is not True:
            failures.append(f"{label}: metadata.paper_compliance_instruction is not true")

        prompt = metadata.get("prompt") or metadata.get("rendered_prompt")
        if isinstance(prompt, str):
            if "## Hints" in prompt:
                failures.append(f"{label}: prompt contains a ## Hints block")
            if PAPER_INSTRUCTION_PREFIX not in prompt:
                failures.append(f"{label}: prompt is missing the paper-compliance instruction")

    trace_files_scanned = 0
    if trace_root is not None and trace_root.exists():
        saw_instruction = False
        for path, text in _iter_text_files(trace_root):
            trace_files_scanned += 1
            if PAPER_INSTRUCTION_PREFIX in text:
                saw_instruction = True
            if "## Hints" in text:
                warnings.append({"file": str(path), "kind": "hints_block", "match": "## Hints"})
            for kind, pattern in SUSPICIOUS_PATTERNS:
                match = pattern.search(text)
                if match:
                    warnings.append({"file": str(path), "kind": kind, "match": match.group(0)[:200]})
        if trace_files_scanned and not saw_instruction:
            warnings.append(
                {
                    "file": str(trace_root),
                    "kind": "instruction_not_seen_in_traces",
                    "match": PAPER_INSTRUCTION_PREFIX,
                }
            )

    return {
        "jsonl": str(jsonl_path),
        "selected_rows": len(rows),
        "trace_files_scanned": trace_files_scanned,
        "failures": failures,
        "warnings": warnings,
        "warnings_are_failures": warnings_are_failures,
        "ok": not failures and (not warnings or not warnings_are_failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", type=Path, help="Unified JSONL output path")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--trace-root", type=Path, default=None)
    parser.add_argument("--allow-warnings", action="store_true", help="Exit zero when only best-effort trace warnings are present")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = check_artifacts(
        args.jsonl_path,
        run_id=args.run_id,
        condition=args.condition,
        name=args.name,
        trace_root=args.trace_root,
        warnings_are_failures=not args.allow_warnings,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"selected_rows={summary['selected_rows']} "
            f"failures={len(summary['failures'])} warnings={len(summary['warnings'])} "
            f"trace_files_scanned={summary['trace_files_scanned']}"
        )
        for failure in summary["failures"]:
            print(f"FAIL {failure}")
        for warning in summary["warnings"]:
            print(f"WARN {warning['kind']} {warning['file']}: {warning['match']}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
