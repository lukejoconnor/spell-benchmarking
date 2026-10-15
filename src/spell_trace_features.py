"""Suffix-only feature counting for Spell's EDN execution traces.

Uses the textual call patterns used by the paper trace analyses. Prefixes and
full replayed .spl programs are deliberately excluded from these counts.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


FUNCTIONS = ("!peek", "!call-now", "!llm-self", "leaf-llm", "rethink", "persist",
             "let", "map", "get", "apply", "if")
NAMESPACES = ("agents", "globals", "io", "strings", "web")
FEATURE_PATTERNS = {
    name: re.compile(r"\(" + (r"!peek(?:-now)?" if name == "!peek" else re.escape(name)) + r"(?=[\s,()\[\]{}]|$)")
    for name in FUNCTIONS
}
FEATURE_PATTERNS.update({
    f"{namespace}/*": re.compile(r"\(" + namespace + r"/[A-Za-z0-9_*+!\-?/.]+")
    for namespace in NAMESPACES
})


@dataclass
class _Form:
    kind: str
    value: object


class _TraceReader:
    """Read EDN structure without evaluating program forms or tagged values."""

    def __init__(self, source: str):
        self.source = source
        self.index = 0

    def whitespace(self) -> None:
        while self.index < len(self.source):
            char = self.source[self.index]
            if char.isspace() or char == ",":
                self.index += 1
            elif char == ";":
                end = self.source.find("\n", self.index)
                self.index = len(self.source) if end < 0 else end + 1
            else:
                break

    def form(self) -> _Form:
        self.whitespace()
        if self.index >= len(self.source):
            raise ValueError("truncated EDN")
        char = self.source[self.index]
        self.index += 1
        if char in "([{":
            closing = {"(": ")", "[": "]", "{": "}"}[char]
            values = []
            while True:
                self.whitespace()
                if self.index >= len(self.source):
                    raise ValueError("unterminated EDN collection")
                if self.source[self.index] == closing:
                    self.index += 1
                    if char == "{" and len(values) % 2:
                        raise ValueError("odd EDN map")
                    return _Form("map" if char == "{" else "sequence", values)
                values.append(self.form())
        if char in ")]}":
            raise ValueError("mismatched EDN collection")
        if char == '"':
            start = self.index
            while self.index < len(self.source):
                char = self.source[self.index]
                if char == '"':
                    raw = self.source[start:self.index]
                    self.index += 1
                    escaped = "".join(f"\\u{ord(c):04x}" if ord(c) < 32 else c for c in raw)
                    return _Form("string", json.loads('"' + escaped + '"'))
                self.index += 2 if char == "\\" else 1
            raise ValueError("unterminated EDN string")
        if char == "\\":
            # EDN character literals can contain delimiter characters, e.g. \).
            if self.index >= len(self.source):
                raise ValueError("truncated EDN character")
            self.index += 1
        elif char in "'`@~":
            return _Form("reader", self.form())
        elif char == "^":
            self.form()  # metadata
            return self.form()
        elif char == "#":
            if self.index < len(self.source) and self.source[self.index] == '"':
                # Clojure regex escapes (e.g. \d) are not JSON string escapes.
                self.index += 1
                start = self.index
                while self.index < len(self.source):
                    char = self.source[self.index]
                    if char == '"':
                        raw = self.source[start:self.index]
                        self.index += 1
                        return _Form("regex", raw)
                    self.index += 2 if char == "\\" else 1
                raise ValueError("unterminated regex literal")
            if self.index < len(self.source) and self.source[self.index] == "{":
                self.index += 1
                values = []
                while True:
                    self.whitespace()
                    if self.index >= len(self.source):
                        raise ValueError("unterminated EDN set")
                    if self.source[self.index] == "}":
                        self.index += 1
                        return _Form("set", values)
                    values.append(self.form())
            if self.index < len(self.source) and self.source[self.index] == "#":
                start = self.index - 1
                while self.index < len(self.source) and self.source[self.index] not in " \t\r\n,()[]{}":
                    self.index += 1
                return _Form("atom", self.source[start:self.index])
            if self.index < len(self.source) and self.source[self.index] == "_":
                self.index += 1
                self.form()  # reader-discarded data
                return self.form()
            if self.index < len(self.source) and self.source[self.index] in "('":
                return _Form("reader", self.form())
            while self.index < len(self.source) and self.source[self.index] not in " \t\r\n,()[]{}":
                self.index += 1
            return _Form("tagged", self.form())
        start = self.index - (2 if char == "\\" else 1)
        while self.index < len(self.source) and not self.source[self.index].isspace() and self.source[self.index] not in ",()[]{};":
            self.index += 1
        return _Form("atom", self.source[start:self.index])


def _map_value(form: _Form, keyword: str) -> _Form | None:
    if form.kind != "map":
        return None
    for key, value in zip(form.value[::2], form.value[1::2]):
        if key.kind == "atom" and key.value == keyword:
            return value
    return None


def response_suffixes(trace_file: Path) -> list[str]:
    reader = _TraceReader(trace_file.read_text(encoding="utf-8"))
    document = reader.form()
    reader.whitespace()
    if reader.index != len(reader.source):
        raise ValueError("extra forms after EDN trace")
    nodes = _map_value(document, ":nodes")
    if nodes is None or nodes.kind != "sequence":
        raise ValueError("trace has no :nodes collection")
    suffixes = []
    for node in nodes.value:
        response = _map_value(node, ":response")
        if response is not None and response.kind == "string":
            suffixes.append(response.value)
    if not suffixes:
        raise ValueError("no string-valued model responses")
    return suffixes


def count_suffixes(suffixes: list[str]) -> dict:
    counts = Counter({name: 0 for name in FEATURE_PATTERNS})
    io_counts = []
    for suffix in suffixes:
        per_turn = {name: len(pattern.findall(suffix)) for name, pattern in FEATURE_PATTERNS.items()}
        counts.update(per_turn)
        io_counts.append(per_turn["io/*"])
    return {"turns": len(suffixes), "counts": dict(counts), "io_calls_per_turn": io_counts}


def analyze_trace_roots(roots: list[Path], label: str = "Spell") -> dict:
    paths = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            paths.add(root.resolve())
        else:
            paths.update(path.resolve() for path in root.rglob("trace.edn"))
    if not paths:
        raise ValueError("no trace.edn files found")
    traces, skipped = [], []
    for path in sorted(paths):
        try:
            metrics = count_suffixes(response_suffixes(path))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            skipped.append({"trace": str(path), "reason": str(exc)})
            continue
        traces.append({"trace": str(path), **metrics})
    io_counts = [value for trace in traces for value in trace["io_calls_per_turn"]]
    return {
        "label": label,
        "traces": traces,
        "skipped": skipped,
        "trace_count": len(traces),
        "turns": len(io_counts),
        "features": {
            feature: {
                "uses": sum(trace["counts"][feature] for trace in traces),
                "traces_using": sum(trace["counts"][feature] > 0 for trace in traces),
                "traces": len(traces),
            } for feature in FEATURE_PATTERNS
        },
        "io_batching": {
            "mean_calls_per_turn": sum(io_counts) / len(io_counts) if io_counts else None,
            "turns_with_multiple_calls": sum(value >= 2 for value in io_counts),
            "max_calls_per_turn": max(io_counts, default=0),
        },
    }
