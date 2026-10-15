from __future__ import annotations

import json
from pathlib import Path

from src.spell_trace_features import analyze_trace_roots, count_suffixes


def write_trace(path: Path, nodes: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{:nodes [" + " ".join("{:prompt " + json.dumps(prompt) + " :response " + json.dumps(response) + "}"
                                          for prompt, response in nodes) + "]}")


def test_counts_new_suffix_calls_without_recounting_prefix(tmp_path):
    trace = tmp_path / "item" / "trace.edn"
    first = "'(!peek result (io/read-lines \"a\"))"
    write_trace(trace, [("(agents/spawn ...) (map f xs)", first),
                        (first + " (def result ...)", "(persist line 42) '(!call-now x (io/ls \".\"))")])
    summary = analyze_trace_roots([tmp_path])
    assert summary["features"]["agents/*"]["uses"] == 0
    assert summary["features"]["map"]["uses"] == 0
    assert summary["features"]["!peek"]["uses"] == 1
    assert summary["features"]["io/*"]["uses"] == 2
    assert summary["features"]["persist"]["traces_using"] == 1
    assert summary["turns"] == 2


def test_direct_call_boundaries_and_namespace_counts():
    counts = count_suffixes(["(if-let [x y] (mapv f xs) (get-in z [:a])) (if true (map f xs) (get x :a)) "
                            "'(agents/!ask :worker) (!peek-now x (io/ls \".\"))"])["counts"]
    assert counts["if"] == 1
    assert counts["map"] == 1
    assert counts["get"] == 1
    assert counts["agents/*"] == 1
    assert counts["!peek"] == 1


def test_deduplicates_roots_and_reports_ineligible_traces(tmp_path):
    write_trace(tmp_path / "ok" / "trace.edn", [("prefix", "(io/ls \".\") (io/stat \"a\")")])
    bad = tmp_path / "empty" / "trace.edn"
    bad.parent.mkdir()
    bad.write_text("{:nodes [{:response nil}]}")
    summary = analyze_trace_roots([tmp_path, tmp_path / "ok"])
    assert summary["trace_count"] == 1
    assert len(summary["skipped"]) == 1
    assert summary["io_batching"]["mean_calls_per_turn"] == 2
    assert summary["io_batching"]["turns_with_multiple_calls"] == 1


def test_runtime_response_fields_are_not_model_turns(tmp_path):
    trace = tmp_path / "trace.edn"
    trace.write_text('{:nodes [{:response "42" :value {:response "(agents/spawn :x {})"}}]}')
    result = analyze_trace_roots([tmp_path])
    assert result["turns"] == 1
    assert result["features"]["agents/*"]["uses"] == 0


def test_truncated_trace_is_reported_instead_of_counted(tmp_path):
    trace = tmp_path / "trace.edn"
    trace.write_text('{:nodes [{:response "(io/ls)"} {:response "unfinished')
    result = analyze_trace_roots([tmp_path])
    assert result["trace_count"] == 0
    assert len(result["skipped"]) == 1


def test_runtime_sets_and_tagged_values_do_not_affect_responses(tmp_path):
    trace = tmp_path / "trace.edn"
    trace.write_text('{:nodes [{:response "(io/ls)" :value #{:a} '
                     ':runtime #object[java.lang.Object 0x123 "opaque"]}]}')
    result = analyze_trace_roots([tmp_path])
    assert result["turns"] == 1
    assert result["features"]["io/*"]["uses"] == 1
    assert not result["skipped"]


def test_regex_literals_with_non_json_escapes_are_readable(tmp_path):
    trace = tmp_path / "trace.edn"
    trace.write_text(r'{:nodes [{:response "42" :program (re-find #"\d+\s*" "123")}]}')
    result = analyze_trace_roots([tmp_path])
    assert result["turns"] == 1
    assert not result["skipped"]
