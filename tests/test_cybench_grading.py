import json
import os
from pathlib import Path

import pytest

from src import cybench_grading as gb


def write_log(
    logs_dir: Path,
    task_dir_name: str,
    model_name: str,
    benchmark_id: str,
    file_name: str,
    challenge_path: str,
    run_with_subtasks: bool,
    answers: list[str],
    scores: list[int],
    stdouts: list[str],
    input_tokens: list[int] | None = None,
    output_tokens: list[int] | None = None,
) -> Path:
    input_tokens = input_tokens or [0] * len(scores)
    output_tokens = output_tokens or [0] * len(scores)

    log_dir = logs_dir / task_dir_name / model_name / benchmark_id
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / file_name

    subtasks: dict[str, dict] = {}
    for i, score in enumerate(scores, start=1):
        subtasks[f"subtask_{i}"] = {
            "score": score,
            "iterations": [
                {
                    "model_input": {"num_tokens": input_tokens[i - 1]},
                    "model_response": {"num_tokens": output_tokens[i - 1]},
                    "execution_output": {"stdout": stdouts[i - 1]},
                }
            ],
        }

    payload = {
        "task": {
            "path": challenge_path,
            "subtasks": [{"answer": answer} for answer in answers],
        },
        "task_run_input": {
            "agent_config": {"deployment_name": model_name},
            "task_run_config": {"run_with_subtasks": run_with_subtasks},
        },
        "subtask_completions": subtasks,
        "usage": {"input_tokens_used": 0, "output_tokens_used": 0},
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return file_path


def test_latest_per_task_keeps_only_newest_run(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    model = "openai/gpt-4o-2024-05-13"
    challenge_path = "benchmark/example/task-a"

    old_file = write_log(
        logs_dir=logs_dir,
        task_dir_name="task-a",
        model_name=model,
        benchmark_id="2025_01_01_00-00-00-000000",
        file_name="old_run_log.json",
        challenge_path=challenge_path,
        run_with_subtasks=False,
        answers=["flag{old}"],
        scores=[0],
        stdouts=["nope"],
    )
    new_file = write_log(
        logs_dir=logs_dir,
        task_dir_name="task-a",
        model_name=model,
        benchmark_id="2025_01_01_00-00-01-000000",
        file_name="new_run_log.json",
        challenge_path=challenge_path,
        run_with_subtasks=False,
        answers=["flag{new}"],
        scores=[1],
        stdouts=["flag{new}"],
    )

    os.utime(old_file, (1000, 1000))
    os.utime(new_file, (2000, 2000))

    records = gb.discover_log_records(str(logs_dir))
    filtered = gb.filter_records(records, mode="unguided", latest_per_task=True)
    summary = gb.summarize_records(filtered)

    assert summary[model]["attempt_count"] == 1
    assert summary[model]["strict_solved_percentage"] == pytest.approx(100.0)


def test_mode_classification_uses_metadata_over_filename(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    model = "openai/gpt-4o-2024-05-13"
    file_path = write_log(
        logs_dir=logs_dir,
        task_dir_name="task-a",
        model_name=model,
        benchmark_id="run-id",
        file_name="contains_subtask_in_name_run_log.json",
        challenge_path="benchmark/example/task-a",
        run_with_subtasks=False,
        answers=["flag{a}"],
        scores=[0],
        stdouts=["none"],
    )

    records = gb.discover_log_records(str(logs_dir))
    record = next(r for r in records if r["file_path"] == str(file_path))
    assert record["mode"] == "unguided"


def test_strict_and_partial_percentages_are_distinct(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    model = "openai/gpt-4o-2024-05-13"

    write_log(
        logs_dir=logs_dir,
        task_dir_name="task-a",
        model_name=model,
        benchmark_id="run-1",
        file_name="run_a.json",
        challenge_path="benchmark/example/task-a",
        run_with_subtasks=True,
        answers=["flag{alpha}"],
        scores=[1],
        stdouts=["flag{alpha}"],
    )
    write_log(
        logs_dir=logs_dir,
        task_dir_name="task-b",
        model_name=model,
        benchmark_id="run-2",
        file_name="run_b.json",
        challenge_path="benchmark/example/task-b",
        run_with_subtasks=True,
        answers=["flag{beta}"],
        scores=[0],
        stdouts=["beta"],
    )

    records = gb.discover_log_records(str(logs_dir))
    filtered = gb.filter_records(records, mode="guided", latest_per_task=True)
    summary = gb.summarize_records(filtered)

    assert summary[model]["strict_solved_percentage"] == pytest.approx(50.0)
    assert summary[model]["partial_solved_percentage"] == pytest.approx(100.0)


def test_output_limit_uses_model_response_tokens(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    model = "openai/gpt-4o-2024-05-13"

    write_log(
        logs_dir=logs_dir,
        task_dir_name="task-a",
        model_name=model,
        benchmark_id="run-1",
        file_name="run_a.json",
        challenge_path="benchmark/example/task-a",
        run_with_subtasks=True,
        answers=["flag{alpha}"],
        scores=[0],
        stdouts=["none"],
        input_tokens=[10],
        output_tokens=[2500],
    )

    records = gb.discover_log_records(str(logs_dir))
    summary = gb.summarize_records(
        records,
        input_limit_threshold=5999,
        output_limit_threshold=1999,
    )

    assert summary[model]["output_limit_hits"] == 1
    assert summary[model]["input_limit_hits"] == 0


def test_grading_does_not_mutate_logs_directory(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    write_log(
        logs_dir=logs_dir,
        task_dir_name="task-a",
        model_name="openai/gpt-4o-2024-05-13",
        benchmark_id="run-1",
        file_name="run_a.json",
        challenge_path="benchmark/example/task-a",
        run_with_subtasks=True,
        answers=["flag{alpha}"],
        scores=[1],
        stdouts=["flag{alpha}"],
    )

    before = sorted(str(path.relative_to(logs_dir)) for path in logs_dir.rglob("*"))

    records = gb.discover_log_records(str(logs_dir))
    filtered = gb.filter_records(records, latest_per_task=True)
    gb.summarize_records(filtered)

    after = sorted(str(path.relative_to(logs_dir)) for path in logs_dir.rglob("*"))
    assert before == after
