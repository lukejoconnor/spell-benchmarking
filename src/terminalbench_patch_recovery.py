"""Terminal-Bench evaluator test-patch collision recovery helpers."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TEST_PATCH_COLLISION_RECOVERY_MARKER = "SPELL_TB_TEST_PATCH_COLLISION_RECOVERED"
TEST_PATCH_COLLISION_CAVEAT = (
    "official test patch collided with agent test-file edits; evaluator test files "
    "were restored and the patch was retried"
)
DEFAULT_TOUCHED_TEST_FILES_PATH = "/tmp/spell-tb-touched-test-files.txt"

PATCH_COLLISION_SIGNATURES = (
    "which already exists",
    "Reversed (or previously applied) patch detected",
    "Assume -R?",
    "Apply anyway?",
)

_PATCH_PATH_PREFIXES = ("a/", "b/")


@dataclass(frozen=True)
class TerminalBenchPatchCollisionArtifact:
    run_dir: str
    task_id: str
    trial_name: str
    failure_mode: str | None
    is_resolved: bool | None
    evidence_paths: list[str]
    signatures: list[str]
    recovered: bool
    status: str


def _normalize_patch_path(path: str) -> str | None:
    cleaned = path.strip().split("\t", 1)[0]
    if not cleaned or cleaned == "/dev/null":
        return None
    for prefix in _PATCH_PATH_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if cleaned.startswith("/") or cleaned in {".", ".."}:
        return None
    if any(part == ".." for part in Path(cleaned).parts):
        return None
    return cleaned


def is_test_like_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip()
    cleaned = _normalize_patch_path(normalized) or normalized
    parts = [part for part in cleaned.split("/") if part]
    if not parts:
        return False
    name = parts[-1].lower()
    lowered_parts = {part.lower() for part in parts[:-1]}
    if lowered_parts & {"test", "tests", "__tests__", "spec", "specs"}:
        return True
    if name.startswith(("test_", "test-", "test.")):
        return True
    if re.search(r"(^|[._-])(test|spec)([._-]|$)", name):
        return True
    if re.search(r"_(test|spec)\.[^.]+$", name):
        return True
    return False


def parse_git_status_touched_test_files(status_output: str) -> list[str]:
    touched: list[str] = []
    seen: set[str] = set()
    for raw_line in status_output.splitlines():
        if not raw_line:
            continue
        path = raw_line[3:] if len(raw_line) > 3 else raw_line
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        normalized = _normalize_patch_path(path.strip().strip('"'))
        if normalized and is_test_like_path(normalized) and normalized not in seen:
            seen.add(normalized)
            touched.append(normalized)
    return touched


def parse_patch_target_paths(patch_text: str) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for line in patch_text.splitlines():
        candidate: str | None = None
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[3]
        elif line.startswith("+++ "):
            candidate = line[4:].strip()
        elif line.startswith("*** ") and not line.startswith("********"):
            candidate = line[4:].strip()
        if candidate is None:
            continue
        normalized = _normalize_patch_path(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            targets.append(normalized)
    return targets


def contains_patch_collision_signature(text: str) -> bool:
    return any(signature in text for signature in PATCH_COLLISION_SIGNATURES)


def contains_recovery_marker(text: str) -> bool:
    return TEST_PATCH_COLLISION_RECOVERY_MARKER in text


def recovery_marker_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(rf"{re.escape(TEST_PATCH_COLLISION_RECOVERY_MARKER)}\s+path=([^\s]+)")
    for match in pattern.finditer(text):
        path = match.group(1)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def build_patch_wrapper_script(
    real_patch_path: str,
    *,
    touched_files_path: str = DEFAULT_TOUCHED_TEST_FILES_PATH,
) -> str:
    collision_regex = "|".join(re.escape(signature) for signature in PATCH_COLLISION_SIGNATURES)
    marker = shlex.quote(TEST_PATCH_COLLISION_RECOVERY_MARKER)
    return f"""#!/bin/sh
set -u

REAL_PATCH={shlex.quote(real_patch_path)}
TOUCHED_FILES_PATH=${{SPELL_TB_TOUCHED_TEST_FILES_PATH:-{shlex.quote(touched_files_path)}}}

payload="$(mktemp "${{TMPDIR:-/tmp}}/spell-tb-patch.XXXXXX")" || exit 2
output="$(mktemp "${{TMPDIR:-/tmp}}/spell-tb-patch-output.XXXXXX")" || exit 2
trap 'rm -f "$payload" "$output"' EXIT
cat > "$payload"

"$REAL_PATCH" --batch --dry-run "$@" < "$payload" > "$output" 2>&1
status=$?
cat "$output" >&2
if [ "$status" -eq 0 ]; then
  : > "$output"
  "$REAL_PATCH" --batch "$@" < "$payload" > "$output" 2>&1
  apply_status=$?
  cat "$output" >&2
  exit "$apply_status"
fi

if ! grep -E {shlex.quote(collision_regex)} "$output" >/dev/null 2>&1; then
  exit "$status"
fi

targets="$(python3 - "$payload" "$TOUCHED_FILES_PATH" <<'PY'
from pathlib import Path
import re
import sys

payload_path = Path(sys.argv[1])
touched_path = Path(sys.argv[2])

def normalize(path):
    cleaned = path.strip().split("\\t", 1)[0]
    if not cleaned or cleaned == "/dev/null":
        return None
    for prefix in ("a/", "b/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    parts = Path(cleaned).parts
    if cleaned.startswith("/") or cleaned in {{".", ".."}} or any(part == ".." for part in parts):
        return None
    return cleaned

def is_test(path):
    normalized = path.replace("\\\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    name = parts[-1].lower()
    parents = {{part.lower() for part in parts[:-1]}}
    if parents & {{"test", "tests", "__tests__", "spec", "specs"}}:
        return True
    if name.startswith(("test_", "test-", "test.")):
        return True
    if re.search(r"(^|[._-])(test|spec)([._-]|$)", name):
        return True
    if re.search(r"_(test|spec)\\.[^.]+$", name):
        return True
    return False

def patch_targets(text):
    seen = set()
    for line in text.splitlines():
        candidate = None
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[3]
        elif line.startswith("+++ "):
            candidate = line[4:].strip()
        elif line.startswith("*** ") and not line.startswith("********"):
            candidate = line[4:].strip()
        if candidate is None:
            continue
        normalized = normalize(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            yield normalized

touched = set()
if touched_path.exists():
    touched = {{line.strip() for line in touched_path.read_text(encoding="utf-8").splitlines() if line.strip()}}

for target in patch_targets(payload_path.read_text(encoding="utf-8", errors="replace")):
    if is_test(target) and (not touched or target in touched):
        print(target)
PY
)"

if [ -z "$targets" ]; then
  exit "$status"
fi

repo="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
printf '%s\\n' "$targets" | while IFS= read -r path; do
  [ -n "$path" ] || continue
  if git -C "$repo" ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
    git -C "$repo" restore -- "$path"
  else
    rm -rf -- "$repo/$path"
  fi
done

: > "$output"
"$REAL_PATCH" --batch "$@" < "$payload" > "$output" 2>&1
retry_status=$?
cat "$output" >&2
if [ "$retry_status" -eq 0 ]; then
  printf '%s\\n' "$targets" | while IFS= read -r path; do
    [ -n "$path" ] || continue
    printf '%s path=%s\\n' {marker} "$path" >&2
  done
fi
exit "$retry_status"
"""


def build_patch_wrapper_install_script(
    *,
    wrapper_path: str = "/usr/local/bin/patch",
    touched_files_path: str = DEFAULT_TOUCHED_TEST_FILES_PATH,
) -> str:
    wrapper = build_patch_wrapper_script("__SPELL_TB_REAL_PATCH__", touched_files_path=touched_files_path)
    wrapper_json = json.dumps(wrapper)
    wrapper_path_json = json.dumps(wrapper_path)
    return f"""set -eu
if [ -x /usr/bin/patch ]; then
  real_patch=/usr/bin/patch
else
  real_patch="$(command -v patch || true)"
fi
if [ -z "$real_patch" ]; then
  echo "patch command not found; cannot install Terminal-Bench recovery wrapper" >&2
  exit 1
fi
if [ "$real_patch" = {shlex.quote(wrapper_path)} ]; then
  exit 0
fi
REAL_PATCH_PATH="$real_patch" python3 - <<'PY'
import os
from pathlib import Path

wrapper = {wrapper_json}
wrapper_path = Path({wrapper_path_json})
wrapper_path.parent.mkdir(parents=True, exist_ok=True)
wrapper_path.write_text(wrapper.replace("__SPELL_TB_REAL_PATCH__", os.environ["REAL_PATCH_PATH"]), encoding="utf-8")
wrapper_path.chmod(0o755)
PY
"""


def recovered_metadata_from_texts(
    texts: list[str],
    *,
    touched_test_files: list[str] | None = None,
) -> dict[str, Any]:
    combined = "\n".join(texts)
    metadata: dict[str, Any] = {}
    if touched_test_files:
        metadata["touched_test_files"] = touched_test_files
    if contains_recovery_marker(combined):
        metadata["test_patch_collision_recovered"] = True
        metadata["test_patch_collision_paths"] = recovery_marker_paths(combined)
        metadata["score_caveat"] = TEST_PATCH_COLLISION_CAVEAT
    return metadata


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _trial_evidence_paths(trial_dir: Path) -> list[Path]:
    candidates = [
        trial_dir / "sessions" / "tests.log",
        trial_dir / "panes" / "post-test.txt",
    ]
    if trial_dir.exists():
        candidates.extend(trial_dir.glob("**/tests.log"))
        candidates.extend(trial_dir.glob("**/post-test.txt"))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path not in seen and path.exists():
            seen.add(path)
            ordered.append(path)
    return ordered


def _text_has_timeout(text: str) -> bool:
    return "timeout" in text.lower() or "timed out" in text.lower()


def scan_terminalbench_patch_collision_artifacts(root: Path) -> list[TerminalBenchPatchCollisionArtifact]:
    records: list[TerminalBenchPatchCollisionArtifact] = []
    for results_path in sorted(root.rglob("results.json")):
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = results_path.parent
        for trial in payload.get("results") or []:
            task_id = str(trial.get("task_id") or "")
            trial_name = str(trial.get("trial_name") or "")
            if not task_id or not trial_name:
                continue
            trial_dir = run_dir / task_id / trial_name
            evidence_paths = _trial_evidence_paths(trial_dir)
            texts = [_read_text(path) for path in evidence_paths]
            combined = "\n".join(texts)
            recovered = contains_recovery_marker(combined)
            signatures = [signature for signature in PATCH_COLLISION_SIGNATURES if signature in combined]
            failure_mode = trial.get("failure_mode")
            timed_out = (
                isinstance(failure_mode, str)
                and "timeout" in failure_mode.lower()
            ) or _text_has_timeout(combined)
            if not recovered and not signatures:
                continue
            status = "test_patch_collision_recovered" if recovered else "test_patch_collision_unrechecked"
            records.append(
                TerminalBenchPatchCollisionArtifact(
                    run_dir=str(run_dir),
                    task_id=task_id,
                    trial_name=trial_name,
                    failure_mode=str(failure_mode) if failure_mode is not None else None,
                    is_resolved=trial.get("is_resolved") if isinstance(trial.get("is_resolved"), bool) else None,
                    evidence_paths=[str(path) for path in evidence_paths],
                    signatures=signatures,
                    recovered=recovered,
                    status=status,
                )
            )
    return records


def artifacts_to_jsonable(records: list[TerminalBenchPatchCollisionArtifact]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
