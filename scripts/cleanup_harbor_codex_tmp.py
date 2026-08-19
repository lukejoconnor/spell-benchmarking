#!/usr/bin/env python3
"""Safely remove replicated Codex temp directories from Harbor trial artifacts."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "harbor-terminal-bench"
EXPECTED_TMP_ENTRIES = frozenset({"plugins", "plugins.sha"})
PLUGIN_CLONE_PATTERN = re.compile(r"plugins-clone-[A-Za-z0-9]+")
CONFIRMATION = "delete-harbor-codex-tmp"


class UnsafeCandidateError(RuntimeError):
    """Raised when a candidate does not match the expected generated layout."""


def resolve_results_root(value: str) -> Path:
    root = Path(value).expanduser()
    if root.is_symlink():
        raise UnsafeCandidateError(f"results root must not be a symlink: {root}")
    root = root.resolve(strict=True)
    allowed_root = DEFAULT_RESULTS_ROOT.resolve(strict=True)
    if root != allowed_root:
        raise UnsafeCandidateError(f"results root must be exactly {allowed_root}: {root}")
    if not root.is_dir():
        raise UnsafeCandidateError(f"results root is not a directory: {root}")
    return root


def validate_candidate(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise UnsafeCandidateError(f"candidate must be a real directory: {path}")
    if path.name != ".tmp" or path.parent.name != "agent":
        raise UnsafeCandidateError(f"candidate has an unexpected path shape: {path}")

    entries = list(path.iterdir())
    entry_names = {entry.name for entry in entries}
    unexpected = sorted(
        entry.name
        for entry in entries
        if entry.name not in EXPECTED_TMP_ENTRIES
        and not (PLUGIN_CLONE_PATTERN.fullmatch(entry.name) and entry.is_dir())
    )
    if unexpected:
        raise UnsafeCandidateError(
            f"candidate contains unexpected entries ({', '.join(unexpected)}): {path}"
        )
    has_plugin_tree = "plugins" in entry_names or any(
        PLUGIN_CLONE_PATTERN.fullmatch(entry.name) and entry.is_dir() for entry in entries
    )
    if not has_plugin_tree:
        raise UnsafeCandidateError(f"candidate does not contain a recognized plugin tree: {path}")
    if any(entry.is_symlink() for entry in entries):
        raise UnsafeCandidateError(f"candidate contains a top-level symlink: {path}")


def find_candidates(root: Path) -> list[Path]:
    candidates = sorted(root.glob("*/*/*/*/agent/.tmp"))
    for candidate in candidates:
        validate_candidate(candidate)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Harbor results root; must resolve to the repository's known results path",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete validated directories; the default is a read-only dry run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation used with --apply",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.yes and not args.apply:
        print("error: --yes requires --apply", file=sys.stderr)
        return 2

    try:
        root = resolve_results_root(args.root)
        candidates = find_candidates(root)
    except (FileNotFoundError, UnsafeCandidateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    action = "delete" if args.apply else "would delete"
    for candidate in candidates:
        print(f"{action}\t{candidate}")
    print(f"validated {len(candidates)} Harbor Codex temp directories")

    if not args.apply or not candidates:
        return 0

    if not args.yes:
        response = input(f"Type {CONFIRMATION!r} to continue: ")
        if response != CONFIRMATION:
            print("aborted; nothing was deleted", file=sys.stderr)
            return 1

    for candidate in candidates:
        shutil.rmtree(candidate)
    print(f"deleted {len(candidates)} Harbor Codex temp directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
