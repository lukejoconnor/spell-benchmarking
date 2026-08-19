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
PLUGIN_CLONE_SIGNATURE = {
    "plugins": "dir",
    ".agents": "dir",
    "README.md": "file",
    ".gitignore": "file",
}
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


def is_real_dir(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir()


def is_real_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def validate_clone_signature(path: Path) -> None:
    if not is_real_dir(path):
        raise UnsafeCandidateError(f"plugin clone must be a real directory: {path}")
    for name, expected_type in PLUGIN_CLONE_SIGNATURE.items():
        child = path / name
        valid = is_real_dir(child) if expected_type == "dir" else is_real_file(child)
        if not valid:
            raise UnsafeCandidateError(
                f"plugin clone lacks expected {expected_type} {name!r}: {path}"
            )


def validate_candidate(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise UnsafeCandidateError(f"candidate is outside the results root: {path}") from exc

    parts = relative.parts
    if (
        len(parts) != 6
        or parts[1] != "codex"
        or not parts[2].startswith("codex-")
        or parts[4:] != ("agent", ".tmp")
    ):
        raise UnsafeCandidateError(f"candidate has an unexpected path shape: {path}")

    current = root
    for part in parts:
        current = current / part
        if not is_real_dir(current):
            raise UnsafeCandidateError(f"candidate has a symlink or non-directory ancestor: {current}")
    if root not in path.resolve(strict=True).parents:
        raise UnsafeCandidateError(f"candidate resolves outside the results root: {path}")

    entries = list(path.iterdir())
    entry_names = {entry.name for entry in entries}
    clones = [entry for entry in entries if PLUGIN_CLONE_PATTERN.fullmatch(entry.name)]
    unexpected = sorted(entry_names - EXPECTED_TMP_ENTRIES - {clone.name for clone in clones})
    if unexpected:
        raise UnsafeCandidateError(
            f"candidate contains unexpected entries ({', '.join(unexpected)}): {path}"
        )

    if "plugins" in entry_names:
        if entry_names != EXPECTED_TMP_ENTRIES or not is_real_dir(path / "plugins"):
            raise UnsafeCandidateError(f"completed cache has an unexpected layout: {path}")
        if not is_real_file(path / "plugins.sha"):
            raise UnsafeCandidateError(f"plugins.sha must be a real file: {path}")
        return

    if len(clones) != 1 or len(entries) != 1:
        raise UnsafeCandidateError(f"partial cache must contain exactly one plugin clone: {path}")
    validate_clone_signature(clones[0])


def find_candidates(root: Path) -> list[Path]:
    candidates = sorted(root.glob("*/codex/codex-*/*/agent/.tmp"))
    for candidate in candidates:
        validate_candidate(candidate, root)
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
