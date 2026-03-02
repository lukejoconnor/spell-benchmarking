#!/usr/bin/env python3
"""Organize loose, untracked root files into a timestamped scratch batch."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from collections import defaultdict
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_untracked_root_files(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    out: list[Path] = []
    for rel in proc.stdout.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        # Only move loose files from repo root; nested paths are left untouched.
        if "/" in rel:
            continue
        p = root / rel
        if p.is_file():
            out.append(p)
    return sorted(out)


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".py"}:
        return "analyses/scripts"
    if suffix in {".md", ".txt"}:
        return "analyses/notes"
    if suffix in {".json", ".jsonl", ".csv", ".tsv", ".lock"}:
        return "results"
    return "misc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Optional batch folder name. Default: root-cleanup-YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves only.")
    args = parser.parse_args()

    root = repo_root()
    batch_name = args.batch_name or f"root-cleanup-{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_root = root / "scratch" / "batches" / batch_name

    files = list_untracked_root_files(root)
    if not files:
        print("No loose untracked root files found.")
        return 0

    grouped: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for src in files:
        bucket = classify(src)
        dst = batch_root / bucket / src.name
        grouped[bucket].append((src, dst))

    for bucket, moves in sorted(grouped.items()):
        print(f"[{bucket}] {len(moves)} files")
        for src, dst in moves:
            print(f"  {src.relative_to(root)} -> {dst.relative_to(root)}")

    if args.dry_run:
        return 0

    for moves in grouped.values():
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)

    manifest = batch_root / "MANIFEST.md"
    total = sum(len(moves) for moves in grouped.values())
    lines = [
        f"# {batch_name}",
        "",
        f"- Created: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Total files moved: {total}",
        "",
        "## Buckets",
    ]
    for bucket in sorted(grouped):
        lines.append(f"- `{bucket}`: {len(grouped[bucket])}")
    lines.append("")
    lines.append("## Files")
    for bucket in sorted(grouped):
        lines.append(f"### `{bucket}`")
        for _, dst in grouped[bucket]:
            lines.append(f"- `{dst.relative_to(root)}`")
        lines.append("")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote manifest: {manifest.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
