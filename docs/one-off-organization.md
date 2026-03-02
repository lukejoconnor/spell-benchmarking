# One-Off Analysis Organization

This repo now separates ephemeral exploratory work from maintained benchmark code.

## Layout

- `scratch/batches/<batch-name>/analyses/scripts/`
  - One-off Python scripts and quick experiments.
- `scratch/batches/<batch-name>/analyses/notes/`
  - Notes, markdown summaries, and text snippets.
- `scratch/batches/<batch-name>/results/`
  - JSON/JSONL/CSV outputs and lockfiles from exploratory runs.
- `scratch/batches/<batch-name>/misc/`
  - Everything else that does not match the above buckets.
- `scratch/batches/<batch-name>/MANIFEST.md`
  - Auto-generated index of moved files.

`scratch/` is gitignored so exploratory artifacts stay local by default.

## Organizer Script

Use:

```bash
uv run python scripts/organize_one_offs.py --dry-run
uv run python scripts/organize_one_offs.py
```

Options:

- `--batch-name <name>`: set a custom folder name.
- `--dry-run`: print planned moves without changing files.

The script only moves untracked files located directly at repository root.
Nested files and directories are intentionally left untouched.
