#!/usr/bin/env python3
"""Run prompt-only orchestration games through the published Spell benchmark API."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spell_benchmark_client import SpellBenchmarkClient
from src.provenance import benchmark_git_versions, resolve_spell_root

GAMES = ("auction", "telephone", "twenty-questions")
ASYNC_RUNTIME_COMMIT = "6ad7d400800eb87a6a34573c9d90dba5dd9c0b13"


def select_runtime(variant: str, override: Path | None = None) -> tuple[Path, dict]:
    if override is not None:
        root = override.expanduser().resolve()
    elif variant == "async":
        configured = os.environ.get("SPELL_ASYNC_ROOT")
        if not configured:
            raise ValueError("Set SPELL_ASYNC_ROOT to the pinned asynchronous-game checkout; see paper-repro/README.md")
        root = Path(configured).expanduser().resolve()
    else:
        root = resolve_spell_root()
    profile = "config/agent-profiles/io-tc.agent.edn" if variant == "async" else "config/agents/io-tc.agent.edn"
    if not (root / "src/spell/benchmark_api.clj").is_file() or not (root / profile).is_file():
        raise ValueError(f"Missing {variant} runtime API/profile in {root}; see paper-repro/README.md")
    versions = benchmark_git_versions(root)
    if variant == "async" and (versions.get("spell") or {}).get("commit") != ASYNC_RUNTIME_COMMIT:
        raise ValueError(f"Asynchronous games require Spell commit {ASYNC_RUNTIME_COMMIT}; check SPELL_ASYNC_ROOT")
    return root, versions


def game_request(variant: str, game: str, model: str, trace_dir: Path, budget: float, depth: int) -> dict:
    request = {
        "mode": "spell", "model": model,
        "prompt": (ROOT / "paper-repro/games" / variant / f"{game}.txt").read_text(),
        "reasoning_effort": "medium",
        "budget": budget, "depth": depth, "trace": True, "trace_dir": str(trace_dir.resolve()),
    }
    if variant == "async":
        request["agent_profile"] = "config/agent-profiles/io-tc.agent.edn"
        # The July runs used corrected tool-call framing. This commit's
        # benchmark API already supplies this control to api/run-internal.
        request["prefill"] = False
    else:
        request["agent"] = "config/agents/io-tc.agent.edn"
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sync", "async"), required=True)
    parser.add_argument("--game", choices=(*GAMES, "all"), default="all")
    parser.add_argument("--model", default="openai-tc:gpt-5.4")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--depth", type=int, default=80)
    parser.add_argument("--budget", type=float, default=0, help="Per-trial Spell budget; zero means no dollar cap")
    parser.add_argument("--clojure-command", default=os.environ.get("CLOJURE_COMMAND", "clojure"))
    parser.add_argument("--spell-root", type=Path, help="Runtime checkout; defaults to SPELL_ROOT for sync, SPELL_ASYNC_ROOT for async")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/games")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.replicates < 1 or args.timeout < 1 or args.depth < 1:
        parser.error("replicates, timeout, and depth must be positive")
    try:
        spell_root, versions = select_runtime(args.variant, args.spell_root)
    except ValueError as exc:
        parser.error(str(exc))
    sdeps = '{:deps {local/spell {:local/root ' + json.dumps(str(spell_root)) + '}}}'
    client = SpellBenchmarkClient(spell_root, [args.clojure_command, "-Sdeps", sdeps, "-M",
                                 "-i", str(ROOT / "scripts/paper_pricing.clj"), "-m", "spell.benchmark-api"])
    games = GAMES if args.game == "all" else (args.game,)
    for game in games:
        for trial in range(args.replicates):
            trial_dir = args.output_dir / args.run_id / args.variant / game / str(trial)
            request = game_request(args.variant, game, args.model, trial_dir / "trace", args.budget, args.depth)
            if args.dry_run:
                print(json.dumps({"game": game, "trial": trial, "versions": versions, "request": request}))
                continue
            if trial_dir.exists():
                raise SystemExit(f"Trial already exists: {trial_dir}; choose a new run-id")
            trial_dir.mkdir(parents=True)
            (trial_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
            (trial_dir / "provenance.json").write_text(json.dumps(versions, indent=2) + "\n")
            result = client.run(request, timeout=args.timeout, cwd=spell_root)
            payload = {"game": game, "variant": args.variant, "trial": trial,
                       "versions": versions, "response": asdict(result)}
            (trial_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
            print(json.dumps({"game": game, "trial": trial, "ok": result.ok, "error_type": result.error_type}))


if __name__ == "__main__":
    main()
