#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.datasets import load_swebench_lite
from src.agent_policy import resolve_spell_agent
from src.swebench_prep import setup_swebench_workspace
from src.workspace import RepoCache
from src.unified_adapters import RunRequest, SpellAdapter, CodexAdapter, PROJECT_ROOT, build_spell_init


def make_prompt(item):
    prompt_parts = [
        f"Repository: {item.repo}",
        "",
        "## Issue to Fix",
        "",
        item.problem_statement.strip(),
        "",
    ]
    if item.hints_text:
        prompt_parts.extend(["## Hints", "", item.hints_text.strip(), ""])

    prompt_parts.extend(
        [
            "## Instructions",
            "",
            "Fix this bug with minimal code changes.",
            "Do not commit any changes.",
            "Run focused checks/tests when possible.",
        ]
    )
    return "\n".join(prompt_parts)


def make_adapters():
    spell_toolcall = SpellAdapter(project_root=PROJECT_ROOT, model_spec="codex-tc:gpt-5.3-codex")
    codex_harness = CodexAdapter(project_root=PROJECT_ROOT, model_spec="codex")
    spell_messages = SpellAdapter(project_root=PROJECT_ROOT, model_spec="codex:gpt-5.3-codex")

    # Allow benchmark-api to run from cloned workspaces by injecting Spell as a local dep.
    sdeps = "{:deps {local/spell {:local/root \"%s\"}}}" % PROJECT_ROOT
    clj_cmd = ["clj", "-Sdeps", sdeps, "-M", "-m", "spell.benchmark-api"]
    spell_toolcall.client.clj_cmd = clj_cmd
    spell_messages.client.clj_cmd = clj_cmd

    return spell_toolcall, codex_harness, spell_messages


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/swebench")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"manual_high_reasoning_{ts}.json"

    items = load_swebench_lite(max_items=3)
    spell_toolcall, codex_harness, spell_messages = make_adapters()

    spell_message_agent = resolve_spell_agent(
        project_root=PROJECT_ROOT,
        bench_kind="swebench",
        model_spec="codex:gpt-5.3-codex",
    )
    spell_toolcall_agent = resolve_spell_agent(
        project_root=PROJECT_ROOT,
        bench_kind="swebench",
        model_spec="codex-tc:gpt-5.3-codex",
    )

    configs = [
        {
            "name": "spell_codex_toolcall",
            "condition": "spell",
            "model_spec": "codex-tc:gpt-5.3-codex",
            "agent_file": spell_toolcall_agent.agent_file,
            "adapter": spell_toolcall,
            "workspace_root": Path(f"/tmp/swebench_workspaces_spell_codex_toolcall_{ts}"),
        },
        {
            "name": "codex_harness",
            "condition": "codex",
            "model_spec": "codex",
            "agent_file": None,
            "adapter": codex_harness,
            "workspace_root": Path(f"/tmp/swebench_workspaces_codex_harness_{ts}"),
        },
        {
            "name": "spell_codex_messages",
            "condition": "spell",
            "model_spec": "codex:gpt-5.3-codex",
            "agent_file": spell_message_agent.agent_file,
            "adapter": spell_messages,
            "workspace_root": Path(f"/tmp/swebench_workspaces_spell_codex_messages_{ts}"),
        },
    ]

    results = {
        "run_timestamp": ts,
        "dataset": "swebench_lite",
        "n_items": len(items),
        "reasoning_effort": "high",
        "items": [
            {
                "instance_id": i.instance_id,
                "repo": i.repo,
                "base_commit": i.base_commit,
                "problem_statement": i.problem_statement,
                "hints_text": i.hints_text,
                "gold_patch": i.patch,
                "test_patch": i.test_patch,
            }
            for i in items
        ],
        "runs": [],
    }

    total = len(configs) * len(items)
    idx = 0
    for cfg in configs:
        root = cfg["workspace_root"]
        root.mkdir(parents=True, exist_ok=True)
        cache = RepoCache(root)
        print(f"\n=== Config: {cfg['name']} ({cfg['model_spec']}) ===", flush=True)

        for item in items:
            idx += 1
            print(f"[{idx}/{total}] {item.instance_id} ...", flush=True)

            wt = setup_swebench_workspace(cache, item, tag=cfg["name"])
            ws_path = wt.path
            run_start = time.time()
            prompt = make_prompt(item)

            req = RunRequest(
                benchmark="swebench",
                dataset="lite",
                item_id=item.instance_id,
                condition=cfg["condition"],
                prompt=prompt,
                model_spec=cfg["model_spec"],
                timeout=1200,
                init=(
                    build_spell_init(
                        prompt,
                        '(quine sys-reminder "Recall: you are writing Spell code, not natural language.")',
                    )
                    if cfg["condition"] == "spell"
                    else None
                ),
                agent_file=cfg["agent_file"],
                budget=10.0,
                depth=8,
                max_tokens=None,
                reasoning_effort="high",
                workspace=ws_path,
            )

            resp = cfg["adapter"].run(req)
            patch = wt.diff()
            changed_files = wt.changed_files()
            wall_ms = (time.time() - run_start) * 1000.0

            print(
                f"    status={resp.status} latency_ms={resp.latency_ms} wall_ms={wall_ms:.0f} "
                f"changed_files={len(changed_files)} patch_chars={len(patch)}",
                flush=True,
            )

            results["runs"].append(
                {
                    "config": cfg["name"],
                    "condition": cfg["condition"],
                    "model_spec": cfg["model_spec"],
                    "instance_id": item.instance_id,
                    "workspace": str(ws_path),
                    "request": {
                        "reasoning_effort": "high",
                        "budget": 10.0,
                        "depth": 8,
                        "max_tokens": None,
                        "timeout": 1200,
                    },
                    "response": asdict(resp),
                    "patch": patch,
                    "changed_files": changed_files,
                    "wall_ms": wall_ms,
                }
            )

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
