#!/usr/bin/env python3
"""Main entry point for running benchmarks."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.datasets import DATASETS, load_longbench_short, load_babilong
from src.runners import BaselineRunner, SpellRunner, run_batch
from src.evaluation import evaluate_result, compute_metrics, results_to_dict, format_summary
import threading


def main():
    parser = argparse.ArgumentParser(description='Run Spell benchmarks')
    parser.add_argument('--dataset', type=str, default='longbench_short',
                        choices=list(DATASETS.keys()) + ['all'],
                        help='Dataset to run')
    parser.add_argument('--condition', type=str, default='baseline',
                        choices=['baseline', 'spell', 'all'],
                        help='Condition to run')
    parser.add_argument('--model', type=str, default='anthropic:claude-sonnet-4-20250514',
                        help='Model spec (provider:model)')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of items to run (default: all)')
    parser.add_argument('--budget', type=float, default=1.0,
                        help='Budget per item for Spell condition')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (default: results/<timestamp>.json)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print config and exit')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (default: 1 = sequential)')

    args = parser.parse_args()

    # Load datasets
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'all':
        items = []
        for name, loader in DATASETS.items():
            items.extend(loader())
    else:
        items = DATASETS[args.dataset]()

    if args.n:
        items = items[:args.n]

    print(f"Loaded {len(items)} items")

    if args.dry_run:
        print(f"Config: {args}")
        print(f"First item: {items[0] if items else 'none'}")
        return

    # Determine conditions to run
    conditions = ['baseline', 'spell'] if args.condition == 'all' else [args.condition]

    all_results = {}
    all_metrics = {}

    for condition in conditions:
        print(f"\n--- Running {condition} condition ---")

        if condition == 'baseline':
            runner = BaselineRunner(args.model)
        else:
            runner = SpellRunner(args.model, budget=args.budget)

        if args.parallel > 1:
            # Parallel execution
            print(f"  Running {len(items)} items with {args.parallel} workers...")
            completed = [0]
            lock = threading.Lock()

            def progress(item_id, result):
                with lock:
                    completed[0] += 1
                    status = '✓' if result.correct else ('✗' if not result.error else 'E')
                    print(f"  [{completed[0]}/{len(items)}] {item_id[:40]}... {status}")

            results = run_batch(items, runner, max_parallel=args.parallel, progress_callback=progress)
            # Evaluate results
            results = [evaluate_result(r, items[i]) for i, r in enumerate(results)]
        else:
            # Sequential execution
            results = []
            for i, item in enumerate(items):
                print(f"  [{i+1}/{len(items)}] {item.id[:40]}...", end=' ', flush=True)
                result = runner.run(item)
                result = evaluate_result(result, item)
                results.append(result)
                status = '✓' if result.correct else ('✗' if not result.error else 'E')
                print(f"{status} ({result.latency_ms:.0f}ms)")

        metrics = compute_metrics(results)
        print()
        print(format_summary(metrics, condition, args.model, args.dataset))

        all_results[condition] = results_to_dict(results)
        all_metrics[condition] = metrics

    # Save results
    output_path = args.output or f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(exist_ok=True)

    output = {
        'config': {
            'dataset': args.dataset,
            'model': args.model,
            'n_items': len(items),
            'conditions': conditions,
            'budget': args.budget,
            'timestamp': datetime.now().isoformat(),
        },
        'metrics': all_metrics,
        'results': all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print comparison if both conditions ran
    if len(conditions) == 2:
        print("\n=== COMPARISON ===")
        baseline_acc = all_metrics['baseline']['accuracy']
        spell_acc = all_metrics['spell']['accuracy']
        diff = spell_acc - baseline_acc
        print(f"Baseline accuracy: {baseline_acc:.1%}")
        print(f"Spell accuracy:    {spell_acc:.1%}")
        print(f"Difference:        {diff:+.1%}")


if __name__ == '__main__':
    main()
