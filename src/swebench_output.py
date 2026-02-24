"""Output generation for SWE-bench evaluation.

Generates JSONL files in the format expected by the SWE-bench evaluation harness.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .runners import SWERunResult


def format_prediction(result: SWERunResult, model_name: str = None) -> dict:
    """Format a single result as a SWE-bench prediction.

    Args:
        result: SWERunResult from running an agent
        model_name: Model name for the predictions file (overrides result.model)

    Returns:
        Dict in SWE-bench prediction format
    """
    return {
        'instance_id': result.instance_id,
        'model_name_or_path': model_name or result.model,
        'model_patch': result.patch,
    }


def write_predictions_jsonl(
    results: list[SWERunResult],
    output_path: Path | str,
    model_name: str = None,
) -> Path:
    """Write results to JSONL file for SWE-bench evaluation.

    Args:
        results: List of SWERunResult objects
        output_path: Path for output JSONL file
        model_name: Model name to use in predictions (overrides result.model)

    Returns:
        Path to the written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for result in results:
            prediction = format_prediction(result, model_name)
            f.write(json.dumps(prediction) + '\n')

    return output_path


def write_full_results_json(
    results: list[SWERunResult],
    output_path: Path | str,
    config: dict = None,
) -> Path:
    """Write full results (including metadata) to JSON file.

    Args:
        results: List of SWERunResult objects
        output_path: Path for output JSON file
        config: Optional config dict to include

    Returns:
        Path to the written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': config or {},
        'results': [asdict(r) for r in results],
        'summary': compute_summary(results),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output_path


def compute_summary(results: list[SWERunResult]) -> dict:
    """Compute summary statistics from results."""
    total = len(results)
    errors = sum(1 for r in results if r.error)
    with_patch = sum(1 for r in results if r.patch and r.patch.strip())

    latencies = [r.latency_ms for r in results if not r.error]
    costs = [r.cost for r in results if r.cost]

    return {
        'total': total,
        'errors': errors,
        'with_patch': with_patch,
        'patch_rate': with_patch / total if total > 0 else 0,
        'latency_mean_ms': sum(latencies) / len(latencies) if latencies else None,
        'latency_median_ms': sorted(latencies)[len(latencies) // 2] if latencies else None,
        'cost_total': sum(costs) if costs else None,
        'cost_mean': sum(costs) / len(costs) if costs else None,
    }


def get_output_path(
    condition: str,
    dataset: str,
    model: str = None,
    suffix: str = None,
) -> Path:
    """Generate a standard output path for results.

    Args:
        condition: 'spell' or 'claude_code'
        dataset: Dataset name (e.g., 'mini', 'verified')
        model: Optional model identifier
        suffix: Optional suffix (e.g., timestamp)

    Returns:
        Path like results/swebench/spell_mini_20260205.jsonl
    """
    results_dir = Path(__file__).parent.parent / 'results' / 'swebench'
    results_dir.mkdir(parents=True, exist_ok=True)

    parts = [condition, dataset]
    if model:
        # Sanitize model name for filename
        model_safe = model.replace(':', '_').replace('/', '_')
        parts.append(model_safe)
    if suffix:
        parts.append(suffix)

    filename = '_'.join(parts) + '.jsonl'
    return results_dir / filename
