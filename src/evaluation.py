"""Evaluation and scoring for benchmark results."""

import re
from dataclasses import asdict

from .datasets import BenchmarkItem
from .runners import RunResult


def extract_mcq_answer(response: str) -> str | None:
    """Extract A/B/C/D answer from response."""
    response = response.strip().upper()

    # Direct letter
    if response in ('A', 'B', 'C', 'D'):
        return response

    # Letter with punctuation: "A.", "A:", "(A)", etc.
    match = re.match(r'^[(\[]?([A-D])[)\].:,]?$', response)
    if match:
        return match.group(1)

    # "The answer is A" or similar
    match = re.search(r'(?:answer|choice|option)\s*(?:is|:)?\s*[(\[]?([A-D])[)\].]?', response, re.I)
    if match:
        return match.group(1).upper()

    # First letter if it's A-D
    if response and response[0] in 'ABCD':
        return response[0]

    return None


def extract_freeform_answer(response: str) -> str:
    """Extract free-form answer (for BABILong)."""
    # Take the first word/phrase, strip punctuation
    response = response.strip().lower()

    # Remove common prefixes (order matters - longer first)
    prefixes = [
        'the answer is', 'answer:', 'answer is', 'it is',
        'in the', 'at the', 'to the', 'on the',  # Location prefixes
        'the', 'a ', 'an ',
    ]
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    # Take first word (BABILong answers are typically single words)
    response = response.split()[0] if response.split() else ''

    # Strip punctuation
    response = re.sub(r'[.,!?;:]$', '', response)

    return response


def evaluate_result(result: RunResult, item: BenchmarkItem) -> RunResult:
    """Add extracted answer and correctness to result."""
    if item.choices:
        # MCQ
        result.answer_extracted = extract_mcq_answer(result.response)
        result.correct = (result.answer_extracted == item.answer) if result.answer_extracted else False
    else:
        # Free-form
        result.answer_extracted = extract_freeform_answer(result.response)
        result.correct = (result.answer_extracted == item.answer.lower())

    return result


def compute_metrics(results: list[RunResult]) -> dict:
    """Compute aggregate metrics from results."""
    if not results:
        return {}

    total = len(results)
    errors = sum(1 for r in results if r.error)
    valid = [r for r in results if not r.error]

    correct = sum(1 for r in valid if r.correct)
    accuracy = correct / len(valid) if valid else 0

    latencies = [r.latency_ms for r in valid]
    costs = [r.cost for r in valid if r.cost]

    return {
        'total': total,
        'errors': errors,
        'valid': len(valid),
        'correct': correct,
        'accuracy': accuracy,
        'latency_mean_ms': sum(latencies) / len(latencies) if latencies else None,
        'latency_p50_ms': sorted(latencies)[len(latencies) // 2] if latencies else None,
        'latency_p90_ms': sorted(latencies)[int(len(latencies) * 0.9)] if latencies else None,
        'cost_total': sum(costs) if costs else None,
        'cost_mean': sum(costs) / len(costs) if costs else None,
    }


def results_to_dict(results: list[RunResult]) -> list[dict]:
    """Convert results to serializable dicts."""
    return [asdict(r) for r in results]


def format_summary(metrics: dict, condition: str, model: str, dataset: str) -> str:
    """Format metrics as human-readable summary."""
    lines = [
        f"=== {condition.upper()} | {model} | {dataset} ===",
        f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['valid']})",
        f"Errors: {metrics['errors']}/{metrics['total']}",
    ]

    if metrics.get('latency_mean_ms'):
        lines.append(f"Latency: {metrics['latency_mean_ms']:.0f}ms mean, {metrics['latency_p50_ms']:.0f}ms p50")

    if metrics.get('cost_total'):
        lines.append(f"Cost: ${metrics['cost_total']:.4f} total, ${metrics['cost_mean']:.6f}/item")

    return '\n'.join(lines)
