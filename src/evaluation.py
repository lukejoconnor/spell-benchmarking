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

    # Last standalone A-D letter not followed by a period (period = list marker like "A. Choice text")
    matches = re.findall(r'(?<![A-Z])([A-D])(?![A-Z])(?!\.)', response)
    if matches:
        return matches[-1]

    return None


def extract_freeform_answer(response: str) -> str:
    """Extract free-form answer (for BABILong).

    Strategy: Look for the answer on the last non-empty line (models often
    put reasoning first, answer last), then fall back to first word.
    """
    response = response.strip().lower()

    # Split into lines, find last non-empty line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if not lines:
        return ''

    # Check if last line is a short answer (1-3 words, typical for BABILong)
    last_line = lines[-1]
    last_words = last_line.split()
    if 1 <= len(last_words) <= 3:
        # Use last line as answer
        answer = last_line
    else:
        # Fall back to first word of response
        answer = response.split()[0] if response.split() else ''

    # Remove common prefixes
    prefixes = [
        'the answer is', 'answer:', 'answer is', 'it is',
        'in the', 'at the', 'to the', 'on the',
        'the', 'a ', 'an ',
    ]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # Take first word if still multi-word
    answer = answer.split()[0] if answer.split() else ''

    # Strip punctuation
    answer = re.sub(r'[.,!?;:]$', '', answer)

    return answer


def extract_numeric_answer(response: str) -> str | None:
    """Extract numeric answer (for GSM8K math problems).

    Handles: 42, -42, 1,200, $42, 3.14, etc.
    Returns normalized string (no commas, no $).
    """
    response = response.strip()

    # Look for "#### N" pattern (GSM8K format in chain-of-thought)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', response)
    if match:
        return match.group(1).replace(',', '')

    # Check if response ends with just a number on its own line (common pattern)
    lines = response.split('\n')
    last_line = lines[-1].strip() if lines else ''
    if re.match(r'^-?\$?[\d,]+\.?\d*$', last_line):
        return last_line.replace('$', '').replace(',', '')

    # Look for "answer is N" or "= N" patterns
    patterns = [
        r'(?:answer|result|total|sum|value)\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)',
        r'=\s*\$?(-?[\d,]+\.?\d*)\s*$',  # Ending with = N
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.I | re.MULTILINE)
        if match:
            num = match.group(1).replace(',', '')
            try:
                float(num)
                return num
            except ValueError:
                continue

    # Fallback: find any number in the response (last one is often the answer)
    numbers = re.findall(r'-?[\d,]+\.?\d*', response)
    # Filter out empty matches and very short numbers that are likely not answers
    numbers = [n for n in numbers if n and len(n.replace(',', '').replace('.', '')) > 0]
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def normalize_numeric(s: str) -> str:
    """Normalize numeric string for comparison."""
    if not s:
        return ''
    # Remove commas, leading zeros, trailing .0
    s = s.replace(',', '')
    try:
        f = float(s)
        # Return as int if whole number, else float
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def extract_math_answer(response: str) -> str | None:
    """Extract answer for MATH competition problems.

    Handles: numbers, fractions (3/4), expressions (2\\sqrt{3}), etc.
    """
    import re
    response = response.strip()

    # Look for \boxed{...} pattern (MATH format)
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()

    # Check if response ends with just the answer on its own line
    lines = response.split('\n')
    last_line = lines[-1].strip() if lines else ''
    if last_line and len(last_line) < 50:  # Reasonable answer length
        return last_line

    # Look for "answer is X" pattern
    match = re.search(r'(?:answer|result)\s*(?:is|=|:)\s*(.+?)(?:\.|$)', response, re.I)
    if match:
        return match.group(1).strip()

    return None


def normalize_math_answer(s: str) -> str:
    """Normalize math answer for string comparison."""
    if not s:
        return ''
    # Remove whitespace
    s = s.strip().replace(' ', '')
    # Remove dollar signs (LaTeX math mode)
    s = s.replace('$', '')
    # Remove degree symbols for comparison
    s = s.replace('°', '').replace('^\\circ', '').replace('\\circ', '')
    # Normalize LaTeX fractions to plain fractions
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)
    # Remove other LaTeX commands
    s = s.replace('\\sqrt', 'sqrt').replace('\\left', '').replace('\\right', '')
    s = s.replace('\\', '')
    return s.lower()


def math_equivalent(got: str, expected: str, timeout: float = 5.0) -> bool:
    """Check if two math expressions are equivalent using sympy.

    Handles LaTeX notation, fractions, degrees, etc.
    Returns True if expressions are mathematically equal.
    """
    import signal

    # First try normalized string comparison
    if normalize_math_answer(got) == normalize_math_answer(expected):
        return True

    # Try sympy equivalence with timeout
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        def check():
            try:
                # Clean up for sympy parsing
                def clean_for_sympy(s):
                    s = s.strip()
                    # Remove degree notation
                    s = s.replace('°', '').replace('^\\circ', '').replace('\\circ', '')
                    # Handle x= prefix
                    if s.lower().startswith('x=') or s.lower().startswith('x ='):
                        s = s.split('=', 1)[1].strip()
                    return s

                got_clean = clean_for_sympy(got)
                exp_clean = clean_for_sympy(expected)

                # Try parsing as LaTeX
                try:
                    expr_got = parse_latex(got_clean)
                    expr_exp = parse_latex(exp_clean)
                except:
                    # Fall back to sympify for plain expressions
                    expr_got = sympy.sympify(got_clean)
                    expr_exp = sympy.sympify(exp_clean)

                # Check equivalence
                diff = sympy.simplify(expr_got - expr_exp)
                return diff == 0
            except:
                return False

        # Run with timeout (signal-based, Unix only)
        def handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            result = check()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

        return result

    except Exception:
        return False


def evaluate_result(result: RunResult, item: BenchmarkItem) -> RunResult:
    """Add extracted answer and correctness to result."""
    if item.choices:
        # MCQ
        result.answer_extracted = extract_mcq_answer(result.response)
        result.correct = (result.answer_extracted == item.answer) if result.answer_extracted else False
    elif item.dataset == 'gsm8k':
        # Numeric (math problems)
        result.answer_extracted = extract_numeric_answer(result.response)
        if result.answer_extracted:
            result.correct = normalize_numeric(result.answer_extracted) == normalize_numeric(item.answer)
        else:
            result.correct = False
    elif item.dataset in ('math', 'omni_math', 'hmmt'):
        # Competition math (can be expressions, fractions, etc.)
        result.answer_extracted = extract_math_answer(result.response)
        if result.answer_extracted:
            result.correct = math_equivalent(result.answer_extracted, item.answer)
        else:
            result.correct = False
    elif item.dataset == 'aime':
        # AIME answers are always integers 0-999
        result.answer_extracted = extract_numeric_answer(result.response)
        if result.answer_extracted:
            try:
                got = int(float(result.answer_extracted))
                expected = int(item.answer)
                result.correct = (got == expected)
            except ValueError:
                result.correct = False
        else:
            result.correct = False
    else:
        # Free-form text
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
