"""Evaluation and scoring for benchmark results."""

from decimal import Decimal, InvalidOperation
import re
import string
from dataclasses import asdict, dataclass

from .datasets import BenchmarkItem


@dataclass
class RunResult:
    """Result of a single benchmark run."""
    item_id: str
    condition: str
    model: str
    response: str
    answer_extracted: str | None
    correct: bool | None  # None if extraction failed
    latency_ms: float
    tokens_in: int | None
    tokens_out: int | None
    cost: float | None
    error: str | None = None
    metadata: dict | None = None  # For Spell: llm_calls, depth, etc.


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

    Handles: 42, -42, 1,200, $42, 3.14, 6.3202e25093, etc.
    Returns normalized string (no commas, no $).
    """
    response = response.strip()
    number = r'-?\$?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?'

    def _normalize_numeric_token(token: str) -> str | None:
        cleaned = token.strip().replace('$', '').replace(',', '')
        try:
            Decimal(cleaned)
        except InvalidOperation:
            return None
        return cleaned

    # Look for "#### N" pattern (GSM8K format in chain-of-thought)
    match = re.search(rf'####\s*({number})', response)
    if match:
        return _normalize_numeric_token(match.group(1))

    # Check if response ends with just a number on its own line (common pattern)
    lines = response.split('\n')
    last_line = lines[-1].strip() if lines else ''
    # Strip markdown bold/italic markers
    last_line = re.sub(r'^\*{1,3}|\*{1,3}$', '', last_line).strip()
    if re.match(rf'^{number}$', last_line):
        return _normalize_numeric_token(last_line)

    # Look for "answer is N" or "= N" patterns
    patterns = [
        rf'(?:answer|result|total|sum|value)\s*(?:is|=|:)\s*({number})',
        rf'=\s*({number})\s*$',  # Ending with = N
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.I | re.MULTILINE)
        if match:
            normalized = _normalize_numeric_token(match.group(1))
            if normalized is not None:
                return normalized

    # Fallback: find any number in the response (last one is often the answer)
    numbers = re.findall(number, response)
    # Filter out empty matches and very short numbers that are likely not answers
    numbers = [n for n in numbers if n and len(n.replace(',', '').replace('.', '').replace('$', '')) > 0]
    if numbers:
        return _normalize_numeric_token(numbers[-1])

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


def _extract_braced(text: str, open_brace_idx: int) -> tuple[str | None, int]:
    """Extract balanced {...} content starting at open_brace_idx."""
    if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != '{':
        return None, open_brace_idx

    depth = 0
    for i in range(open_brace_idx, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[open_brace_idx + 1:i], i + 1

    return None, open_brace_idx


def _strip_math_delimiters(s: str) -> str:
    s = s.strip()
    changed = True
    while changed and s:
        changed = False
        wrappers = (
            ('$$', '$$'),
            ('\\[', '\\]'),
            ('\\(', '\\)'),
            ('$', '$'),
        )
        for left, right in wrappers:
            if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
                s = s[len(left):-len(right)].strip()
                changed = True
                break
    return s


def _extract_boxed_content(response: str) -> str | None:
    for token in ('\\boxed{', '\\fbox{'):
        idx = response.find(token)
        if idx == -1:
            continue
        content, _ = _extract_braced(response, idx + len(token) - 1)
        if content is not None:
            return _strip_math_delimiters(content)
    return None


def _replace_latex_fractions(text: str) -> str:
    """Convert \\frac-like commands to explicit divisions."""
    out: list[str] = []
    i = 0
    frac_cmds = ('\\dfrac{', '\\tfrac{', '\\frac{')
    while i < len(text):
        best_idx = -1
        best_cmd = None
        for cmd in frac_cmds:
            idx = text.find(cmd, i)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_cmd = cmd

        if best_idx == -1 or best_cmd is None:
            out.append(text[i:])
            break

        out.append(text[i:best_idx])
        num, j = _extract_braced(text, best_idx + len(best_cmd) - 1)
        if num is None or j >= len(text) or text[j] != '{':
            out.append(best_cmd)
            i = best_idx + len(best_cmd)
            continue
        den, k = _extract_braced(text, j)
        if den is None:
            out.append(best_cmd)
            i = best_idx + len(best_cmd)
            continue

        out.append(f'(({num})/({den}))')
        i = k

    return ''.join(out)


def _replace_latex_sqrt(text: str) -> str:
    """Convert \\sqrt{...} to sqrt(...)."""
    out: list[str] = []
    i = 0
    token = '\\sqrt{'
    while i < len(text):
        idx = text.find(token, i)
        if idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:idx])
        arg, j = _extract_braced(text, idx + len(token) - 1)
        if arg is None:
            out.append(token)
            i = idx + len(token)
            continue
        out.append(f'sqrt({arg})')
        i = j
    return ''.join(out)


def _latex_to_sympy(text: str) -> str:
    s = _strip_math_delimiters(text)
    s = s.replace('−', '-').replace('–', '-')
    s = s.replace('π', 'pi')
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\cdot', '*').replace('\\times', '*')
    s = s.replace('\\,', '')
    s = _replace_latex_fractions(s)
    s = _replace_latex_sqrt(s)
    s = s.replace('\\pi', 'pi')
    s = s.replace('^', '**')
    # Handle simple assignment wrappers like x=...
    if '=' in s and re.match(r'^\s*[a-zA-Z]\s*=', s):
        s = s.split('=', 1)[1]
    s = s.replace('{', '(').replace('}', ')')
    s = s.replace('\\', '')
    return s.strip()


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_wrapping_quotes(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1].strip()
    return s


def _sympy_equivalent(got: str, expected: str) -> bool:
    import sympy
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (implicit_multiplication_application,)
    local_dict = {"sqrt": sympy.sqrt, "pi": sympy.pi}
    got_expr = parse_expr(_latex_to_sympy(got), transformations=transformations, local_dict=local_dict)
    expected_expr = parse_expr(_latex_to_sympy(expected), transformations=transformations, local_dict=local_dict)
    return bool(sympy.simplify(got_expr - expected_expr) == 0)


def extract_math_answer(response: str) -> str | None:
    """Extract answer for MATH competition problems.

    Handles: numbers, fractions (3/4), expressions (2\\sqrt{3}), etc.
    """
    response = response.strip()
    if not response:
        return None

    boxed = _extract_boxed_content(response)
    if boxed:
        return boxed

    # Handle spell-style wrappers like {:result "..."}, preserving braces in quoted LaTeX.
    map_match = re.search(r'\{:result\s+"((?:\\.|[^"\\])*)"\}\s*$', response, re.S)
    if map_match:
        candidate = _strip_math_delimiters(map_match.group(1).strip())
        if candidate:
            return candidate

    # Handle unquoted map values like {:result 123}
    map_match = re.search(r'\{:result\s+([^}]+)\}\s*$', response, re.S)
    if map_match:
        candidate = _strip_wrapping_quotes(_strip_math_delimiters(map_match.group(1).strip()))
        if candidate:
            return candidate

    # Check if response ends with just the answer on its own line
    lines = response.split('\n')
    last_line = lines[-1].strip() if lines else ''
    if last_line and len(last_line) < 200:
        return _strip_math_delimiters(last_line)

    # Look for "answer is X" pattern
    match = re.search(r'(?:answer|result)\s*(?:is|=|:)\s*(.+?)(?:\.|$)', response, re.I)
    if match:
        return _strip_math_delimiters(match.group(1).strip())

    return None


def normalize_math_answer(s: str) -> str:
    """Normalize math answer for string comparison."""
    if not s:
        return ''
    s = _latex_to_sympy(s)
    s = s.replace('°', '').replace('^\\circ', '').replace('\\circ', '')
    s = re.sub(r'\s+', '', s)
    return s.lower()


def math_equivalent(got: str, expected: str, timeout: float = 5.0) -> bool:
    """Check if two math expressions are equivalent using sympy.

    Handles LaTeX notation, fractions, degrees, etc.
    Returns True if expressions are mathematically equal.
    """
    # First try normalized string comparison
    if normalize_math_answer(got) == normalize_math_answer(expected):
        return True

    import signal

    def check() -> bool:
        try:
            got_parts = _split_top_level_commas(_strip_math_delimiters(got))
            expected_parts = _split_top_level_commas(_strip_math_delimiters(expected))

            if len(got_parts) > 1 or len(expected_parts) > 1:
                if len(got_parts) != len(expected_parts):
                    return False
                remaining = expected_parts[:]
                for g in got_parts:
                    g_norm = normalize_math_answer(g)
                    match_idx = None
                    for i, e in enumerate(remaining):
                        if g_norm == normalize_math_answer(e):
                            match_idx = i
                            break
                        try:
                            if _sympy_equivalent(g, e):
                                match_idx = i
                                break
                        except Exception:
                            continue
                    if match_idx is None:
                        return False
                    remaining.pop(match_idx)
                return True

            return _sympy_equivalent(got, expected)
        except Exception:
            return False

    def handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return check()
    except TimeoutError:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _gaia_normalize_str(s: str, remove_punct: bool = True) -> str:
    """Normalize string per GAIA official scorer: strip whitespace, lowercase, optionally remove punctuation."""
    no_spaces = re.sub(r'\s', '', s)
    if remove_punct:
        translator = str.maketrans('', '', string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _gaia_normalize_number(s: str) -> float:
    """Normalize number string per GAIA official scorer."""
    for char in ['$', '%', ',']:
        s = s.replace(char, '')
    try:
        return float(s)
    except ValueError:
        return float('inf')


def gaia_question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Score a GAIA answer using the official scoring logic.

    Handles three answer types: numbers, comma/semicolon-separated lists, and strings.
    """
    if model_answer is None:
        model_answer = 'None'

    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # If ground truth is a number
    if is_float(ground_truth):
        return _gaia_normalize_number(model_answer) == float(ground_truth)

    # If ground truth is a list (comma or semicolon separated)
    if any(ch in ground_truth for ch in [',', ';']):
        gt_elems = re.split(r'[,;]', ground_truth)
        ma_elems = re.split(r'[,;]', model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                if _gaia_normalize_number(ma_elem) != float(gt_elem):
                    return False
            else:
                if _gaia_normalize_str(ma_elem, remove_punct=False) != _gaia_normalize_str(gt_elem, remove_punct=False):
                    return False
        return True

    # String comparison
    return _gaia_normalize_str(model_answer) == _gaia_normalize_str(ground_truth)


def evaluate_browsecomp_judge(
    item: "BenchmarkItem",
    response_text: str,
    judge_model: str = "claude-opus-4-6",
) -> tuple[str | None, bool | None]:
    """LLM-as-judge evaluation for BrowseComp-Plus items.

    Returns (extracted_answer, is_correct).
    Uses the Anthropic API directly (not via Spell).
    """
    try:
        import anthropic
    except ImportError:
        # Fallback: simple string match
        ans = response_text.strip()
        return ans, ans.lower() == item.answer.strip().lower()

    client = anthropic.Anthropic()
    prompt = (
        f"You are a judge evaluating whether a model's answer is correct.\n\n"
        f"Question: {item.question}\n\n"
        f"Expected answer: {item.answer}\n\n"
        f"Model's response:\n{response_text}\n\n"
        f"Is the model's response correct? Extract the model's final answer and determine if it matches "
        f"the expected answer. Reply with exactly one of:\n"
        f"CORRECT: <extracted answer>\n"
        f"INCORRECT: <extracted answer>\n"
        f"Be flexible about phrasing and formatting — what matters is factual equivalence."
    )
    try:
        msg = client.messages.create(
            model=judge_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        judge_text = msg.content[0].text.strip()
        if judge_text.upper().startswith("CORRECT:"):
            extracted = judge_text[len("CORRECT:"):].strip()
            return extracted, True
        elif judge_text.upper().startswith("INCORRECT:"):
            extracted = judge_text[len("INCORRECT:"):].strip()
            return extracted, False
        # Fallback if format unexpected
        is_correct = "CORRECT" in judge_text.upper() and "INCORRECT" not in judge_text.upper()
        return judge_text, is_correct
    except Exception:
        ans = response_text.strip()
        return ans, ans.lower() == item.answer.strip().lower()


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
