"""Dataset loading and formatting for benchmark tasks."""

import html as _html
import json as _json
import os
import random
import re as _re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from datasets import load_dataset
from huggingface_hub import snapshot_download

_BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SWEBenchItem:
    """Item from SWE-bench dataset for GitHub issue resolution."""
    instance_id: str       # e.g., "django__django-16379"
    repo: str              # e.g., "django/django"
    base_commit: str       # SHA to checkout
    problem_statement: str # Issue description
    hints_text: str        # Optional hints
    test_patch: str        # Tests that verify fix
    patch: str             # Gold patch (not given to model)
    metadata: dict = field(default_factory=dict)


@dataclass
class ScienceAgentBenchItem:
    """Item from ScienceAgentBench for data-driven scientific discovery tasks."""
    instance_id: str           # Numeric ID (1-102)
    domain: str                # e.g., "Computational Chemistry"
    subtask_categories: str    # e.g., "Feature Engineering, Deep Learning"
    task_inst: str             # Task instruction
    domain_knowledge: str      # Expert domain knowledge
    dataset_folder_tree: str   # File tree of input data
    dataset_preview: str       # Preview of dataset files
    output_fname: str          # Expected output filename
    gold_program_name: str     # Gold program filename
    eval_script_name: str      # Evaluation script filename
    github_name: str           # Source GitHub repo
    metadata: dict = field(default_factory=dict)


@dataclass
class ExercismItem:
    """Item from Exercism Python practice exercises."""
    slug: str              # e.g., "hello-world"
    name: str              # e.g., "Hello World"
    difficulty: int        # 1-9
    instructions: str      # Markdown instructions
    stub_file: str         # e.g., "hello_world.py"
    stub_content: str      # Initial stub code
    test_file: str         # e.g., "hello_world_test.py"
    test_content: str      # Pytest test file
    extra_files: list[dict] = field(default_factory=list)  # [{name, content}]
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkItem:
    """Unified format for benchmark items."""
    id: str
    dataset: str
    context: str
    question: str
    answer: str  # Ground truth
    choices: list[str] | None = None  # For MCQ
    metadata: dict = None


def load_longbench_short(max_chars: int = 100_000) -> list[BenchmarkItem]:
    """Load LongBench v2 examples with context < max_chars."""
    ds = load_dataset('THUDM/LongBench-v2', split='train')

    items = []
    for row in ds:
        if len(row['context']) >= max_chars:
            continue

        items.append(BenchmarkItem(
            id=row['_id'],
            dataset='longbench_short',
            context=row['context'],
            question=row['question'],
            answer=row['answer'],
            choices=[
                row['choice_A'],
                row['choice_B'],
                row['choice_C'],
                row['choice_D'],
            ],
            metadata={
                'domain': row['domain'],
                'sub_domain': row['sub_domain'],
                'difficulty': row['difficulty'],
                'length': row['length'],
            }
        ))

    return items


def load_longbench_sampled(n_per_domain: int = 8, seed: int = 42) -> list[BenchmarkItem]:
    """Load LongBench v2 with N items sampled per domain (all context lengths)."""
    ds = load_dataset('THUDM/LongBench-v2', split='train')

    # Group by domain
    by_domain = {}
    for row in ds:
        domain = row['domain']
        by_domain.setdefault(domain, []).append(row)

    rng = random.Random(seed)
    items = []
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        sample = rng.sample(rows, min(n_per_domain, len(rows)))
        for row in sample:
            items.append(BenchmarkItem(
                id=row['_id'],
                dataset='longbench_sampled',
                context=row['context'],
                question=row['question'],
                answer=row['answer'],
                choices=[
                    row['choice_A'],
                    row['choice_B'],
                    row['choice_C'],
                    row['choice_D'],
                ],
                metadata={
                    'domain': domain,
                    'sub_domain': row['sub_domain'],
                    'difficulty': row['difficulty'],
                    'length': row['length'],
                    'context_chars': len(row['context']),
                }
            ))

    return items


def load_math(split: str = 'test', level: int = None, subject: str = None, max_items: int = None) -> list[BenchmarkItem]:
    """Load MATH competition math problems (nlile/hendrycks-MATH-benchmark).

    Args:
        split: 'train' or 'test'
        level: Filter by difficulty (1-5, where 5 is hardest)
        subject: Filter by subject (Algebra, Counting & Probability, Geometry,
                 Intermediate Algebra, Number Theory, Prealgebra, Precalculus)
        max_items: Optional limit on number of items
    """
    ds = load_dataset('nlile/hendrycks-MATH-benchmark', split=split)

    items = []
    for i, row in enumerate(ds):
        if max_items and len(items) >= max_items:
            break

        # Filter by level if specified (level is int 1-5)
        if level and row['level'] != level:
            continue

        # Filter by subject if specified
        if subject and row['subject'].lower() != subject.lower():
            continue

        items.append(BenchmarkItem(
            id=f'math_{split}_{i}',
            dataset='math',
            context='',  # Self-contained
            question=row['problem'],
            answer=row['answer'],  # Already extracted from \boxed{}
            choices=None,
            metadata={
                'level': row['level'],
                'subject': row['subject'],
                'full_solution': row['solution'],
            }
        ))

    return items


def load_gsm8k(split: str = 'test', max_items: int = None) -> list[BenchmarkItem]:
    """Load GSM8K grade school math problems.

    Args:
        split: 'train' or 'test'
        max_items: Optional limit on number of items
    """
    ds = load_dataset('openai/gsm8k', 'main', split=split)

    items = []
    for i, row in enumerate(ds):
        if max_items and i >= max_items:
            break

        # GSM8K answer format: "#### 42" at the end
        # Extract just the numeric answer
        answer_full = row['answer']
        answer = answer_full.split('####')[-1].strip()

        items.append(BenchmarkItem(
            id=f'gsm8k_{split}_{i}',
            dataset='gsm8k',
            context='',  # No separate context - question is self-contained
            question=row['question'],
            answer=answer,
            choices=None,  # Free-form numeric
            metadata={
                'full_answer': answer_full,  # Contains chain-of-thought
            }
        ))

    return items


def load_aime(year: int = 2025, max_items: int = None) -> list[BenchmarkItem]:
    """Load AIME competition problems.

    Args:
        year: Competition year (2024 or 2025 available)
        max_items: Optional limit on number of items
    """
    if year == 2026:
        ds = load_dataset('MathArena/aime_2026', split='train')
    elif year == 2025:
        ds = load_dataset('MathArena/aime_2025', split='train')
    elif year == 2024:
        ds = load_dataset('HuggingFaceH4/aime_2024', split='train')
    else:
        raise ValueError(f"AIME year {year} not available")

    items = []
    for row in ds:
        if max_items and len(items) >= max_items:
            break

        # AIME answers are always integers 000-999
        answer = str(row['answer'])

        items.append(BenchmarkItem(
            id=f'aime_{year}_{row["problem_idx"]}',
            dataset='aime',
            context='',  # Self-contained
            question=row['problem'],
            answer=answer,
            choices=None,
            metadata={
                'year': year,
                'problem_idx': row['problem_idx'],
                'problem_type': row.get('problem_type', []),
            }
        ))

    return items


def load_hmmt(competition: str = 'feb', year: int = 2025, max_items: int = None) -> list[BenchmarkItem]:
    """Load HMMT (Harvard-MIT Math Tournament) problems.

    Args:
        competition: 'feb' or 'nov'
        year: Competition year (2025 available)
        max_items: Optional limit on number of items
    """
    ds_name = f'MathArena/hmmt_{competition}_{year}'
    ds = load_dataset(ds_name, split='train')

    items = []
    for row in ds:
        if max_items and len(items) >= max_items:
            break

        items.append(BenchmarkItem(
            id=f'hmmt_{competition}_{year}_{row["problem_idx"]}',
            dataset='hmmt',
            context='',
            question=row['problem'],
            answer=str(row['answer']),
            choices=None,
            metadata={
                'competition': competition,
                'year': year,
                'problem_idx': row['problem_idx'],
                'problem_type': row.get('problem_type', []),
            }
        ))

    return items


def load_omni_math(min_difficulty: float = None, max_difficulty: float = None, max_items: int = None) -> list[BenchmarkItem]:
    """Load Omni-MATH olympiad-level problems (KbsdJames/Omni-MATH).

    4,430 problems with difficulty 1.0-9.5. At difficulty >5, even o1-mini gets ~49%.

    Args:
        min_difficulty: Minimum difficulty (inclusive), e.g. 6.0 for hard
        max_difficulty: Maximum difficulty (inclusive)
        max_items: Optional limit on number of items
    """
    ds = load_dataset('KbsdJames/Omni-MATH', split='test')

    items = []
    for i, row in enumerate(ds):
        if max_items and len(items) >= max_items:
            break

        difficulty = float(row['difficulty'])
        if min_difficulty is not None and difficulty < min_difficulty:
            continue
        if max_difficulty is not None and difficulty > max_difficulty:
            continue

        items.append(BenchmarkItem(
            id=f'omni_math_{i}',
            dataset='omni_math',
            context='',
            question=row['problem'],
            answer=row['answer'],
            choices=None,
            metadata={
                'difficulty': difficulty,
                'domain': row['domain'],
                'source': row['source'],
                'full_solution': row['solution'],
            }
        ))

    return items


PROJECT_EULER_REPO_URL = "https://github.com/Orbiter/project-euler-llm-benchmark.git"
PROJECT_EULER_BASE_URL = "https://projecteuler.net/minimal={problem_number}"
PROJECT_EULER_DIR = _BENCHMARK_ROOT / "data" / "project-euler-llm-benchmark"
MATHCONSTRUCT_REPO_URL = "https://github.com/eth-sri/mathconstruct.git"
MATHCONSTRUCT_DIR = _BENCHMARK_ROOT / "data" / "mathconstruct"
_MATHCONSTRUCT_CLASS_CACHE: dict[str, type] | None = None


def _ensure_project_euler_repo() -> Path:
    """Clone Orbiter's Project Euler benchmark repo if not already present."""
    if not PROJECT_EULER_DIR.is_dir():
        PROJECT_EULER_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", PROJECT_EULER_REPO_URL, str(PROJECT_EULER_DIR)],
            check=True,
        )
    return PROJECT_EULER_DIR


def _project_euler_problem_path(repo_dir: Path, problem_number: int) -> Path:
    return repo_dir / "problems" / f"{problem_number:04d}.txt"


def _download_project_euler_problem_text(problem_number: int, timeout_s: int = 30) -> str:
    """Fetch Project Euler problem statement from the minimal endpoint."""
    url = PROJECT_EULER_BASE_URL.format(problem_number=problem_number)
    req = urllib.request.Request(url, headers={"User-Agent": "spell-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    # minimal=<id> returns compact HTML fragments. Keep paragraph boundaries.
    text = raw.replace("<p>", "").replace("</p>", "\n")
    text = _re.sub(r"<[^>]+>", " ", text)
    text = _html.unescape(text)
    text = _re.sub(r"[ \t]+\n", "\n", text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_project_euler_problem(repo_dir: Path, problem_number: int) -> str:
    """Load cached problem statement or fetch it from Project Euler."""
    path = _project_euler_problem_path(repo_dir, problem_number)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = _download_project_euler_problem_text(problem_number)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Failed to download Project Euler problem {problem_number} ({exc})."
        ) from exc

    if not text:
        raise RuntimeError(f"Downloaded empty statement for Project Euler problem {problem_number}.")

    path.write_text(text + "\n", encoding="utf-8")
    return text


def load_project_euler(
    max_items: int | None = None,
    start_problem: int = 1,
    end_problem: int | None = None,
    problem_numbers: list[int] | None = None,
) -> list[BenchmarkItem]:
    """Load Project Euler benchmark problems and canonical numeric answers."""
    repo_dir = _ensure_project_euler_repo()
    solutions_path = repo_dir / "solutions.json"
    solutions = _json.loads(solutions_path.read_text(encoding="utf-8"))

    if problem_numbers is not None:
        requested = sorted({int(n) for n in problem_numbers})
        problem_keys = [f"{n:04d}" for n in requested if f"{n:04d}" in solutions]
    else:
        problem_keys = sorted(solutions.keys(), key=lambda k: int(k))
    items: list[BenchmarkItem] = []

    for key in problem_keys:
        problem_number = int(key)
        if problem_number < start_problem:
            continue
        if end_problem is not None and problem_number > end_problem:
            continue
        if max_items is not None and len(items) >= max_items:
            break

        row = solutions[key]
        statement = _load_project_euler_problem(repo_dir, problem_number)
        title = str(row.get("title", "")).strip()
        question = f"Project Euler Problem {problem_number}: {title}\n\n{statement}" if title else statement
        answer = str(row.get("solution", "")).strip()

        items.append(BenchmarkItem(
            id=f"project_euler_{problem_number:04d}",
            dataset="project_euler",
            context="",
            question=question,
            answer=answer,
            choices=None,
            metadata={
                "problem_number": problem_number,
                "title": title,
                "points": row.get("points"),
                "percentage_solved": row.get("percentage_solved"),
                "project_euler_url": f"https://projecteuler.net/problem={problem_number}",
            },
        ))

    return items


def _ensure_mathconstruct_repo() -> Path:
    """Clone eth-sri/mathconstruct if not already present."""
    if not MATHCONSTRUCT_DIR.is_dir():
        MATHCONSTRUCT_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", MATHCONSTRUCT_REPO_URL, str(MATHCONSTRUCT_DIR)],
            check=True,
        )
    return MATHCONSTRUCT_DIR


def _load_mathconstruct_problem_classes() -> dict[str, type]:
    """Load MathConstruct problem classes keyed by benchmark problem name."""
    global _MATHCONSTRUCT_CLASS_CACHE
    if _MATHCONSTRUCT_CLASS_CACHE is not None:
        return _MATHCONSTRUCT_CLASS_CACHE

    repo_dir = _ensure_mathconstruct_repo()
    src_dir = repo_dir / "src"
    src_str = str(src_dir.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    from math_construct.problems import get_all_problem_classes  # type: ignore[import-not-found]

    classes: dict[str, type] = {}
    for cls in get_all_problem_classes():
        classes[cls.config.name] = cls
    _MATHCONSTRUCT_CLASS_CACHE = classes
    return classes


def load_mathconstruct(
    max_items: int | None = None,
    problem_names: list[str] | None = None,
) -> list[BenchmarkItem]:
    """Load MathConstruct problems from a local clone.

    Uses one deterministic instance per problem (the original parameters).
    """
    classes = _load_mathconstruct_problem_classes()
    selected_names = sorted(problem_names) if problem_names else sorted(classes.keys())

    items: list[BenchmarkItem] = []
    for problem_name in selected_names:
        if max_items is not None and len(items) >= max_items:
            break
        cls = classes.get(problem_name)
        if cls is None:
            continue

        instance = cls.generate_multiple(1)[0]
        statement = instance.get_problem().strip()
        formatting = (instance.get_formatting() or "").strip()
        question = statement
        if formatting:
            question = f"{statement}\n\nFormatting instructions:\n{formatting}"

        items.append(BenchmarkItem(
            id=f"mathconstruct_{problem_name}",
            dataset="mathconstruct",
            context="",
            question=question,
            answer=str(instance.get_solution()),
            choices=None,
            metadata={
                "mathconstruct_problem_name": problem_name,
                "mathconstruct_problem_json": instance.to_json(),
                "formatting_instructions": formatting,
                "source": instance.config.source,
                "problem_url": instance.config.problem_url,
            },
        ))

    return items


def evaluate_mathconstruct_response(item: BenchmarkItem, response_text: str | None) -> tuple[str | None, bool | None, str]:
    """Score a MathConstruct response via the benchmark's native parser/checker."""
    if not response_text:
        return None, False, "Empty response"

    metadata: dict[str, Any] = item.metadata or {}
    problem_name = metadata.get("mathconstruct_problem_name")
    problem_json = metadata.get("mathconstruct_problem_json")
    if not problem_name or not isinstance(problem_json, dict):
        return None, False, "Missing MathConstruct metadata"

    classes = _load_mathconstruct_problem_classes()
    cls = classes.get(str(problem_name))
    if cls is None:
        return None, False, f"Unknown MathConstruct problem: {problem_name}"

    try:
        instance = cls.from_json(problem_json)
        parsed_answer, is_correct, details = instance.parse_and_check(
            [{"role": "assistant", "content": response_text}]
        )
    except Exception as exc:
        return None, False, f"MathConstruct evaluation error: {exc}"

    extracted = None if parsed_answer is None else str(parsed_answer)
    return extracted, bool(is_correct), details


def load_babilong(context_size: str = '32k', tasks: list[str] = None) -> list[BenchmarkItem]:
    """Load BABILong at specified context size.

    Args:
        context_size: One of '0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k'
        tasks: List of task names like ['qa1', 'qa2', ...]. Default: qa1-qa10
    """
    if tasks is None:
        tasks = [f'qa{i}' for i in range(1, 11)]

    ds = load_dataset('RMT-team/babilong', context_size)

    items = []
    for task in tasks:
        if task not in ds:
            continue
        for i, row in enumerate(ds[task]):
            items.append(BenchmarkItem(
                id=f'{task}_{context_size}_{i}',
                dataset=f'babilong_{context_size}',
                context=row['input'],
                question=row['question'],
                answer=row['target'],
                choices=None,  # Free-form
                metadata={
                    'task': task,
                    'context_size': context_size,
                }
            ))

    return items


LONGBENCH_SEARCH_GUIDANCE = (
    "The document may be very large. Do not read the entire file at once. "
    "Search for relevant keywords first, then inspect only the specific sections you need."
)


def format_benchmark_prompt(item: BenchmarkItem, context_file: str | None = None) -> str:
    """Format a benchmark item without execution-condition-specific wording."""
    parts = []

    if item.context and context_file:
        doc_len = len(item.context)
        parts.extend([
            f'Document path: {context_file}',
            f'Document size: {doc_len:,} chars',
            '',
        ])
        if item.dataset in ('longbench_sampled', 'longbench_short'):
            parts.extend([LONGBENCH_SEARCH_GUIDANCE, ''])
        parts.extend([
            'Question:',
            item.question.strip(),
        ])
    elif item.context:
        parts.extend([
            'Context:',
            item.context,
            '',
            'Question:',
            item.question.strip(),
        ])
    elif item.dataset == 'gaia':
        parts.append(item.question.strip())
        file_path = (item.metadata or {}).get('file_path', '')
        if file_path:
            parts.extend(['', f'Attached file: {file_path}'])
    else:
        parts.extend([
            'Problem:',
            '',
            item.question.strip(),
        ])

    if item.choices:
        parts.append('')
        parts.append('Choices:')
        for i, choice in enumerate(item.choices):
            letter = chr(ord('A') + i)
            parts.append(f'  {letter}. {choice}')
        parts.append('')
        parts.append('Return just the letter (A, B, C, or D).')
    elif item.dataset in ('gsm8k', 'math'):
        parts.append('')
        parts.append('Return the answer with no explanation (e.g., 42 or 3/4 or 2\\sqrt{3}).')
    elif item.dataset == 'mathconstruct':
        parts.append('')
        parts.append('Follow the formatting instructions exactly and put the final answer in \\boxed{...}.')
    elif item.dataset == 'project_euler':
        parts.append('')
        parts.append('Solve efficiently. Return just the final integer answer, no explanation.')
    elif item.dataset == 'omni_math':
        parts.append('')
        parts.append('Return the answer with no explanation (e.g., 42 or 3/4 or 2\\sqrt{3}).')
    elif item.dataset == 'aime':
        parts.append('')
        parts.append('Return the answer with no explanation (AIME answers are always integers 0-999).')
    elif item.dataset == 'hmmt':
        parts.append('')
        parts.append('Return the answer with no explanation (e.g., 42 or 3/4 or 2\\sqrt{3}).')
    elif item.dataset == 'gaia':
        parts.append('')
        parts.append('Use web search, Python, and any tools needed to answer this question.')
        parts.append('Return ONLY the final answer — a number, a short string, or a comma-separated list. No explanation.')
    else:
        parts.append('')
        parts.append('Return just the answer (e.g., "bathroom").')

    return "\n".join(parts)

def load_gaia(level: int = None, max_items: int = None) -> list[BenchmarkItem]:
    """Load GAIA benchmark for general AI assistant evaluation.

    165 validation items across 3 levels. Questions require web search,
    file reading, reasoning, and tool use. Some items have file attachments.

    Args:
        level: Filter by difficulty (1=easy, 2=medium, 3=hard). None = all.
        max_items: Optional limit on number of items.
    """
    ds = load_dataset('gaia-benchmark/GAIA', '2023_all', split='validation')

    # Download repo snapshot for file attachments
    data_dir = snapshot_download(repo_id='gaia-benchmark/GAIA', repo_type='dataset')

    items = []
    for row in ds:
        if max_items and len(items) >= max_items:
            break

        row_level = int(row['Level'])
        if level is not None and row_level != level:
            continue

        # Resolve file path if present
        file_path = None
        if row.get('file_name') and row.get('file_path'):
            file_path = os.path.join(data_dir, row['file_path'])

        metadata = {
            'level': row_level,
            'file_name': row.get('file_name', ''),
            'file_path': file_path or '',
            'annotator_steps': (row.get('Annotator Metadata') or {}).get('Steps', ''),
        }

        items.append(BenchmarkItem(
            id=row['task_id'],
            dataset='gaia',
            context='',  # No bulk context — questions are self-contained or need web/file access
            question=row['Question'],
            answer=row['Final answer'],
            choices=None,
            metadata=metadata,
        ))

    return items


# =============================================================================
# Exercism Dataset Loader
# =============================================================================

EXERCISM_REPO_URL = "https://github.com/exercism/python.git"
EXERCISM_DIR = _BENCHMARK_ROOT / "data" / "exercism-python"


def _ensure_exercism_repo() -> Path:
    """Clone exercism/python if not already present."""
    if not EXERCISM_DIR.is_dir():
        EXERCISM_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", EXERCISM_REPO_URL, str(EXERCISM_DIR)],
            check=True,
        )
    return EXERCISM_DIR


def _slug_to_snake(slug: str) -> str:
    return slug.replace("-", "_")


def load_exercism(
    difficulty_min: int | None = None,
    difficulty_max: int | None = None,
    slugs: set[str] | None = None,
    max_items: int | None = None,
) -> list[ExercismItem]:
    """Load Exercism Python practice exercises.

    Args:
        difficulty_min: Minimum difficulty (inclusive, 1-9)
        difficulty_max: Maximum difficulty (inclusive, 1-9)
        slugs: If provided, only load these specific exercise slugs
        max_items: Optional limit on number of items
    """
    repo = _ensure_exercism_repo()
    config_path = repo / "config.json"
    config = _json.loads(config_path.read_text(encoding="utf-8"))

    items: list[ExercismItem] = []
    for ex in config["exercises"]["practice"]:
        if max_items and len(items) >= max_items:
            break

        slug = ex["slug"]
        difficulty = ex.get("difficulty", 0)
        status = ex.get("status", "active")

        if status == "deprecated":
            continue
        if slugs and slug not in slugs:
            continue
        if difficulty_min is not None and difficulty < difficulty_min:
            continue
        if difficulty_max is not None and difficulty > difficulty_max:
            continue

        snake = _slug_to_snake(slug)
        base_dir = repo / "exercises" / "practice" / slug
        instructions_file = base_dir / ".docs" / "instructions.md"
        stub_file = base_dir / f"{snake}.py"
        test_file = base_dir / f"{snake}_test.py"

        if not all(f.exists() for f in (instructions_file, stub_file, test_file)):
            continue

        # Collect extra .py files (e.g., test_utils.py)
        extra_files = []
        for py_file in sorted(base_dir.glob("*.py")):
            name = py_file.name
            if name in (f"{snake}.py", f"{snake}_test.py"):
                continue
            extra_files.append({"name": name, "content": py_file.read_text(encoding="utf-8")})

        items.append(ExercismItem(
            slug=slug,
            name=ex.get("name", slug),
            difficulty=difficulty,
            instructions=instructions_file.read_text(encoding="utf-8"),
            stub_file=f"{snake}.py",
            stub_content=stub_file.read_text(encoding="utf-8"),
            test_file=f"{snake}_test.py",
            test_content=test_file.read_text(encoding="utf-8"),
            extra_files=extra_files,
            metadata={
                "topics": ex.get("topics", []),
            },
        ))

    return items


EXERCISM_DATASETS = {
    'exercism': load_exercism,
    'exercism_d1_d3': lambda: load_exercism(difficulty_min=1, difficulty_max=3),
    'exercism_d4_d5': lambda: load_exercism(difficulty_min=4, difficulty_max=5),
    'exercism_d5_d9': lambda: load_exercism(difficulty_min=5, difficulty_max=9),
}


# Dataset registry
DATASETS = {
    'longbench_short': load_longbench_short,
    'longbench_sampled': load_longbench_sampled,
    'babilong_32k': lambda: load_babilong('32k'),
    'babilong_32k_hard': lambda: load_babilong('32k', tasks=['qa2', 'qa3']),  # Multi-hop
    'babilong_32k_qa2': lambda: load_babilong('32k', tasks=['qa2']),  # QA2 only
    'babilong_16k': lambda: load_babilong('16k'),
    'babilong_8k': lambda: load_babilong('8k'),
    'gsm8k': lambda: load_gsm8k('test'),
    'gsm8k_train': lambda: load_gsm8k('train', max_items=100),  # Sample for dev
    'math': lambda: load_math('test'),  # Full MATH test set
    'math_easy': lambda: load_math('test', level=1),  # Level 1 (easiest)
    'math_medium': lambda: load_math('test', level=3),  # Level 3
    'math_hard': lambda: load_math('test', level=5),  # Level 5 (hardest)
    'math_algebra': lambda: load_math('test', subject='algebra'),
    'mathconstruct': lambda: load_mathconstruct(),  # MathConstruct constructive-proof benchmark
    'project_euler': lambda: load_project_euler(),  # Full Project Euler set with canonical answers
    'project_euler_200': lambda: load_project_euler(max_items=200),  # First 200 problems
    'aime_2026': lambda: load_aime(2026),  # AIME 2026 (30 problems)
    'aime_2025': lambda: load_aime(2025),  # AIME 2025 (30 problems)
    'aime_2024': lambda: load_aime(2024),  # AIME 2024 (30 problems)
    'hmmt_feb_2025': lambda: load_hmmt('feb', 2025),  # HMMT Feb 2025 (30 problems)
    'hmmt_nov_2025': lambda: load_hmmt('nov', 2025),  # HMMT Nov 2025 (30 problems)
    'omni_math': lambda: load_omni_math(),  # All 4,430 olympiad problems
    'omni_math_hard': lambda: load_omni_math(min_difficulty=6.0),  # Hard subset (difficulty >= 6.0)
    'gaia': lambda: load_gaia(),  # All 165 validation items
    'gaia_l1': lambda: load_gaia(level=1),  # Level 1 (53 items, easiest)
    'gaia_l2': lambda: load_gaia(level=2),  # Level 2 (86 items)
    'gaia_l3': lambda: load_gaia(level=3),  # Level 3 (26 items, hardest)
}


def load_all() -> list[BenchmarkItem]:
    """Load all benchmark datasets."""
    items = []
    items.extend(load_longbench_short())
    items.extend(load_babilong('32k'))
    return items


# =============================================================================
# SWE-bench Dataset Loaders
# =============================================================================

def _load_swebench_dataset(dataset_name: str, max_items: int = None) -> list[SWEBenchItem]:
    """Load a SWE-bench-style dataset from Hugging Face.

    Supports SWE-bench Lite, Verified, and Pro, which share core schema fields
    used by this harness (`instance_id`, `repo`, `base_commit`,
    `problem_statement`, `hints_text`, `test_patch`, and `patch`).
    """
    ds = load_dataset(dataset_name, split='test')

    items = []
    for row in ds:
        if max_items and len(items) >= max_items:
            break

        metadata = {}
        for key in (
            'created_at',
            'version',
            'environment_setup_commit',
            'requirements',
            'interface',
            'difficulty',
        ):
            value = row.get(key, '')
            if value not in ('', None):
                metadata[key] = value

        items.append(SWEBenchItem(
            instance_id=row['instance_id'],
            repo=row['repo'],
            base_commit=row['base_commit'],
            problem_statement=row['problem_statement'],
            hints_text=row.get('hints_text', ''),
            test_patch=row['test_patch'],
            patch=row['patch'],
            metadata=metadata,
        ))

    return items


def load_swebench_verified(max_items: int = None) -> list[SWEBenchItem]:
    """Load SWE-bench Verified dataset (500 human-validated instances).

    Args:
        max_items: Optional limit on number of items
    """
    return _load_swebench_dataset('princeton-nlp/SWE-bench_Verified', max_items=max_items)


def load_swebench_lite(max_items: int = None) -> list[SWEBenchItem]:
    """Load SWE-bench Lite dataset (300 instances, subset for faster eval).

    Args:
        max_items: Optional limit on number of items
    """
    return _load_swebench_dataset('princeton-nlp/SWE-bench_Lite', max_items=max_items)


def load_swebench_pro(max_items: int = None) -> list[SWEBenchItem]:
    """Load SWE-bench Pro dataset.

    SWE-bench Pro is a newer, harder coding benchmark with production-like
    tasks and stricter issue statements.

    Args:
        max_items: Optional limit on number of items.
    """
    return _load_swebench_dataset('ScaleAI/SWE-bench_Pro', max_items=max_items)


def load_swebench_verified_mini(n: int = 50) -> list[SWEBenchItem]:
    """Load first N items from SWE-bench Verified for quick testing.

    Args:
        n: Number of items to load (default: 50)
    """
    return load_swebench_verified(max_items=n)


# SWE-bench dataset registry
SWEBENCH_DATASETS = {
    'swebench_verified': load_swebench_verified,
    'swebench_lite': load_swebench_lite,
    'swebench_pro': load_swebench_pro,
    'swebench_mini': lambda: load_swebench_verified_mini(50),
}


# =============================================================================
# ScienceAgentBench Dataset Loaders
# =============================================================================

def load_scienceagentbench(max_items: int = None, domain: str = None) -> list[ScienceAgentBenchItem]:
    """Load ScienceAgentBench tasks for data-driven scientific discovery.

    102 tasks from 44 peer-reviewed publications across 4 disciplines:
    Bioinformatics, Computational Chemistry, Geographic Information Science,
    Psychology/Cognitive Neuroscience.

    Args:
        max_items: Optional limit on number of items
        domain: Optional filter by domain name (case-insensitive partial match)
    """
    ds = load_dataset('osunlp/ScienceAgentBench', split='validation')

    items = []
    for row in ds:
        if max_items and len(items) >= max_items:
            break

        if domain and domain.lower() not in row['domain'].lower():
            continue

        items.append(ScienceAgentBenchItem(
            instance_id=str(row['instance_id']),
            domain=row['domain'],
            subtask_categories=row['subtask_categories'],
            task_inst=row['task_inst'],
            domain_knowledge=row['domain_knowledge'],
            dataset_folder_tree=row['dataset_folder_tree'],
            dataset_preview=row['dataset_preview'],
            output_fname=row['output_fname'],
            gold_program_name=row['gold_program_name'],
            eval_script_name=row['eval_script_name'],
            github_name=row['github_name'],
            metadata={
                'src_file_or_path': row.get('src_file_or_path', ''),
            }
        ))

    return items


SCIENCEAGENTBENCH_DATASETS = {
    'scienceagentbench': load_scienceagentbench,
    'scienceagentbench_chem': lambda: load_scienceagentbench(domain='chemistry'),
    'scienceagentbench_bio': lambda: load_scienceagentbench(domain='bioinformatics'),
    'scienceagentbench_geo': lambda: load_scienceagentbench(domain='geographic'),
    'scienceagentbench_psych': lambda: load_scienceagentbench(domain='psychology'),
}
