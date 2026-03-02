"""Dataset loading and formatting for benchmark tasks."""

import os
import random
from dataclasses import dataclass, field
from typing import Literal
from datasets import load_dataset
from huggingface_hub import snapshot_download


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


def format_prompt_baseline(item: BenchmarkItem) -> str:
    """Format item as a single prompt for baseline condition."""
    parts = [
        "Context:",
        item.context,
        "",
        "Question:",
        item.question,
    ]

    if item.choices:
        parts.append("")
        parts.append("Choices:")
        for i, choice in enumerate(item.choices):
            letter = chr(ord('A') + i)
            parts.append(f"  {letter}. {choice}")
        parts.append("")
        parts.append("Answer with just the letter (A, B, C, or D).")
    else:
        parts.append("")
        parts.append("Answer with just the answer, no explanation.")

    return "\n".join(parts)


def _escape_for_spell(s: str) -> str:
    """Escape a string for embedding in a Spell string literal."""
    return (s
        .replace('\\', '\\\\')
        .replace('"', '\\"')
        .replace('\n', '\\n')
        .replace('\t', '\\t'))


def format_prompt_spell(item: BenchmarkItem, context_file: str = None) -> str:
    """Format item as a prompt for Spell condition.

    For items with context: context_file points to external file (forces programmatic search).
    For math problems: question is self-contained, no context file.
    NOTE: Prompts that start with '(' bypass CLI's wrap-nl and are treated as raw
    Spell code prefixes. All others get wrapped with the completion wrapper.
    """
    parts = []

    if item.context and context_file:
        # External context (BABILong, LongBench, etc.)
        doc_len = len(item.context)
        parts.extend([
            f'Document path: {context_file}',
            f'Document size: {doc_len:,} chars',
            '',
            f'Question: {item.question.strip()}',
        ])
    elif item.dataset == 'gaia':
        # GAIA: general question, optional file attachment
        parts.append(item.question.strip())
        file_path = (item.metadata or {}).get('file_path', '')
        if file_path:
            parts.extend(['', f'Attached file: {file_path}'])
    else:
        # Self-contained problem (GSM8K math, etc.)
        parts.extend([
            'Math problem:',
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
        parts.append('Solve step by step. Return just the final answer (e.g., 42 or 3/4 or 2\\sqrt{3}).')
    elif item.dataset == 'omni_math':
        parts.append('')
        parts.append('Solve step by step. Return just the final answer.')
    elif item.dataset == 'aime':
        parts.append('')
        parts.append('Solve step by step. Return just the final integer answer (AIME answers are always integers 0-999).')
    elif item.dataset == 'hmmt':
        parts.append('')
        parts.append('Solve step by step. Return just the final answer (e.g., 42 or 3/4 or 2\\sqrt{3}).')
    elif item.dataset == 'gaia':
        parts.append('')
        parts.append('Use web search and any tools needed. Return ONLY the final answer — a number, a short string, or a comma-separated list. No explanation.')
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
