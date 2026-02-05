"""Dataset loading and formatting for benchmark tasks."""

from dataclasses import dataclass
from typing import Literal
from datasets import load_dataset


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


def format_prompt_spell(item: BenchmarkItem, context_file: str) -> str:
    """Format item as a prompt for Spell condition.

    Context is NOT in the prompt - model must load it from context_file.
    This forces programmatic search rather than visual scanning.

    NOTE: Prompt must NOT start with '(' so CLI wraps it with quine preamble
    (needed for patterns/call-now which uses completion binding).
    """
    doc_len = len(item.context)
    parts = [
        f'Document path: {context_file}',
        f'Document size: {doc_len:,} chars',
        '',
        f'Question: {item.question.strip()}',
    ]

    if item.choices:
        parts.append('Choices:')
        for i, choice in enumerate(item.choices):
            letter = chr(ord('A') + i)
            parts.append(f'  {letter}. {choice}')
        parts.append('')
        parts.append('Return just the letter (A, B, C, or D).')
    else:
        parts.append('')
        parts.append('Return just the answer (e.g., "bathroom").')

    return "\n".join(parts)


# Dataset registry
DATASETS = {
    'longbench_short': load_longbench_short,
    'babilong_32k': lambda: load_babilong('32k'),
    'babilong_32k_hard': lambda: load_babilong('32k', tasks=['qa2', 'qa3']),  # Multi-hop
    'babilong_16k': lambda: load_babilong('16k'),
    'babilong_8k': lambda: load_babilong('8k'),
}


def load_all() -> list[BenchmarkItem]:
    """Load all benchmark datasets."""
    items = []
    items.extend(load_longbench_short())
    items.extend(load_babilong('32k'))
    return items
