"""Runners for baseline and Spell conditions."""

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
import openai

from .datasets import BenchmarkItem, format_prompt_baseline, format_prompt_spell

# Directory for context files (within spell project for tools/read-file access)
CONTEXT_DIR = Path(__file__).parent.parent.parent / 'benchmarking' / 'data' / 'contexts'


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
    metadata: dict = None  # For Spell: llm_calls, depth, etc.


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Parse 'provider:model' into (provider, model)."""
    if ':' not in model_spec:
        raise ValueError(f"Model spec must be 'provider:model', got: {model_spec}")
    provider, model = model_spec.split(':', 1)
    return provider, model


class BaselineRunner:
    """Run baseline condition: single LLM call."""

    def __init__(self, model_spec: str):
        self.provider, self.model = parse_model_spec(model_spec)

        if self.provider == 'anthropic':
            self.client = anthropic.Anthropic()
        elif self.provider == 'openai':
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def run(self, item: BenchmarkItem) -> RunResult:
        """Run a single item through baseline condition."""
        prompt = format_prompt_baseline(item)
        start = time.time()

        try:
            if self.provider == 'anthropic':
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                response = resp.content[0].text
                tokens_in = resp.usage.input_tokens
                tokens_out = resp.usage.output_tokens
                # Rough cost estimate (Sonnet pricing)
                cost = (tokens_in * 3 + tokens_out * 15) / 1_000_000

            elif self.provider == 'openai':
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                response = resp.choices[0].message.content
                tokens_in = resp.usage.prompt_tokens
                tokens_out = resp.usage.completion_tokens
                # Rough cost estimate
                cost = (tokens_in * 2.5 + tokens_out * 10) / 1_000_000

            latency_ms = (time.time() - start) * 1000

            return RunResult(
                item_id=item.id,
                condition='baseline',
                model=f'{self.provider}:{self.model}',
                response=response,
                answer_extracted=None,  # Filled by evaluator
                correct=None,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
            )

        except Exception as e:
            return RunResult(
                item_id=item.id,
                condition='baseline',
                model=f'{self.provider}:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=str(e),
            )


class SpellRunner:
    """Run Spell condition: LLM writes orchestrating program."""

    def __init__(self, model_spec: str, spell_cli_path: str = None, budget: float = 1.0):
        self.provider, self.model = parse_model_spec(model_spec)
        self.budget = budget

        # Find Spell CLI
        if spell_cli_path:
            self.spell_cli = Path(spell_cli_path)
        else:
            # Default: assume we're in benchmarking/ subdir of spell project
            self.spell_cli = Path(__file__).parent.parent.parent / 'bin' / 'spell'

        if not self.spell_cli.exists():
            # Try running via clj
            self.spell_cli = None

    def run(self, item: BenchmarkItem) -> RunResult:
        """Run a single item through Spell condition."""
        start = time.time()

        try:
            # Write context to a file (model must read it programmatically)
            CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
            context_file = CONTEXT_DIR / f'{item.id}.txt'
            context_file.write_text(item.context)

            # Format prompt with path to context file
            prompt = format_prompt_spell(item, str(context_file))

            # Write prompt to temp file (.spl so CLI treats it as prompt content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.spl', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name

            # Run Spell CLI
            if self.spell_cli and self.spell_cli.exists():
                cmd = [
                    str(self.spell_cli),
                    '-m', f'{self.provider}:{self.model}',
                    '-b', str(self.budget),
                    '-v',  # Verbose for metadata
                    prompt_file,
                ]
            else:
                # Run via clj
                project_root = Path(__file__).parent.parent.parent
                cmd = [
                    'clj', '-M', '-m', 'spell.cli',
                    '-m', f'{self.provider}:{self.model}',
                    '-b', str(self.budget),
                    '-v',
                    prompt_file,
                ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 min timeout
                cwd=str(Path(__file__).parent.parent.parent),  # Project root
            )

            latency_ms = (time.time() - start) * 1000

            # Parse output
            stdout = result.stdout
            stderr = result.stderr

            # Extract final result (last non-empty line before usage stats)
            lines = stdout.strip().split('\n')
            response = ''
            metadata = {'stderr': stderr, 'exit_code': result.returncode}

            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('Usage:') and not line.startswith('  '):
                    response = line
                    break

            # Try to parse usage stats from verbose output
            for line in lines:
                if 'tokens' in line.lower() or 'cost' in line.lower():
                    metadata['usage_line'] = line

            # Clean temp file
            Path(prompt_file).unlink(missing_ok=True)

            return RunResult(
                item_id=item.id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                response=response,
                answer_extracted=None,
                correct=None,
                latency_ms=latency_ms,
                tokens_in=None,  # TODO: parse from verbose
                tokens_out=None,
                cost=None,  # TODO: parse from verbose
                error=stderr if result.returncode != 0 else None,
                metadata=metadata,
            )

        except subprocess.TimeoutExpired:
            return RunResult(
                item_id=item.id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error='Timeout (120s)',
            )
        except Exception as e:
            return RunResult(
                item_id=item.id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=str(e),
            )


def run_batch(
    items: list[BenchmarkItem],
    runner: BaselineRunner | SpellRunner,
    max_parallel: int = 5,
    progress_callback=None,
) -> list[RunResult]:
    """Run a batch of items with parallelism.

    Args:
        items: List of benchmark items to run
        runner: Runner instance (BaselineRunner or SpellRunner)
        max_parallel: Maximum concurrent workers (default 5)
        progress_callback: Optional callback(item_id, result) for progress updates
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(items)  # Preserve order

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all items
        future_to_idx = {
            executor.submit(runner.run, item): i
            for i, item in enumerate(items)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                if progress_callback:
                    progress_callback(items[idx].id, result)
            except Exception as e:
                # Create error result
                results[idx] = RunResult(
                    item_id=items[idx].id,
                    condition=runner.__class__.__name__.lower().replace('runner', ''),
                    model='unknown',
                    response='',
                    answer_extracted=None,
                    correct=None,
                    latency_ms=0,
                    tokens_in=None,
                    tokens_out=None,
                    cost=None,
                    error=str(e),
                )
                if progress_callback:
                    progress_callback(items[idx].id, results[idx])

    return results
