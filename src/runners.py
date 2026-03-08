"""Legacy runners for one-shot and Spell conditions.

Prefer `benchmarking/bench.py` + `src/unified_adapters.py` for active benchmark runs.
"""

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import anthropic
import openai

from .datasets import BenchmarkItem, SWEBenchItem, format_prompt_one_shot, format_prompt_spell

# Directory for context files (within spell project for tools/read-file access)
CONTEXT_DIR = Path(__file__).parent.parent.parent / 'benchmarking' / 'data' / 'contexts'

# Directory for detailed traces
TRACES_DIR = Path(__file__).parent.parent / 'traces'


class TraceLogger:
    """Log detailed traces for each benchmark run."""

    def __init__(self, run_name: str = None):
        """Initialize trace logger.

        Args:
            run_name: Optional name for this run (default: timestamp)
        """
        self.run_name = run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trace_dir = TRACES_DIR / self.run_name
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def log_run(self, item_id: str, condition: str, trace_data: dict):
        """Log a single run's trace.

        Args:
            item_id: Benchmark item ID
            condition: 'one_shot', 'spell', or 'claude_code'
            trace_data: Dict with keys like:
                - prompt: The full prompt sent
                - raw_response: Full response (including reasoning)
                - final_answer: Extracted answer
                - tokens_in, tokens_out: Token counts
                - latency_ms: Timing
                - error: Any error message
                - stdout, stderr: For subprocess-based runners
        """
        trace_file = self.trace_dir / f'{item_id}_{condition}.json'
        trace_data['timestamp'] = datetime.now().isoformat()
        trace_data['item_id'] = item_id
        trace_data['condition'] = condition

        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)

    def get_trace_path(self) -> Path:
        """Return the trace directory path."""
        return self.trace_dir


# Global trace logger (set by run_benchmark.py)
_trace_logger: TraceLogger | None = None


def set_trace_logger(logger: TraceLogger):
    """Set the global trace logger."""
    global _trace_logger
    _trace_logger = logger


def get_trace_logger() -> TraceLogger | None:
    """Get the global trace logger."""
    return _trace_logger


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


@dataclass
class SWERunResult:
    """Result of a SWE-bench run."""
    instance_id: str
    condition: str
    model: str
    patch: str  # Git diff patch (empty if no changes)
    latency_ms: float
    tokens_in: int | None
    tokens_out: int | None
    cost: float | None
    error: str | None = None
    metadata: dict = None
    changed_files: list[str] = None


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Parse 'provider:model' into (provider, model)."""
    if ':' not in model_spec:
        raise ValueError(f"Model spec must be 'provider:model', got: {model_spec}")
    provider, model = model_spec.split(':', 1)
    return provider, model


# Per-million-token pricing: {model_prefix: (input_cost, output_cost)}
MODEL_COSTS = {
    'claude-opus-4-5': (15.0, 75.0),
    'claude-opus-4-6': (5.0, 25.0),
    'claude-sonnet-4-5': (3.0, 15.0),
    'claude-sonnet-5': (3.0, 15.0),
    'claude-haiku-4-5': (1.0, 5.0),
    'o3': (2.0, 8.0),
    'o4-mini': (1.1, 4.4),
    'gpt-4o': (2.5, 10.0),
    'gpt-5': (1.25, 10.0),
    'gpt-5.2': (1.75, 14.0),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate cost using prefix-matched model pricing."""
    for prefix, (in_cost, out_cost) in MODEL_COSTS.items():
        if model.startswith(prefix):
            return (tokens_in * in_cost + tokens_out * out_cost) / 1_000_000
    # Fallback: use rough middle-ground pricing
    return (tokens_in * 3.0 + tokens_out * 15.0) / 1_000_000


class OneShotRunner:
    """Run one-shot condition: single LLM call."""

    def __init__(self, model_spec: str, thinking: bool = False, thinking_budget: int = 10000,
                 reasoning_effort: str = "high"):
        self.provider, self.model = parse_model_spec(model_spec)
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort

        if self.provider == 'anthropic-pf':
            self.client = anthropic.Anthropic()
        elif self.provider == 'openai':
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def run(self, item: BenchmarkItem) -> RunResult:
        """Run a single item through one-shot condition."""
        prompt = format_prompt_one_shot(item)
        start = time.time()

        try:
            if self.provider == 'anthropic-pf':
                kwargs = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                }
                if self.thinking:
                    kwargs['thinking'] = {
                        'type': 'enabled',
                        'budget_tokens': self.thinking_budget,
                    }
                    # max_tokens must exceed thinking budget
                    kwargs['max_tokens'] = self.thinking_budget + 4096
                else:
                    kwargs['max_tokens'] = 256
                if self.thinking:
                    # Use streaming for thinking (avoids 10-min timeout)
                    response = ''
                    tokens_in = 0
                    tokens_out = 0
                    with self.client.messages.stream(**kwargs) as stream:
                        resp = stream.get_final_message()
                    response = next(
                        (b.text for b in resp.content if b.type == 'text'), ''
                    )
                    tokens_in = resp.usage.input_tokens
                    tokens_out = resp.usage.output_tokens
                else:
                    resp = self.client.messages.create(**kwargs)
                    response = resp.content[0].text
                    tokens_in = resp.usage.input_tokens
                    tokens_out = resp.usage.output_tokens
                cost = estimate_cost(self.model, tokens_in, tokens_out)

            elif self.provider == 'openai':
                kwargs = {
                    'model': self.model,
                    'max_tokens': 256,
                    'messages': [{'role': 'user', 'content': prompt}],
                }
                if self.reasoning_effort:
                    kwargs['reasoning_effort'] = self.reasoning_effort
                    kwargs.pop('max_tokens')  # reasoning models use max_completion_tokens
                    kwargs['max_completion_tokens'] = 4096
                resp = self.client.chat.completions.create(**kwargs)
                response = resp.choices[0].message.content
                tokens_in = resp.usage.prompt_tokens
                tokens_out = resp.usage.completion_tokens
                cost = estimate_cost(self.model, tokens_in, tokens_out)

            latency_ms = (time.time() - start) * 1000

            run_result = RunResult(
                item_id=item.id,
                condition='one_shot',
                model=f'{self.provider}:{self.model}',
                response=response,
                answer_extracted=None,  # Filled by evaluator
                correct=None,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
            )

            # Log trace if logger is configured
            if _trace_logger:
                _trace_logger.log_run(item.id, 'one_shot', {
                    'prompt': prompt,
                    'response': response,
                    'tokens_in': tokens_in,
                    'tokens_out': tokens_out,
                    'cost': cost,
                    'latency_ms': latency_ms,
                    'model': f'{self.provider}:{self.model}',
                    'question': item.question,
                    'expected_answer': item.answer,
                })

            return run_result

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            error_msg = str(e)

            # Log trace for error
            if _trace_logger:
                _trace_logger.log_run(item.id, 'one_shot', {
                    'prompt': prompt,
                    'error': error_msg,
                    'latency_ms': latency_ms,
                    'model': f'{self.provider}:{self.model}',
                    'question': item.question,
                    'expected_answer': item.answer,
                })

            return RunResult(
                item_id=item.id,
                condition='one_shot',
                model=f'{self.provider}:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=error_msg,
            )


class SpellRunner:
    """Run Spell condition: LLM writes orchestrating program."""

    def __init__(self, model_spec: str, spell_cli_path: str = None, budget: float = 1.0, agent: str = None,
                 max_tokens: int = None, depth: int = None, thinking: int = None,
                 reasoning_effort: str = "high", verbosity: str = None, responses_api: bool = False,
                 suffix_grammar: bool = False):
        self.provider, self.model = parse_model_spec(model_spec)
        self.budget = budget
        self.agent = agent
        self.max_tokens = max_tokens
        self.depth = depth
        self.thinking = thinking  # thinking budget_tokens (e.g. 10000)
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.responses_api = responses_api
        self.suffix_grammar = suffix_grammar

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
            # Write context to a file if present (model reads programmatically)
            context_file = None
            if item.context:
                CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
                context_file = CONTEXT_DIR / f'{item.id}.txt'
                context_file.write_text(item.context)
                context_file = str(context_file)

            # Format prompt (with context file if applicable)
            prompt = format_prompt_spell(item, context_file)

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
                ]
            else:
                # Run via clj
                project_root = Path(__file__).parent.parent.parent
                cmd = [
                    'clj', '-M', '-m', 'spell.cli',
                    '-m', f'{self.provider}:{self.model}',
                    '-b', str(self.budget),
                    '-v',
                ]
            if self.max_tokens:
                cmd.extend(['-M', str(self.max_tokens)])
            if self.depth:
                cmd.extend(['-d', str(self.depth)])
            if self.thinking:
                cmd.extend(['-K', str(self.thinking)])
            if self.reasoning_effort:
                cmd.extend(['-R', self.reasoning_effort])
            if self.verbosity:
                cmd.extend(['--verbosity', self.verbosity])
            if self.responses_api:
                cmd.append('--responses-api')
            if self.suffix_grammar:
                cmd.append('--suffix-grammar')
            if self.agent:
                cmd.extend(['-a', self.agent])
            cmd.append(prompt_file)

            # Use Popen so we can capture partial output on timeout
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent),  # Project root
            )
            try:
                stdout, stderr = proc.communicate(timeout=300)  # 5 min timeout
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()  # capture partial output
                raise  # re-raise so outer except block handles it

            latency_ms = (time.time() - start) * 1000
            result_returncode = proc.returncode

            # Extract final result (last non-empty line before usage stats)
            lines = stdout.strip().split('\n')
            response = ''
            metadata = {'stderr': stderr, 'exit_code': result_returncode}

            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('Usage:') and not line.startswith('  '):
                    response = line
                    break

            # Parse usage stats from verbose output
            # Format: "  Total: 37 in / 29 out (1 calls) $0.0972 [cache: ...]"
            tokens_in = None
            tokens_out = None
            cost = None

            import re
            for line in lines:
                if 'Total:' in line and 'in /' in line:
                    match = re.search(r'Total:\s*([\d,]+)\s*in\s*/\s*([\d,]+)\s*out.*\$([0-9.]+)', line)
                    if match:
                        tokens_in = int(match.group(1).replace(',', ''))
                        tokens_out = int(match.group(2).replace(',', ''))
                        cost = float(match.group(3))
                        metadata['usage_line'] = line.strip()

            # Clean temp file
            Path(prompt_file).unlink(missing_ok=True)

            run_result = RunResult(
                item_id=item.id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                response=response,
                answer_extracted=None,
                correct=None,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                error=stderr if result_returncode != 0 else None,
                metadata=metadata,
            )

            # Log trace if logger is configured
            if _trace_logger:
                _trace_logger.log_run(item.id, 'spell', {
                    'prompt': prompt,
                    'stdout': stdout,
                    'stderr': stderr,
                    'exit_code': result_returncode,
                    'response': response,
                    'tokens_in': tokens_in,
                    'tokens_out': tokens_out,
                    'cost': cost,
                    'latency_ms': latency_ms,
                    'model': f'{self.provider}:{self.model}',
                    'question': item.question,
                    'expected_answer': item.answer,
                })

            return run_result

        except subprocess.TimeoutExpired:
            latency_ms = (time.time() - start) * 1000
            error_msg = 'Timeout (300s)'

            # stdout/stderr already captured by the inner kill()+communicate()
            # They're in the local scope from the Popen try block

            # Log trace for timeout (include partial output)
            if _trace_logger:
                _trace_logger.log_run(item.id, 'spell', {
                    'prompt': prompt if 'prompt' in dir() else None,
                    'stdout': stdout if 'stdout' in dir() else '',
                    'stderr': stderr if 'stderr' in dir() else '',
                    'error': error_msg,
                    'latency_ms': latency_ms,
                    'model': f'{self.provider}:{self.model}',
                    'question': item.question,
                    'expected_answer': item.answer,
                })

            return RunResult(
                item_id=item.id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=error_msg,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            error_msg = str(e)

            # Log trace for error
            if _trace_logger:
                _trace_logger.log_run(item.id, 'spell', {
                    'prompt': prompt if 'prompt' in dir() else None,
                    'error': error_msg,
                    'latency_ms': latency_ms,
                    'model': f'{self.provider}:{self.model}',
                    'question': item.question,
                    'expected_answer': item.answer,
                })

            return RunResult(
                item_id=item.id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=error_msg,
            )


class ClaudeCodeRunner:
    """Run Claude Code condition: agentic with tool use."""

    def __init__(self, model: str = 'sonnet', timeout: int = 300, max_budget: float = 1.0):
        """Initialize Claude Code runner.

        Args:
            model: Model alias ('sonnet', 'opus') or full name
            timeout: Max seconds per run
            max_budget: Max USD per run
        """
        self.model = model
        self.timeout = timeout
        self.max_budget = max_budget

    def run(self, item: BenchmarkItem) -> RunResult:
        """Run a single item through Claude Code."""
        start = time.time()

        try:
            # Write context to a file if present
            context_file = None
            if item.context:
                CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
                context_file = CONTEXT_DIR / f'{item.id}.txt'
                context_file.write_text(item.context)
                context_file = str(context_file)

            # Format prompt
            prompt = self._format_prompt(item, context_file)

            # Build command
            cmd = [
                'claude',
                '--setting-sources', '',
                '-p',  # Print mode
                '--output-format', 'json',
                '--model', self.model,
                '--max-budget-usd', str(self.max_budget),
                '--add-dir', str(CONTEXT_DIR),
                '--dangerously-skip-permissions',  # Non-interactive
                prompt,
            ]

            # Unset CLAUDECODE env var to allow nested sessions
            env = {k: v for k, v in __import__('os').environ.items() if k != 'CLAUDECODE'}

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )

            latency_ms = (time.time() - start) * 1000

            # Parse JSON output
            if result.returncode == 0 and result.stdout.strip():
                try:
                    output = json.loads(result.stdout)
                    response = output.get('result', '')
                    usage = output.get('usage', {})
                    tokens_in = (
                        usage.get('input_tokens', 0) +
                        usage.get('cache_creation_input_tokens', 0) +
                        usage.get('cache_read_input_tokens', 0)
                    )
                    tokens_out = usage.get('output_tokens', 0)
                    cost = output.get('total_cost_usd')
                    num_turns = output.get('num_turns', 1)

                    run_result = RunResult(
                        item_id=item.id,
                        condition='claude_code',
                        model=f'claude_code:{self.model}',
                        response=response,
                        answer_extracted=None,
                        correct=None,
                        latency_ms=latency_ms,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost=cost,
                        metadata={'num_turns': num_turns, 'session_id': output.get('session_id')},
                    )

                    # Log trace if logger is configured
                    if _trace_logger:
                        _trace_logger.log_run(item.id, 'claude_code', {
                            'prompt': prompt,
                            'raw_output': result.stdout,
                            'stderr': result.stderr,
                            'response': response,
                            'tokens_in': tokens_in,
                            'tokens_out': tokens_out,
                            'cost': cost,
                            'latency_ms': latency_ms,
                            'num_turns': num_turns,
                            'session_id': output.get('session_id'),
                            'model': f'claude_code:{self.model}',
                            'question': item.question,
                            'expected_answer': item.answer,
                        })

                    return run_result
                except json.JSONDecodeError as e:
                    error_msg = f'JSON parse error: {e}'

                    # Log trace for parse error
                    if _trace_logger:
                        _trace_logger.log_run(item.id, 'claude_code', {
                            'prompt': prompt,
                            'raw_output': result.stdout,
                            'stderr': result.stderr,
                            'error': error_msg,
                            'latency_ms': latency_ms,
                            'model': f'claude_code:{self.model}',
                            'question': item.question,
                            'expected_answer': item.answer,
                        })

                    return RunResult(
                        item_id=item.id,
                        condition='claude_code',
                        model=f'claude_code:{self.model}',
                        response=result.stdout,
                        answer_extracted=None,
                        correct=None,
                        latency_ms=latency_ms,
                        tokens_in=None,
                        tokens_out=None,
                        cost=None,
                        error=error_msg,
                    )
            else:
                error_msg = result.stderr or f'Exit code {result.returncode}'

                # Log trace for non-zero exit
                if _trace_logger:
                    _trace_logger.log_run(item.id, 'claude_code', {
                        'prompt': prompt,
                        'raw_output': result.stdout,
                        'stderr': result.stderr,
                        'exit_code': result.returncode,
                        'error': error_msg,
                        'latency_ms': latency_ms,
                        'model': f'claude_code:{self.model}',
                        'question': item.question,
                        'expected_answer': item.answer,
                    })

                return RunResult(
                    item_id=item.id,
                    condition='claude_code',
                    model=f'claude_code:{self.model}',
                    response='',
                    answer_extracted=None,
                    correct=None,
                    latency_ms=latency_ms,
                    tokens_in=None,
                    tokens_out=None,
                    cost=None,
                    error=error_msg,
                )

        except subprocess.TimeoutExpired:
            latency_ms = (time.time() - start) * 1000
            error_msg = f'Timeout ({self.timeout}s)'

            # Log trace for timeout
            if _trace_logger:
                _trace_logger.log_run(item.id, 'claude_code', {
                    'prompt': prompt if 'prompt' in dir() else None,
                    'error': error_msg,
                    'latency_ms': latency_ms,
                    'model': f'claude_code:{self.model}',
                    'question': item.question,
                    'expected_answer': item.answer,
                })

            return RunResult(
                item_id=item.id,
                condition='claude_code',
                model=f'claude_code:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=error_msg,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            error_msg = str(e)

            # Log trace for error
            if _trace_logger:
                _trace_logger.log_run(item.id, 'claude_code', {
                    'prompt': prompt if 'prompt' in dir() else None,
                    'error': error_msg,
                    'latency_ms': latency_ms,
                    'model': f'claude_code:{self.model}',
                    'question': item.question,
                    'expected_answer': item.answer,
                })

            return RunResult(
                item_id=item.id,
                condition='claude_code',
                model=f'claude_code:{self.model}',
                response='',
                answer_extracted=None,
                correct=None,
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=error_msg,
            )

    def _format_prompt(self, item: BenchmarkItem, context_file: str = None) -> str:
        """Format item as a prompt for Claude Code.

        For context items: tells it to read the context file.
        For math problems: presents the problem directly.
        """
        parts = []

        if item.context and context_file:
            doc_len = len(item.context)
            parts.extend([
                f'Read the document at {context_file} ({doc_len:,} chars) and answer this question:',
                '',
                item.question.strip(),
            ])
        else:
            # Self-contained problem (GSM8K math, AIME, etc.)
            parts.extend([
                'Solve this math problem:',
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
            parts.append('Reply with ONLY the letter (A, B, C, or D). No explanation.')
        elif item.dataset in ('gsm8k', 'math', 'aime', 'hmmt'):
            parts.append('')
            parts.append('You can use Python (via `uv run python3`) to compute the answer if helpful.')
            parts.append('Reply with ONLY the final numeric answer (e.g., 42). No explanation.')
        else:
            parts.append('')
            parts.append('Reply with ONLY the answer (e.g., "bathroom"). No explanation.')

        return "\n".join(parts)


def run_batch(
    items: list[BenchmarkItem],
    runner: OneShotRunner | SpellRunner | ClaudeCodeRunner,
    max_parallel: int = 5,
    progress_callback=None,
) -> list[RunResult]:
    """Run a batch of items with parallelism.

    Args:
        items: List of benchmark items to run
        runner: Runner instance (OneShotRunner or SpellRunner)
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


# =============================================================================
# SWE-bench Runners
# =============================================================================

class SpellSWERunner:
    """Run Spell on SWE-bench tasks in cloned repo workspaces."""

    def __init__(self, model_spec: str, budget: float = 10.0, timeout: int = 600, depth: int = 8):
        """Initialize Spell SWE-bench runner.

        Args:
            model_spec: Model spec (provider:model)
            budget: Max USD per instance
            timeout: Max seconds per instance
            depth: Max LLM recursion depth
        """
        self.provider, self.model = parse_model_spec(model_spec)
        self.budget = budget
        self.timeout = timeout
        self.depth = depth

        # Find Spell CLI
        project_root = Path(__file__).parent.parent.parent
        self.spell_cli = project_root / 'bin' / 'spell'
        if not self.spell_cli.exists():
            self.spell_cli = None

    def run(self, item: SWEBenchItem, workspace_path: Path) -> SWERunResult:
        """Run Spell on a SWE-bench item in the given workspace.

        Args:
            item: SWE-bench item with problem statement
            workspace_path: Path to cloned repo at base_commit

        Returns:
            SWERunResult with patch extracted from git diff
        """
        start = time.time()

        try:
            # Format the prompt
            prompt = self._format_prompt(item, workspace_path)

            # Write prompt to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.spl', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name

            # Build command — choose I/O-capable agent profile by transport/provider.
            project_root = Path(__file__).parent.parent.parent
            if self.provider in {'codex-tc', 'anthropic-tc'}:
                agent_rel = Path('config/agents/io-tc.agent.edn')
            elif self.provider == 'anthropic-pf':
                agent_rel = Path('config/agents/io-pf.agent.edn')
            else:
                agent_rel = Path('config/agents/io-msg.agent.edn')
            agent_path = project_root / agent_rel
            if self.spell_cli and self.spell_cli.exists():
                cmd = [
                    str(self.spell_cli),
                    '-a', str(agent_path),
                    '-m', f'{self.provider}:{self.model}',
                    '-b', str(self.budget),
                    '-d', str(self.depth),
                    '-M', '16384',
                    '-v',
                    prompt_file,
                ]
            else:
                cmd = [
                    'clj', '-M:run', '-m', 'spell.cli',
                    '-a', str(agent_path),
                    '-m', f'{self.provider}:{self.model}',
                    '-b', str(self.budget),
                    '-d', str(self.depth),
                    '--max-tokens', '16384',
                    '-v',
                    prompt_file,
                ]

            # Run Spell in the workspace directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(workspace_path),
            )

            latency_ms = (time.time() - start) * 1000

            # Clean temp file
            Path(prompt_file).unlink(missing_ok=True)

            # Extract patch from git diff
            diff_result = subprocess.run(
                ['git', 'diff', 'HEAD'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            patch = diff_result.stdout if diff_result.returncode == 0 else ''

            # Get changed files
            files_result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            changed_files = [f.strip() for f in files_result.stdout.strip().split('\n') if f.strip()]

            return SWERunResult(
                instance_id=item.instance_id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                patch=patch,
                latency_ms=latency_ms,
                tokens_in=None,  # TODO: parse from verbose output
                tokens_out=None,
                cost=None,
                error=result.stderr if result.returncode != 0 else None,
                metadata={
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'exit_code': result.returncode,
                },
                changed_files=changed_files,
            )

        except subprocess.TimeoutExpired as e:
            # Capture partial output from the exception
            stdout = e.stdout or ''
            stderr = e.stderr or ''

            # Still try to extract any partial changes made before timeout
            patch = ''
            changed_files = []
            try:
                diff_result = subprocess.run(
                    ['git', 'diff', 'HEAD'],
                    cwd=workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                patch = diff_result.stdout if diff_result.returncode == 0 else ''

                files_result = subprocess.run(
                    ['git', 'diff', '--name-only', 'HEAD'],
                    cwd=workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                changed_files = [f.strip() for f in files_result.stdout.strip().split('\n') if f.strip()]
            except:
                pass

            return SWERunResult(
                instance_id=item.instance_id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                patch=patch,
                latency_ms=(time.time() - start) * 1000,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=f'Timeout ({self.timeout}s)',
                metadata={
                    'stdout': stdout,
                    'stderr': stderr,
                    'partial': True,
                },
                changed_files=changed_files,
            )
        except Exception as e:
            return SWERunResult(
                instance_id=item.instance_id,
                condition='spell',
                model=f'{self.provider}:{self.model}',
                patch='',
                latency_ms=(time.time() - start) * 1000,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=str(e),
            )

    def _format_prompt(self, item: SWEBenchItem, workspace_path: Path) -> str:
        """Format SWE-bench item as a prompt for Spell.

        Guides the model to explore the repo, understand the issue, and fix it.
        """
        parts = [
            f"You are working in the repository at {workspace_path}",
            f"Repository: {item.repo}",
            "",
            "## Issue to Fix",
            "",
            item.problem_statement.strip(),
            "",
        ]

        if item.hints_text:
            parts.extend([
                "## Hints",
                "",
                item.hints_text.strip(),
                "",
            ])

        parts.extend([
            "## Instructions",
            "",
            "Fix this bug with minimal code changes.",
            "",
            "Use patterns/call-now between steps so you can see each result before deciding the next action:",
            "",
            "1. Search: (patterns/call-now (:out (io/sh \"grep -rn 'pattern' src/\")) 'grep-result)",
            "2. Read: (patterns/call-now (io/read-file \"path\" start end) 'code)",
            "3. Fix: (io/replace-lines \"path\" start end \"new content\")",
            "4. Test: (patterns/call-now (:out (io/sh \"python -m pytest tests/relevant_test.py -x -q 2>&1 | tail -30\")) 'test-result)",
            "",
            "Each call-now spawns a child that sees the result and continues. This lets you:",
            "- See grep results before choosing which file to read",
            "- See exact code + line numbers before choosing the edit range",
            "- See test output and fix further if needed",
            "",
            "io/replace-lines takes (path start-line end-line new-content), 1-indexed, inclusive.",
            "To add new code at end of file, use replace-lines on the last line.",
            "To insert between lines N and N+1, replace line N with itself plus new content.",
            "",
            "Do not install packages or modify project dependencies.",
        ])

        return "\n".join(parts)


class ClaudeCodeSWERunner:
    """Run Claude Code on SWE-bench tasks in cloned repo workspaces."""

    def __init__(self, model: str = 'sonnet', budget: float = 10.0, timeout: int = 600):
        """Initialize Claude Code SWE-bench runner.

        Args:
            model: Model alias ('sonnet', 'opus') or full name
            budget: Max USD per instance
            timeout: Max seconds per instance
        """
        self.model = model
        self.budget = budget
        self.timeout = timeout

    def run(self, item: SWEBenchItem, workspace_path: Path) -> SWERunResult:
        """Run Claude Code on a SWE-bench item in the given workspace.

        Args:
            item: SWE-bench item with problem statement
            workspace_path: Path to cloned repo at base_commit

        Returns:
            SWERunResult with patch extracted from git diff
        """
        start = time.time()

        try:
            # Format the prompt
            prompt = self._format_prompt(item)

            # Build command
            cmd = [
                'claude',
                '--setting-sources', '',
                '-p',  # Print mode (non-interactive)
                '--output-format', 'json',
                '--model', self.model,
                '--max-budget-usd', str(self.budget),
                '--dangerously-skip-permissions',
                prompt,
            ]

            # Run Claude Code in the workspace directory
            # Remove CLAUDECODE env var to avoid nested session error
            import os as _os
            env = {k: v for k, v in _os.environ.items() if k != 'CLAUDECODE'}
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(workspace_path),
                env=env,
            )

            latency_ms = (time.time() - start) * 1000

            # Extract patch from git diff
            diff_result = subprocess.run(
                ['git', 'diff', 'HEAD'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            patch = diff_result.stdout if diff_result.returncode == 0 else ''

            # Get changed files
            files_result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            changed_files = [f.strip() for f in files_result.stdout.strip().split('\n') if f.strip()]

            # Parse JSON output for metadata
            tokens_in = None
            tokens_out = None
            cost = None
            num_turns = None

            if result.returncode == 0 and result.stdout.strip():
                try:
                    output = json.loads(result.stdout)
                    usage = output.get('usage', {})
                    tokens_in = (
                        usage.get('input_tokens', 0) +
                        usage.get('cache_creation_input_tokens', 0) +
                        usage.get('cache_read_input_tokens', 0)
                    )
                    tokens_out = usage.get('output_tokens', 0)
                    cost = output.get('total_cost_usd')
                    num_turns = output.get('num_turns', 1)
                except json.JSONDecodeError:
                    pass

            return SWERunResult(
                instance_id=item.instance_id,
                condition='claude_code',
                model=f'claude_code:{self.model}',
                patch=patch,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                error=result.stderr if result.returncode != 0 and not patch else None,
                metadata={
                    'num_turns': num_turns,
                    'exit_code': result.returncode,
                },
                changed_files=changed_files,
            )

        except subprocess.TimeoutExpired:
            return SWERunResult(
                instance_id=item.instance_id,
                condition='claude_code',
                model=f'claude_code:{self.model}',
                patch='',
                latency_ms=(time.time() - start) * 1000,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=f'Timeout ({self.timeout}s)',
            )
        except Exception as e:
            return SWERunResult(
                instance_id=item.instance_id,
                condition='claude_code',
                model=f'claude_code:{self.model}',
                patch='',
                latency_ms=(time.time() - start) * 1000,
                tokens_in=None,
                tokens_out=None,
                cost=None,
                error=str(e),
            )

    def _format_prompt(self, item: SWEBenchItem) -> str:
        """Format SWE-bench item as a prompt for Claude Code."""
        parts = [
            f"Fix the following issue in the {item.repo} repository:",
            "",
            "## Issue",
            "",
            item.problem_statement.strip(),
            "",
        ]

        if item.hints_text:
            parts.extend([
                "## Hints",
                "",
                item.hints_text.strip(),
                "",
            ])

        parts.extend([
            "## Instructions",
            "",
            "1. Explore the codebase to understand the issue",
            "2. Identify the relevant files",
            "3. Make the necessary code changes to fix the issue",
            "4. Do NOT run tests or create commits",
        ])

        return "\n".join(parts)
