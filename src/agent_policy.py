"""Spell benchmark agent selection policy.

Selects a non-benchmark-specific agent profile from model/provider transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SpellMode = Literal["message", "toolcall"]
BenchKind = Literal["general", "swebench", "featurebench"]
GeneralFamily = Literal["general_math", "general_longcontext", "general_other"]
AgentSource = Literal["default", "agent", "agent_message", "agent_toolcall"]
ProviderFamily = Literal["prefill", "message", "toolcall"]

MATH_DATASETS = {"gsm8k", "math", "aime", "hmmt", "omni_math"}


DEFAULT_AGENT_POLICY: dict[tuple[ProviderFamily, SpellMode], str] = {
    ("prefill", "message"): "config/agents/io-prefill.agent.edn",
    ("message", "message"): "config/agents/io-message.agent.edn",
    ("toolcall", "toolcall"): "config/agents/io-toolcall.agent.edn",
}


@dataclass(frozen=True)
class ResolvedSpellAgent:
    agent_file: str
    spell_mode: SpellMode
    source: AgentSource
    policy_key: str
    family: str


def detect_spell_mode(model_spec: str | None) -> SpellMode:
    """Infer Spell mode from model spec prefix."""
    spec = (model_spec or "").strip()
    if ":" not in spec:
        return "message"
    prefix, _ = spec.split(":", 1)
    return "toolcall" if prefix == "codex-toolcall" else "message"


def _provider_prefix(model_spec: str | None) -> str | None:
    spec = (model_spec or "").strip()
    if ":" not in spec:
        return None
    prefix, _ = spec.split(":", 1)
    return prefix or None


def classify_provider_family(model_spec: str | None, spell_mode: SpellMode) -> ProviderFamily:
    if spell_mode == "toolcall":
        return "toolcall"
    return "prefill" if _provider_prefix(model_spec) == "anthropic" else "message"


def _is_longcontext_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith("longbench_") or dataset_name.startswith("babilong_")


def classify_general_family(item_datasets: set[str]) -> GeneralFamily:
    """Classify general benchmark runs by loaded item datasets."""
    if item_datasets and item_datasets.issubset(MATH_DATASETS):
        return "general_math"
    if item_datasets and all(_is_longcontext_dataset(d) for d in item_datasets):
        return "general_longcontext"
    return "general_other"


def _resolve_family(bench_kind: BenchKind, item_datasets: set[str] | None) -> str:
    if bench_kind == "general":
        return classify_general_family(item_datasets or set())
    if bench_kind == "swebench":
        return "swebench"
    if bench_kind == "featurebench":
        return "featurebench"
    raise ValueError(f"Unknown benchmark kind: {bench_kind}")


def _absolute_agent_path(project_root: Path, agent_path: str) -> str:
    path = Path(agent_path)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def resolve_spell_agent(
    *,
    project_root: Path,
    bench_kind: BenchKind,
    model_spec: str | None,
    item_datasets: set[str] | None = None,
    agent: str | None = None,
    agent_message: str | None = None,
    agent_toolcall: str | None = None,
) -> ResolvedSpellAgent:
    """Resolve Spell agent path with mode-specific overrides and defaults.

    Precedence:
    1) mode-specific override (--agent-message/--agent-toolcall)
    2) shared override (--agent)
    3) default provider/transport policy
    """
    spell_mode = detect_spell_mode(model_spec)
    family = _resolve_family(bench_kind, item_datasets)
    provider_family = classify_provider_family(model_spec, spell_mode)

    if spell_mode == "message" and agent_message:
        selected = agent_message
        source: AgentSource = "agent_message"
    elif spell_mode == "toolcall" and agent_toolcall:
        selected = agent_toolcall
        source = "agent_toolcall"
    elif agent:
        selected = agent
        source = "agent"
    else:
        key = (provider_family, spell_mode)
        try:
            selected = DEFAULT_AGENT_POLICY[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"No default Spell agent policy for key: {key}") from exc
        source = "default"

    abs_path = _absolute_agent_path(project_root, selected)
    if not Path(abs_path).exists():
        raise FileNotFoundError(
            f"Resolved Spell agent file not found ({source}, mode={spell_mode}, family={family}): {abs_path}"
        )

    return ResolvedSpellAgent(
        agent_file=abs_path,
        spell_mode=spell_mode,
        source=source,
        policy_key=f"{provider_family}:{spell_mode}",
        family=family,
    )
