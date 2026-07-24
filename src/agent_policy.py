"""Spell benchmark agent selection policy.

Selects a non-benchmark-specific agent profile from model/provider transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SpellMode = Literal["msg", "tc"]
BenchKind = Literal["general", "swebench", "featurebench", "exercism"]
GeneralFamily = Literal["general_longcontext", "general_other"]
AgentSource = Literal["default", "agent", "agent_msg", "agent_tc"]
ProviderFamily = Literal["pf", "msg", "tc"]
PREFILL_PROVIDER_PREFIXES = {"anthropic-pf", "fireworks"}

DEFAULT_AGENT_POLICY: dict[tuple[ProviderFamily, SpellMode], str] = {
    ("pf", "msg"): "config/agent-profiles/io-pf.agent.edn",
    ("msg", "msg"): "config/agent-profiles/io-msg.agent.edn",
    ("tc", "tc"): "config/agent-profiles/io-tc.agent.edn",
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
        return "msg"
    prefix, _ = spec.split(":", 1)
    return "tc" if prefix in {"codex-tc", "anthropic-tc", "openai-tc", "fireworks-tc"} else "msg"


def _provider_prefix(model_spec: str | None) -> str | None:
    spec = (model_spec or "").strip()
    if ":" not in spec:
        return None
    prefix, _ = spec.split(":", 1)
    return prefix or None


def classify_provider_family(model_spec: str | None, spell_mode: SpellMode) -> ProviderFamily:
    if spell_mode == "tc":
        return "tc"
    return "pf" if _provider_prefix(model_spec) in PREFILL_PROVIDER_PREFIXES else "msg"


def _is_longcontext_dataset(dataset_name: str) -> bool:
    return dataset_name == "longbench" or dataset_name.startswith("longbench_") or dataset_name.startswith("babilong_")


def classify_general_family(item_datasets: set[str]) -> GeneralFamily:
    """Classify general benchmark runs by loaded item datasets."""
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
    if bench_kind == "exercism":
        return "exercism"
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
    agent_msg: str | None = None,
    agent_tc: str | None = None,
) -> ResolvedSpellAgent:
    """Resolve Spell agent path with mode-specific overrides and defaults.

    Precedence:
    1) mode-specific override (--agent-msg/--agent-tc)
    2) shared override (--agent)
    3) default provider/transport policy
    """
    spell_mode = detect_spell_mode(model_spec)
    family = _resolve_family(bench_kind, item_datasets)
    provider_family = classify_provider_family(model_spec, spell_mode)

    if spell_mode == "msg" and agent_msg:
        selected = agent_msg
        source: AgentSource = "agent_msg"
    elif spell_mode == "tc" and agent_tc:
        selected = agent_tc
        source = "agent_tc"
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
