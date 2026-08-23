"""
consistency-forecasting/src/hybrid_acd_config.py

Single global configuration surface for HybridACD. Every tunable that
currently lives as a scattered constructor kwarg or magic number should
read from here instead, so one file answers "what does this run look like."

Design constraints this respects (see AGENTS.md / DECISIONS.md):
- ADR-002: TCD bounds + adversarial rewrite are separate toggles, not fused.
- ADR-003: retrieval cutoff MUST be an absolute API-level date param, never
  prompt text, and Perplexity's recency filter is explicitly disqualified
  because it can't express an absolute cutoff at all.
- F002: dump_config()/load_config() must round-trip these fields exactly —
  do not add a field here without a matching serialization test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Any
import json


# ---------------------------------------------------------------------------
# Model tier — used by F007's small-vs-large comparison table so "small" and
# "large" are a config choice, not a string literal typed differently in
# five different eval scripts.
# ---------------------------------------------------------------------------

class ModelTier(str, Enum):
    SMALL = "small"
    LARGE = "large"
    REASONING = "reasoning"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    tier: ModelTier
    provider: str  # "openai" | "anthropic" | "openrouter" | "together" | ...

    def __post_init__(self):
        if not self.model_id:
            raise ValueError("ModelConfig.model_id must not be empty")


# Known reference points from the existing on-disk baselines (F007 evidence,
# 2026-08-23 survey). Keep this list append-only as new baselines land —
# don't repurpose an entry for a different run, per the irreversible-evidence
# rule in feature_list.json.
KNOWN_BASELINES: dict[str, ModelConfig] = {
    "gpt-4o-mini": ModelConfig("gpt-4o-mini-2024-07-18", ModelTier.SMALL, "openai"),
    "gpt-4o": ModelConfig("gpt-4o-2024-08-06", ModelTier.LARGE, "openai"),
    "claude-3.5-sonnet": ModelConfig("claude-3.5-sonnet", ModelTier.LARGE, "anthropic"),
    "o1-preview": ModelConfig("o1-preview", ModelTier.REASONING, "openai"),
    "llama-3.1-8b": ModelConfig("llama-3.1-8B", ModelTier.SMALL, "together"),
    "llama-3.1-70b": ModelConfig("llama-3.1-70B", ModelTier.LARGE, "together"),
    "llama-3.1-405b": ModelConfig("llama-3.1-405B", ModelTier.LARGE, "together"),
}


# ---------------------------------------------------------------------------
# TCD config — ADR-002. Nothing here changes existing behavior; it just
# names the two known intervention paths so config, not code branching,
# selects between them (relevant once F006's soft-penalty alternative
# exists alongside hard clipping).
# ---------------------------------------------------------------------------

class TCDMode(str, Enum):
    LOGIT_BIAS = "logit_bias"          # primary mechanism per architecture.md
    HARD_CLIP = "hard_clip"            # deterministic fallback, always available
    SOFT_PENALTY = "soft_penalty"      # F006 candidate — NOT YET IMPLEMENTED


@dataclass
class TCDConfig:
    enabled: bool = True
    mode: TCDMode = TCDMode.LOGIT_BIAS
    logit_bias_value: float = -100.0     # per architecture.md's documented value
    probability_grid_step: float = 0.01  # V_num — hard design constraint, not tunable
    fallback_on_logit_bias_unsupported: bool = True  # never hard-fail per conventions.md

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = TCDMode(self.mode)
        if self.mode == TCDMode.SOFT_PENALTY:
            raise NotImplementedError(
                "SOFT_PENALTY is an F006 design candidate, not implemented. "
                "Do not select it until F006 has a passing verification command."
            )
        if not (0 < self.probability_grid_step <= 1):
            raise ValueError("probability_grid_step must be in (0, 1]")


# ---------------------------------------------------------------------------
# Adversarial rewrite config — ADR-002's second mechanism.
# ---------------------------------------------------------------------------

@dataclass
class AdversarialConfig:
    enabled: bool = True
    adversarial_model: Optional[str] = None  # None => reuse primary model
    max_rewrites_per_question: int = 3
    preserve_resolution_criteria: bool = True  # non-negotiable per ADR-002


# ---------------------------------------------------------------------------
# Retrieval config — ADR-003 / F007. This is the part with a hard
# correctness requirement: the cutoff MUST be an absolute API parameter,
# never prompt text. The disqualified-provider list exists specifically
# because ResolverBasedForecaster's leak was caused by not having this list.
# ---------------------------------------------------------------------------

DISQUALIFIED_SEARCH_PROVIDERS = {
    "perplexity": (
        "search_recency_filter is relative-only (month/week/day/hour); "
        "cannot express an absolute historical cutoff. Confirmed via audit "
        "of ResolverBasedForecaster, 2026-08-23 — see DECISIONS.md ADR-003."
    ),
}

# Providers verified to expose an absolute end-date parameter, and the exact
# param name to use. FILL THIS IN ONLY AFTER CHECKING THE PROVIDER'S CURRENT
# DOCS DIRECTLY — do not assume the param name below is still correct;
# it is a placeholder pending verification (see docs/retrieval_step_spec.md).
VERIFIED_SEARCH_PROVIDERS: dict[str, str] = {
    # "tavily": "end_date",
    # "exa": "end_published_date",
    # "bing_news": "freshness",  # <- CONFIRM: bing's 'freshness' may be
    #                                relative too; verify before uncommenting.
}


@dataclass
class RetrievalConfig:
    enabled: bool = False  # off by default until a provider is verified
    provider: Optional[str] = None
    api_key_env_var: Optional[str] = None
    cutoff_date_field: str = "created_date"  # NEVER "resolution_date"
    max_results: int = 5
    calls_per_base_question: int = 1  # NEVER per adversarial-rewrite variant

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.provider is None:
            raise ValueError("RetrievalConfig.enabled=True requires a provider")
        if self.provider in DISQUALIFIED_SEARCH_PROVIDERS:
            raise ValueError(
                f"Provider '{self.provider}' is disqualified: "
                f"{DISQUALIFIED_SEARCH_PROVIDERS[self.provider]}"
            )
        if self.provider not in VERIFIED_SEARCH_PROVIDERS:
            raise ValueError(
                f"Provider '{self.provider}' has no verified absolute "
                f"end-date parameter recorded in VERIFIED_SEARCH_PROVIDERS. "
                f"Verify against that provider's current API docs and add "
                f"it there before enabling — do not bypass this check."
            )
        if self.cutoff_date_field == "resolution_date":
            raise ValueError(
                "cutoff_date_field must never be 'resolution_date' — this "
                "is exactly the outcome-leakage bug found in "
                "ResolverBasedForecaster (DECISIONS.md ADR-003)."
            )
        if self.calls_per_base_question != 1:
            raise ValueError(
                "calls_per_base_question must be 1 — retrieval fires once "
                "per base question, before adversarial rewriting, per "
                "docs/retrieval_step_spec.md call order."
            )


# ---------------------------------------------------------------------------
# Top-level run config — what a single evaluation invocation is.
# This is the object dump_config()/load_config() (F002) should serialize.
# ---------------------------------------------------------------------------

@dataclass
class HybridACDRunConfig:
    model: str
    tcd: TCDConfig = field(default_factory=TCDConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    research_enabled: bool = False  # existing F002 field — preserved as-is

    def validate(self) -> None:
        self.retrieval.validate()

    def dump_config(self) -> dict[str, Any]:
        """Returns standard dictionary config conforming to Forecaster.dump_config()."""
        self.validate()
        d = asdict(self)
        # convert Enum to str
        if "tcd" in d and "mode" in d["tcd"] and isinstance(d["tcd"]["mode"], TCDMode):
            d["tcd"]["mode"] = d["tcd"]["mode"].value
        return d

    def to_json(self) -> str:
        """JSON serialization helper."""
        return json.dumps(self.dump_config(), default=str, indent=2)

    @classmethod
    def load_config(cls, raw: dict[str, Any] | str) -> "HybridACDRunConfig":
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw
            
        tcd_raw = data.get("tcd", {})
        if "mode" in tcd_raw and isinstance(tcd_raw["mode"], str):
            tcd_raw["mode"] = TCDMode(tcd_raw["mode"])
        tcd = TCDConfig(**tcd_raw)
        
        adversarial = AdversarialConfig(**data.get("adversarial", {}))
        retrieval = RetrievalConfig(**data.get("retrieval", {}))
        cfg = cls(
            model=data["model"],
            tcd=tcd,
            adversarial=adversarial,
            retrieval=retrieval,
            research_enabled=data.get("research_enabled", False),
        )
        cfg.validate()
        return cfg
