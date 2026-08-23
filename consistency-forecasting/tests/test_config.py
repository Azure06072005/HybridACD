import pytest
from hybrid_acd_config import (
    ModelTier,
    ModelConfig,
    KNOWN_BASELINES,
    TCDMode,
    TCDConfig,
    AdversarialConfig,
    RetrievalConfig,
    HybridACDRunConfig,
    VERIFIED_SEARCH_PROVIDERS,
)


def test_model_config():
    assert "gpt-4o-mini" in KNOWN_BASELINES
    assert KNOWN_BASELINES["gpt-4o-mini"].tier == ModelTier.SMALL
    assert KNOWN_BASELINES["gpt-4o"].tier == ModelTier.LARGE
    assert KNOWN_BASELINES["o1-preview"].tier == ModelTier.REASONING

    with pytest.raises(ValueError, match="must not be empty"):
        ModelConfig(model_id="", tier=ModelTier.SMALL, provider="openai")


def test_tcd_config():
    tcd = TCDConfig(enabled=True, mode=TCDMode.LOGIT_BIAS, logit_bias_value=-100.0, probability_grid_step=0.01)
    assert tcd.enabled is True
    assert tcd.logit_bias_value == -100.0
    assert tcd.probability_grid_step == 0.01

    # F006 candidate error
    with pytest.raises(NotImplementedError, match="SOFT_PENALTY is an F006 design candidate"):
        TCDConfig(mode=TCDMode.SOFT_PENALTY)

    # Invalid probability grid step
    with pytest.raises(ValueError, match="probability_grid_step must be in"):
        TCDConfig(probability_grid_step=0.0)


def test_retrieval_config_validation():
    # 1. Disabled retrieval should pass validation without provider
    retrieval_off = RetrievalConfig(enabled=False)
    retrieval_off.validate()

    # 2. Enabled retrieval requires a provider
    with pytest.raises(ValueError, match="requires a provider"):
        RetrievalConfig(enabled=True, provider=None).validate()

    # 3. Disqualified provider (Perplexity) rejection
    with pytest.raises(ValueError, match="Provider 'perplexity' is disqualified"):
        RetrievalConfig(enabled=True, provider="perplexity").validate()

    # 4. Unverified provider rejection
    with pytest.raises(ValueError, match="has no verified absolute end-date parameter"):
        RetrievalConfig(enabled=True, provider="unknown_engine").validate()

    # 5. resolution_date rejection (outcome leakage guardrail)
    with pytest.raises(ValueError, match="cutoff_date_field must never be 'resolution_date'"):
        # Temporarily mock a verified provider to test date field validation
        VERIFIED_SEARCH_PROVIDERS["mock_engine"] = "end_date"
        try:
            RetrievalConfig(enabled=True, provider="mock_engine", cutoff_date_field="resolution_date").validate()
        finally:
            del VERIFIED_SEARCH_PROVIDERS["mock_engine"]

    # 6. Single search per base question enforcement
    with pytest.raises(ValueError, match="calls_per_base_question must be 1"):
        VERIFIED_SEARCH_PROVIDERS["mock_engine"] = "end_date"
        try:
            RetrievalConfig(enabled=True, provider="mock_engine", calls_per_base_question=3).validate()
        finally:
            del VERIFIED_SEARCH_PROVIDERS["mock_engine"]


def test_hybrid_acd_run_config_roundtrip():
    cfg = HybridACDRunConfig(
        model="gpt-4o-mini-2024-07-18",
        tcd=TCDConfig(enabled=True, mode=TCDMode.HARD_CLIP),
        adversarial=AdversarialConfig(enabled=True, adversarial_model="gpt-4o-mini-2024-07-18"),
        retrieval=RetrievalConfig(enabled=False),
        research_enabled=False,
    )

    # Dump to dict
    d = cfg.dump_config()
    assert d["model"] == "gpt-4o-mini-2024-07-18"
    assert d["tcd"]["mode"] == "hard_clip"
    assert d["adversarial"]["enabled"] is True

    # Load from dict
    loaded_from_dict = HybridACDRunConfig.load_config(d)
    assert loaded_from_dict.model == cfg.model
    assert loaded_from_dict.tcd.mode == TCDMode.HARD_CLIP

    # Dump to json and load from json
    json_str = cfg.to_json()
    loaded_from_json = HybridACDRunConfig.load_config(json_str)
    assert loaded_from_json.model == cfg.model
    assert loaded_from_json.tcd.mode == TCDMode.HARD_CLIP
    assert loaded_from_json.adversarial.adversarial_model == "gpt-4o-mini-2024-07-18"
