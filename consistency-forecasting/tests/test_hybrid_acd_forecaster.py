import pytest
from datetime import datetime
import uuid
from common.datatypes import ForecastingQuestion, Forecast
from forecasters.hybrid_acd_forecaster import HybridACDForecaster

default_small_model = "gpt-5.4-mini"

@pytest.fixture
def mock_questions():
    q_P = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will the US inflation rate decrease in 2026?",
        body="Resolves YES if US inflation rate in 2026 is lower than 2025.",
        question_type="binary",
        resolution_date=datetime(2027, 1, 1),
    )
    q_not_P = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will the US inflation rate not decrease in 2026?",
        body="Resolves YES if US inflation rate in 2026 is equal to or higher than 2025.",
        question_type="binary",
        resolution_date=datetime(2027, 1, 1),
    )
    return {"P": q_P, "not_P": q_not_P}

def test_hybrid_acd_config_dump_load():
    forecaster = HybridACDForecaster(
        model=default_small_model,
        adversarial_enabled=True,
        tcd_enabled=True
    )
    config = forecaster.dump_config()
    assert config["model"] == default_small_model
    assert config["adversarial_enabled"] is True
    assert config["tcd_enabled"] is True
    
    loaded = HybridACDForecaster.load_config(config)
    assert loaded.model == default_small_model
    assert loaded.adversarial_enabled is True
    assert loaded.tcd_enabled is True

def test_bounds_neg_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "not_P"]
    
    # Test bound for P when not_P is given
    lower, upper = forecaster.get_consistency_bounds(keys, {"not_P": 0.4}, "P")
    assert lower == pytest.approx(0.6)
    assert upper == pytest.approx(0.6)
    
    # Test bound for not_P when P is given
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.75}, "not_P")
    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(0.25)

def test_bounds_paraphrase_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "para_P"]
    
    # Test bound for P when para_P is given
    lower, upper = forecaster.get_consistency_bounds(keys, {"para_P": 0.8}, "P")
    assert lower == pytest.approx(0.8)
    assert upper == pytest.approx(0.8)

def test_bounds_and_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "Q", "P_and_Q"]
    
    # Test P_and_Q given P and Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.7, "Q": 0.6}, "P_and_Q")
    assert lower == pytest.approx(0.3)
    assert upper == pytest.approx(0.6)
    
    # Test P given Q and P_and_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"Q": 0.6, "P_and_Q": 0.4}, "P")
    assert lower == pytest.approx(0.4)
    assert upper == pytest.approx(0.8)

def test_bounds_or_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "Q", "P_or_Q"]
    
    # Test P_or_Q given P and Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.4, "Q": 0.5}, "P_or_Q")
    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(0.9)
    
    # Test P given Q and P_or_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"Q": 0.5, "P_or_Q": 0.8}, "P")
    assert lower == pytest.approx(0.3)
    assert upper == pytest.approx(0.8)

def test_bounds_cond_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "Q_given_P", "P_and_Q"]
    
    # Test P_and_Q given P and Q_given_P
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.8, "Q_given_P": 0.5}, "P_and_Q")
    assert lower == pytest.approx(0.4)
    assert upper == pytest.approx(0.4)

    # Test Q_given_P given P and P_and_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.8, "P_and_Q": 0.4}, "Q_given_P")
    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(0.5)

    # Test P given Q_given_P and P_and_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"Q_given_P": 0.5, "P_and_Q": 0.4}, "P")
    assert lower == pytest.approx(0.8)
    assert upper == pytest.approx(0.8)

def test_bounds_cond_cond_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "Q_given_P", "R_given_P_and_Q", "P_and_Q_and_R"]
    
    # Test P_and_Q_and_R given P, Q_given_P, and R_given_P_and_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.8, "Q_given_P": 0.5, "R_given_P_and_Q": 0.25}, "P_and_Q_and_R")
    assert lower == pytest.approx(0.1)
    assert upper == pytest.approx(0.1)

    # Test R_given_P_and_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.8, "Q_given_P": 0.5, "P_and_Q_and_R": 0.1}, "R_given_P_and_Q")
    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(0.25)

def test_bounds_and_or_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "Q", "P_and_Q", "P_or_Q"]

    # Test P_and_Q given P, Q, P_or_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.6, "Q": 0.7, "P_or_Q": 0.9}, "P_and_Q")
    assert lower == pytest.approx(0.4)
    assert upper == pytest.approx(0.4)

    # Test P given Q, P_and_Q, P_or_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"Q": 0.7, "P_and_Q": 0.4, "P_or_Q": 0.9}, "P")
    assert lower == pytest.approx(0.6)
    assert upper == pytest.approx(0.6)

def test_bounds_but_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "Q_and_not_P", "P_or_Q"]

    # Test P_or_Q given P and Q_and_not_P
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.6, "Q_and_not_P": 0.3}, "P_or_Q")
    assert lower == pytest.approx(0.9)
    assert upper == pytest.approx(0.9)

    # Test P given Q_and_not_P and P_or_Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"Q_and_not_P": 0.3, "P_or_Q": 0.9}, "P")
    assert lower == pytest.approx(0.6)
    assert upper == pytest.approx(0.6)

def test_bounds_consequence_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "cons_P"]

    # Test P given cons_P
    lower, upper = forecaster.get_consistency_bounds(keys, {"cons_P": 0.7}, "P")
    assert lower == 0.0
    assert upper == 0.7

    # Test cons_P given P
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.3}, "cons_P")
    assert lower == 0.3
    assert upper == 1.0

def test_bounds_expected_evidence_checker():
    forecaster = HybridACDForecaster(model=default_small_model)
    keys = ["P", "Q", "P_given_Q", "P_given_not_Q"]

    # Test P
    lower, upper = forecaster.get_consistency_bounds(keys, {"Q": 0.6, "P_given_Q": 0.8, "P_given_not_Q": 0.3}, "P")
    assert lower == pytest.approx(0.6)  # 0.6*0.8 + 0.4*0.3 = 0.48 + 0.12 = 0.6
    assert upper == pytest.approx(0.6)

    # Test Q
    lower, upper = forecaster.get_consistency_bounds(keys, {"P": 0.6, "P_given_Q": 0.8, "P_given_not_Q": 0.3}, "Q")
    assert lower == pytest.approx(0.6)  # (0.6 - 0.3) / (0.8 - 0.3) = 0.3 / 0.5 = 0.6
    assert upper == pytest.approx(0.6)

@pytest.mark.asyncio
async def test_hybrid_acd_elicit_async_mocked(mock_questions):
    # Test sequential execution and dynamic bounds enforcement
    forecaster = HybridACDForecaster(
        model=default_small_model,
        adversarial_enabled=False,  # disabled for deterministic mocked call
        tcd_enabled=True
    )
    
    # Mock call_with_tcd_async to simulate normal predictions
    async def mock_call_with_tcd_async(fq, lower, upper, **kwargs):
        # Return probability at upper bound for test stability
        return Forecast(prob=upper, metadata={"chain_of_thought": "Mock reason"})
        
    forecaster.call_with_tcd_async = mock_call_with_tcd_async
    
    results = await forecaster.elicit_async(mock_questions)
    assert "P" in results
    assert "not_P" in results
    
    # With upper bound logic:
    # First: P returns upper limit of [0, 1] which is 1.0.
    # Second: not_P constraint is calculated as 1.0 - F(P) = 1.0 - 1.0 = 0.0.
    # So not_P must be returned as 0.0.
    # Total P + not_P must be exactly 1.0 (logical consistency).
    assert results["P"].prob + results["not_P"].prob == pytest.approx(1.0)
