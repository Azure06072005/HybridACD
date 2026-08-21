import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid

from common.llm_utils import Example
from common.datatypes import ForecastingQuestion, Forecast
from forecasters.custom_forecaster import CustomLLMForecaster

mock_response = MagicMock(prob=0.45)

@pytest.fixture
def custom_forecaster():
    return CustomLLMForecaster(model="gpt-5.4-mini")

@pytest.fixture
def mock_forecasting_question():
    return ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will a custom test run succeed?",
        body="Resolves YES if this test succeeds.",
        question_type="binary",
        resolution_date=datetime(2025, 1, 1),
        data_source="synthetic",
        url="http://example.com",
    )

@patch("forecasters.custom_forecaster.answer_sync", return_value=mock_response)
def test_custom_forecaster_call(mock_answer_sync, custom_forecaster, mock_forecasting_question):
    forecast = custom_forecaster.call(mock_forecasting_question)
    assert forecast.prob == pytest.approx(0.45)
    assert forecast.metadata == {"model": "gpt-5.4-mini"}
    mock_answer_sync.assert_called_once()

@pytest.mark.asyncio
@patch("forecasters.custom_forecaster.answer", return_value=mock_response)
async def test_custom_forecaster_call_async(mock_answer, custom_forecaster, mock_forecasting_question):
    forecast = await custom_forecaster.call_async(mock_forecasting_question)
    assert forecast.prob == pytest.approx(0.45)
    assert forecast.metadata == {"model": "gpt-5.4-mini"}
    mock_answer.assert_called_once()

def test_custom_forecaster_config(custom_forecaster):
    config = custom_forecaster.dump_config()
    assert config["model"] == "gpt-5.4-mini"
    assert "preface" in config
    assert isinstance(config["examples"], list)
