from forecasters.forecaster import Forecaster
from common.datatypes import (
    ForecastingQuestion_stripped,
    ForecastingQuestion,
    Forecast,
    Prob,
)
from common.llm_utils import answer, answer_sync, Example
from common.utils import make_json_serializable

class CustomLLMForecaster(Forecaster):
    def __init__(self, model: str = "gpt-5.4-mini", preface: str = None, examples: list = None):
        self.model = model
        self.preface = preface or (
            "You are a highly calibrated and informed forecaster. Answer with a single float "
            "between 0.0 and 1.0 representing the probability of the question resolving YES."
        )
        self.examples = examples or []

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        response = answer_sync(
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob,
            model=self.model,
            **kwargs,
        )
        return Forecast(prob=response.prob, metadata={"model": self.model})

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        response = await answer(
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob,
            model=self.model,
            **kwargs,
        )
        return Forecast(prob=response.prob, metadata={"model": self.model})

    def dump_config(self) -> dict:
        return {
            "model": self.model,
            "preface": self.preface,
            "examples": make_json_serializable(self.examples),
        }

    @classmethod
    def load_config(cls, config: dict) -> "CustomLLMForecaster":
        return cls(
            model=config.get("model", "gpt-5.4-mini"),
            preface=config.get("preface"),
            examples=[
                Example(
                    user=ForecastingQuestion_stripped.load_model_json(e["user"]),
                    assistant=e["assistant"],
                )
                for e in config.get("examples", [])
            ],
        )
