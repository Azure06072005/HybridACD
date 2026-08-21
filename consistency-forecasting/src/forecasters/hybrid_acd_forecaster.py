from forecasters.forecaster import Forecaster
from common.datatypes import ForecastingQuestion, Forecast, Prob
from common.llm_utils import query_api_chat_native, query_api_chat
from common.perscache import register_model_for_cache
import numpy as np
import re
from pydantic import BaseModel

class AdversarialOutput(BaseModel):
    title: str
    body: str

register_model_for_cache(AdversarialOutput)


def _extract_prob_from_cot(text: str) -> float | None:
    """
    Extract the final probability from a CoT response using the same logic as Page 1.
    Priority 1: explicit 'FINAL PROBABILITY: X.XXXX' tag
    Priority 2: last token that looks like a probability (0.XXXX or 1.0)
    Returns None if nothing found.
    """
    import re
    # Priority 1: structured tag
    fp_match = re.search(r'FINAL\s*PROBABILITY.*?([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if fp_match:
        val = float(fp_match.group(1))
        return max(0.0, min(1.0, val))
    # Priority 2: last number in (0,1) range — more restrictive than \d+\.\d+
    # Matches: 0.0, 0.03, 0.1234, 1.0 — avoids years like 2026.07
    nums = re.findall(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', text)
    if nums:
        return max(0.0, min(1.0, float(nums[-1])))
    return None


ADVERSARIAL_AGENT_PREFACE = (
    "You are an adversarial question generator. Your job is to rewrite the input forecasting question "
    "to make it syntactically complex and linguistically challenging, while keeping its mathematical "
    "meaning and resolution criteria completely identical."
)

ADVERSARIAL_AGENT_PROMPT = """
Please rewrite the following forecasting question to expose cognitive vulnerabilities and bypass simple syntactic pattern matching.
Apply grammatical noise, entity swapping, nested conditional logic (CONDCOND), or evidence rules (EXPEVIDENCE) while preserving the exact mathematical resolution criteria.
Make it syntactically complex and difficult to understand, but logically equivalent.

Original Question:
Title: {title}
Body: {body}

You must return a JSON object with 'title' and 'body' keys.
Example:
{{"title": "Rewritten title...", "body": "Rewritten body..."}}
"""

class HybridACDForecaster(Forecaster):
    def __init__(
        self,
        model: str,
        adversarial_model: str = None,
        preface: str = None,
        examples: list = None,
        adversarial_enabled: bool = True,
        tcd_enabled: bool = True,
        research_enabled: bool = True,
    ):
        self.model = model
        self.adversarial_model = adversarial_model or model
        self.preface = preface
        self.examples = examples or []
        self.adversarial_enabled = adversarial_enabled
        self.tcd_enabled = tcd_enabled
        self.research_enabled = research_enabled

    def dump_config(self) -> dict:
        return {
            "model": self.model,
            "adversarial_model": self.adversarial_model,
            "preface": self.preface,
            "examples": self.examples,
            "adversarial_enabled": self.adversarial_enabled,
            "tcd_enabled": self.tcd_enabled,
            "research_enabled": self.research_enabled,
        }

    @classmethod
    def load_config(cls, config: dict) -> "HybridACDForecaster":
        return cls(**config)

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        # Default behavior: call without specific bounds constraints (except [0.0, 1.0])
        return self.call_with_tcd_sync(fq, 0.0, 1.0, **kwargs)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        return await self.call_with_tcd_async(fq, 0.0, 1.0, **kwargs)

    def elicit(self, fqs, **kwargs) -> dict[str, Forecast]:
        # If running on a single question, just delegate to default behavior
        if isinstance(fqs, ForecastingQuestion):
            return {"P": self.call_full(fqs, **kwargs)}
        
        # Convert fqs to dict if it is a BaseModel
        from common.utils import shallow_dict
        if not isinstance(fqs, dict):
            fqs = shallow_dict(fqs)
            
        previous_predictions = {}
        results = {}
        keys = list(fqs.keys())
        rule = kwargs.get("rule")
        
        for key in keys:
            fq = fqs[key]
            
            # Step 1: Adversarial perturbation
            if self.adversarial_enabled:
                fq_query = self.adversarial_rewrite_sync(fq, **kwargs)
            else:
                fq_query = fq
                
            # Compute TCD bounds based on previous predictions
            lower_bound, upper_bound = self.get_consistency_bounds(keys, previous_predictions, key, rule=rule)
            
            # Step 2 & 3: CoT and TCD Call
            forecast = self.call_with_tcd_sync(fq_query, lower_bound, upper_bound, **kwargs)
            
            # Record prediction
            previous_predictions[key] = forecast.prob
            results[key] = forecast
            
        return results

    async def elicit_async(self, fqs, **kwargs) -> dict[str, Forecast]:
        # If running on a single question, just delegate to default behavior
        if isinstance(fqs, ForecastingQuestion):
            return {"P": await self.call_async_full(fqs, **kwargs)}
        
        from common.utils import shallow_dict
        if not isinstance(fqs, dict):
            fqs = shallow_dict(fqs)
            
        previous_predictions = {}
        results = {}
        keys = list(fqs.keys())
        rule = kwargs.get("rule")
        
        for key in keys:
            fq = fqs[key]
            
            # Step 1: Adversarial perturbation
            if self.adversarial_enabled:
                fq_query = await self.adversarial_rewrite_async(fq, **kwargs)
            else:
                fq_query = fq
                
            # Compute TCD bounds based on previous predictions
            lower_bound, upper_bound = self.get_consistency_bounds(keys, previous_predictions, key, rule=rule)
            
            # Step 2 & 3: CoT and TCD Call
            forecast = await self.call_with_tcd_async(fq_query, lower_bound, upper_bound, **kwargs)
            
            # Record prediction
            previous_predictions[key] = forecast.prob
            results[key] = forecast
            
        return results

    async def adversarial_rewrite_async(self, fq: ForecastingQuestion, **kwargs) -> ForecastingQuestion:
        try:
            prompt = ADVERSARIAL_AGENT_PROMPT.format(title=fq.title, body=fq.body)
            response = await query_api_chat(

                messages=[
                    {"role": "system", "content": ADVERSARIAL_AGENT_PREFACE},
                    {"role": "user", "content": prompt}
                ],
                model=self.adversarial_model,
                response_model=AdversarialOutput,
                temperature=0.7,
                **kwargs,
            )
            
            fq_copy = fq.model_copy()
            fq_copy.title = response.title
            fq_copy.body = response.body
            return fq_copy
        except Exception as e:
            print(f"Adversarial rewrite failed: {e}. Falling back to original question.")
            return fq

    def adversarial_rewrite_sync(self, fq: ForecastingQuestion, **kwargs) -> ForecastingQuestion:
        try:
            from common.llm_utils import query_api_chat_sync
            prompt = ADVERSARIAL_AGENT_PROMPT.format(title=fq.title, body=fq.body)
            response = query_api_chat_sync(

                messages=[
                    {"role": "system", "content": ADVERSARIAL_AGENT_PREFACE},
                    {"role": "user", "content": prompt}
                ],
                model=self.adversarial_model,
                response_model=AdversarialOutput,
                temperature=0.7,
                **kwargs,
            )
            
            fq_copy = fq.model_copy()
            fq_copy.title = response.title
            fq_copy.body = response.body
            return fq_copy
        except Exception as e:
            print(f"Adversarial rewrite failed: {e}. Falling back to original question.")
            return fq

    async def call_with_tcd_async(
        self, fq: ForecastingQuestion, lower_bound: float, upper_bound: float, **kwargs
    ) -> Forecast:
        from datetime import date
        today_str = date.today().strftime("%Y-%m-%d")

        preface = self.preface or (
            f"You are an informed and well-calibrated forecaster. Today's date is {today_str}. "
            "Please provide a detailed step-by-step reasoning for the question, using all available "
            "evidence—especially any confirmed real-world outcomes. "
            "If the event has already occurred or been resolved, your probability MUST reflect that "
            "(e.g., 0.0 if it definitely did NOT happen, 1.0 if it definitely DID happen). "
            f"CRITICAL CONSTRAINT: Due to logical consistency with prior answers, your probability MUST be strictly within [{lower_bound:.4f}, {upper_bound:.4f}]. "
            "At the END of your response, you MUST output a line in exactly this format: "
            "'FINAL PROBABILITY: X.XXXX' where X.XXXX is a number within that exact range."
        )

        research_summary = ""
        if getattr(self, "research_enabled", False):
            rag_urls = kwargs.get("rag_urls", None)
            rag_files = kwargs.get("rag_files", None)
            # Extract api credentials — pass them explicitly to the research agent
            _api_key = kwargs.get("api_key")
            _base_url = kwargs.get("base_url")
            from common.agent_tools import execute_research_phase_async
            research_summary = await execute_research_phase_async(
                fq.to_str_forecast_mode(), self.model,
                api_key=_api_key, base_url=_base_url,
                rag_urls=rag_urls, rag_files=rag_files
            )
            preface += (
                f"\n\nResearch Summary (gathered by an Agent on {today_str}):\n"
                f"{research_summary}\n\n"
                "Use the research above—especially any confirmed outcomes—as the primary basis for your probability. "
                "If the research confirms an event has already resolved, set your probability accordingly."
            )

        # Local blind query to generate CoT reasoning
        # Strip unknown kwargs before passing to CoT call
        _safe_kwargs = {k: v for k, v in kwargs.items() if k in ("api_key", "base_url", "temperature", "max_tokens")}
        native_response = await query_api_chat_native(
            model=self.model,
            messages=[
                {"role": "system", "content": preface},
                {"role": "user", "content": fq.to_str_forecast_mode()},
            ],
            **_safe_kwargs,
        )

        # --- Extract probability: same 3-tier logic as Page 1 ---
        # Tier 1: FINAL PROBABILITY tag (structured output we requested)
        # Tier 2: last probability-shaped number in CoT text
        # Tier 3: structured parser model call (fallback for complex responses)
        prob_val = _extract_prob_from_cot(native_response)

        if prob_val is None:
            # Tier 3: use the session model (not hardcoded gpt-4o-mini-2024-07-18)
            parsing_model = self.model
            logit_bias_map = self.get_logit_bias_for_bounds(parsing_model, lower_bound, upper_bound)
            parser_prompt = (
                f"Please parse the probability estimate from this forecaster response: '{native_response}'.\n"
                f"IMPORTANT: The mathematically valid probability range is strictly [{lower_bound:.4f}, {upper_bound:.4f}]. "
                f"You MUST return a probability within this range."
            )
            try:
                parsed_response = await query_api_chat(
                    response_model=Prob,
                    model=parsing_model,
                    messages=[
                        {"role": "system", "content": "Extract the probability as a JSON object with key 'prob'."},
                        {"role": "user", "content": parser_prompt}
                    ],
                    logit_bias=logit_bias_map,
                    **_safe_kwargs,
                )
                prob_val = parsed_response.prob
            except Exception as e:
                print(f"Parser model call failed: {e}. Defaulting to midpoint.")
                prob_val = (lower_bound + upper_bound) / 2

        if self.tcd_enabled:
            prob_val = float(np.clip(prob_val, lower_bound, upper_bound))

        return Forecast(
            prob=round(prob_val, 4),
            metadata={
                "chain_of_thought": native_response,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "research_summary": research_summary
            }
        )

    def call_with_tcd_sync(
        self, fq: ForecastingQuestion, lower_bound: float, upper_bound: float, **kwargs
    ) -> Forecast:
        from datetime import date
        today_str = date.today().strftime("%Y-%m-%d")

        preface = self.preface or (
            f"You are an informed and well-calibrated forecaster. Today's date is {today_str}. "
            "Please provide a detailed step-by-step reasoning for the question, using all available "
            "evidence—especially any confirmed real-world outcomes. "
            "If the event has already occurred or been resolved, your probability MUST reflect that "
            "(e.g., 0.0 if it definitely did NOT happen, 1.0 if it definitely DID happen). "
            f"CRITICAL CONSTRAINT: Due to logical consistency with prior answers, your probability MUST be strictly within [{lower_bound:.4f}, {upper_bound:.4f}]. "
            "At the END of your response, you MUST output a line in exactly this format: "
            "'FINAL PROBABILITY: X.XXXX' where X.XXXX is a number within that exact range."
        )

        from common.llm_utils import query_api_chat_sync_native, query_api_chat_sync

        research_summary = ""
        if getattr(self, "research_enabled", False):
            rag_urls = kwargs.get("rag_urls", None)
            rag_files = kwargs.get("rag_files", None)
            # Extract api credentials — pass them explicitly to the research agent
            _api_key = kwargs.get("api_key")
            _base_url = kwargs.get("base_url")
            from common.agent_tools import execute_research_phase_sync
            research_summary = execute_research_phase_sync(
                fq.to_str_forecast_mode(), self.model,
                api_key=_api_key, base_url=_base_url,
                rag_urls=rag_urls, rag_files=rag_files
            )
            preface += (
                f"\n\nResearch Summary (gathered by an Agent on {today_str}):\n"
                f"{research_summary}\n\n"
                "Use the research above—especially any confirmed outcomes—as the primary basis for your probability. "
                "If the research confirms an event has already resolved, set your probability accordingly."
            )

        native_response = query_api_chat_sync_native(
            model=self.model,
            messages=[
                {"role": "system", "content": preface},
                {"role": "user", "content": fq.to_str_forecast_mode()},
            ],
            **{k: v for k, v in kwargs.items() if k in ("api_key", "base_url", "temperature", "max_tokens")},
        )

        # --- Extract probability: same 3-tier logic as Page 1 ---
        prob_val = _extract_prob_from_cot(native_response)

        if prob_val is None:
            parsing_model = self.model  # use session model, not hardcoded gpt-4o-mini-2024-07-18
            logit_bias_map = self.get_logit_bias_for_bounds(parsing_model, lower_bound, upper_bound)
            parser_prompt = (
                f"Please parse the probability estimate from this forecaster response: '{native_response}'.\n"
                f"IMPORTANT: The mathematically valid probability range is strictly [{lower_bound:.4f}, {upper_bound:.4f}]. "
                f"You MUST return a probability within this range."
            )
            try:
                parsed_response = query_api_chat_sync(
                    response_model=Prob,
                    model=parsing_model,
                    messages=[
                        {"role": "system", "content": "Extract the probability as a JSON object with key 'prob'."},
                        {"role": "user", "content": parser_prompt}
                    ],
                    logit_bias=logit_bias_map,
                    **{k: v for k, v in kwargs.items() if k in ("api_key", "base_url", "temperature", "max_tokens")},
                )
                prob_val = parsed_response.prob
            except Exception as e:
                print(f"Parser model call failed: {e}. Defaulting to midpoint.")
                prob_val = (lower_bound + upper_bound) / 2

        if self.tcd_enabled:
            prob_val = float(np.clip(prob_val, lower_bound, upper_bound))

        return Forecast(
            prob=round(prob_val, 4),
            metadata={
                "chain_of_thought": native_response,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "research_summary": research_summary
            }
        )

    def get_logit_bias_for_bounds(self, model_name: str, lower: float, upper: float) -> dict[str, int]:
        try:
            import tiktoken
            if "o1" in model_name or "gpt-4o" in model_name:
                encoding = tiktoken.get_encoding("o200k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return {}
            
        bias = {}
        for i in range(101):
            val = i / 100.0
            if val < lower or val > upper:
                for fmt in [f"{val:.2f}", f"{val:.1f}"]:
                    tokens = encoding.encode(fmt)
                    if len(tokens) == 1:
                        bias[str(tokens[0])] = -100
        return bias

    def get_consistency_bounds(
        self, keys: list[str], previous_predictions: dict[str, float], target_key: str, rule: str = None
    ) -> tuple[float, float]:
        keys_set = set(keys)
        lower, upper = 0.0, 1.0
        
        # 1. NegChecker: {"P", "not_P"}
        if rule == "NegChecker" or (rule is None and keys_set == {"P", "not_P"}):
            if target_key == "P" and "not_P" in previous_predictions:
                val = 1.0 - previous_predictions["not_P"]
                lower, upper = val, val
            elif target_key == "not_P" and "P" in previous_predictions:
                val = 1.0 - previous_predictions["P"]
                lower, upper = val, val
                
        # 2. ParaphraseChecker: {"P", "para_P"}
        elif rule == "ParaphraseChecker" or (rule is None and keys_set == {"P", "para_P"}):
            if target_key == "P" and "para_P" in previous_predictions:
                val = previous_predictions["para_P"]
                lower, upper = val, val
            elif target_key == "para_P" and "P" in previous_predictions:
                val = previous_predictions["P"]
                lower, upper = val, val
                
        # 3. AndChecker: {"P", "Q", "P_and_Q"}
        elif rule == "AndChecker" or (rule is None and keys_set == {"P", "Q", "P_and_Q"}):
            if target_key == "P_and_Q":
                if "P" in previous_predictions and "Q" in previous_predictions:
                    lower = max(0.0, previous_predictions["P"] + previous_predictions["Q"] - 1.0)
                    upper = min(previous_predictions["P"], previous_predictions["Q"])
            elif target_key in {"P", "Q"}:
                other = "Q" if target_key == "P" else "P"
                if other in previous_predictions and "P_and_Q" in previous_predictions:
                    lower = previous_predictions["P_and_Q"]
                    upper = 1.0 - previous_predictions[other] + previous_predictions["P_and_Q"]
                    
        # 4. OrChecker: {"P", "Q", "P_or_Q"}
        elif rule == "OrChecker" or (rule is None and keys_set == {"P", "Q", "P_or_Q"}):
            if target_key == "P_or_Q":
                if "P" in previous_predictions and "Q" in previous_predictions:
                    lower = max(previous_predictions["P"], previous_predictions["Q"])
                    upper = min(1.0, previous_predictions["P"] + previous_predictions["Q"])
            elif target_key in {"P", "Q"}:
                other = "Q" if target_key == "P" else "P"
                if other in previous_predictions and "P_or_Q" in previous_predictions:
                    lower = previous_predictions["P_or_Q"] - previous_predictions[other]
                    upper = previous_predictions["P_or_Q"]
                    
        # 5. CondChecker: {"P", "Q_given_P", "P_and_Q"}
        elif rule == "CondChecker" or (rule is None and keys_set == {"P", "Q_given_P", "P_and_Q"}):
            if target_key == "P_and_Q":
                if "P" in previous_predictions and "Q_given_P" in previous_predictions:
                    val = previous_predictions["P"] * previous_predictions["Q_given_P"]
                    lower, upper = val, val
            elif target_key == "Q_given_P":
                if "P" in previous_predictions and "P_and_Q" in previous_predictions:
                    if previous_predictions["P"] > 1e-6:
                        val = previous_predictions["P_and_Q"] / previous_predictions["P"]
                        lower, upper = val, val
            elif target_key == "P":
                if "Q_given_P" in previous_predictions and "P_and_Q" in previous_predictions:
                    if previous_predictions["Q_given_P"] > 1e-6:
                        val = previous_predictions["P_and_Q"] / previous_predictions["Q_given_P"]
                        lower, upper = val, val
                        
        # 6. CondCondChecker: {"P", "Q_given_P", "R_given_P_and_Q", "P_and_Q_and_R"}
        elif rule == "CondCondChecker" or (rule is None and keys_set == {"P", "Q_given_P", "R_given_P_and_Q", "P_and_Q_and_R"}):
            if target_key == "P_and_Q_and_R":
                if "P" in previous_predictions and "Q_given_P" in previous_predictions and "R_given_P_and_Q" in previous_predictions:
                    val = previous_predictions["P"] * previous_predictions["Q_given_P"] * previous_predictions["R_given_P_and_Q"]
                    lower, upper = val, val
            elif target_key == "R_given_P_and_Q":
                if "P" in previous_predictions and "Q_given_P" in previous_predictions and "P_and_Q_and_R" in previous_predictions:
                    denom = previous_predictions["P"] * previous_predictions["Q_given_P"]
                    if denom > 1e-6:
                        val = previous_predictions["P_and_Q_and_R"] / denom
                        lower, upper = val, val
            elif target_key == "Q_given_P":
                if "P" in previous_predictions and "R_given_P_and_Q" in previous_predictions and "P_and_Q_and_R" in previous_predictions:
                    denom = previous_predictions["P"] * previous_predictions["R_given_P_and_Q"]
                    if denom > 1e-6:
                        val = previous_predictions["P_and_Q_and_R"] / denom
                        lower, upper = val, val
            elif target_key == "P":
                if "Q_given_P" in previous_predictions and "R_given_P_and_Q" in previous_predictions and "P_and_Q_and_R" in previous_predictions:
                    denom = previous_predictions["Q_given_P"] * previous_predictions["R_given_P_and_Q"]
                    if denom > 1e-6:
                        val = previous_predictions["P_and_Q_and_R"] / denom
                        lower, upper = val, val
                        
        # 7. AndOrChecker: {"P", "Q", "P_and_Q", "P_or_Q"}
        elif rule == "AndOrChecker" or (rule is None and keys_set == {"P", "Q", "P_and_Q", "P_or_Q"}):
            if target_key == "P_and_Q":
                if "P" in previous_predictions and "Q" in previous_predictions and "P_or_Q" in previous_predictions:
                    val = previous_predictions["P"] + previous_predictions["Q"] - previous_predictions["P_or_Q"]
                    lower, upper = val, val
                elif "P" in previous_predictions and "Q" in previous_predictions:
                    lower = max(0.0, previous_predictions["P"] + previous_predictions["Q"] - 1.0)
                    upper = min(previous_predictions["P"], previous_predictions["Q"])
            elif target_key == "P_or_Q":
                if "P" in previous_predictions and "Q" in previous_predictions and "P_and_Q" in previous_predictions:
                    val = previous_predictions["P"] + previous_predictions["Q"] - previous_predictions["P_and_Q"]
                    lower, upper = val, val
                elif "P" in previous_predictions and "Q" in previous_predictions:
                    lower = max(previous_predictions["P"], previous_predictions["Q"])
                    upper = min(1.0, previous_predictions["P"] + previous_predictions["Q"])
            elif target_key in {"P", "Q"}:
                other = "Q" if target_key == "P" else "P"
                if other in previous_predictions and "P_and_Q" in previous_predictions and "P_or_Q" in previous_predictions:
                    val = previous_predictions["P_and_Q"] + previous_predictions["P_or_Q"] - previous_predictions[other]
                    lower, upper = val, val
                elif "P_and_Q" in previous_predictions and "P_or_Q" in previous_predictions:
                    lower = previous_predictions["P_and_Q"]
                    upper = previous_predictions["P_or_Q"]
                    
        # 8. ButChecker: {"P", "Q_and_not_P", "P_or_Q"}
        elif rule == "ButChecker" or (rule is None and keys_set == {"P", "Q_and_not_P", "P_or_Q"}):
            if target_key == "P_or_Q":
                if "P" in previous_predictions and "Q_and_not_P" in previous_predictions:
                    val = previous_predictions["P"] + previous_predictions["Q_and_not_P"]
                    lower, upper = val, val
            elif target_key == "Q_and_not_P":
                if "P" in previous_predictions and "P_or_Q" in previous_predictions:
                    val = previous_predictions["P_or_Q"] - previous_predictions["P"]
                    lower, upper = val, val
            elif target_key == "P":
                if "Q_and_not_P" in previous_predictions and "P_or_Q" in previous_predictions:
                    val = previous_predictions["P_or_Q"] - previous_predictions["Q_and_not_P"]
                    lower, upper = val, val
                    
        # 9. ConsequenceChecker: {"P", "cons_P"}
        elif rule == "ConsequenceChecker" or (rule is None and keys_set == {"P", "cons_P"}):
            if target_key == "P" and "cons_P" in previous_predictions:
                lower, upper = 0.0, previous_predictions["cons_P"]
            elif target_key == "cons_P" and "P" in previous_predictions:
                lower, upper = previous_predictions["P"], 1.0
                
        # 10. ExpectedEvidenceChecker: {"P", "Q", "P_given_Q", "P_given_not_Q"}
        elif rule == "ExpectedEvidenceChecker" or (rule is None and keys_set == {"P", "Q", "P_given_Q", "P_given_not_Q"}):
            if target_key == "P":
                if "Q" in previous_predictions and "P_given_Q" in previous_predictions and "P_given_not_Q" in previous_predictions:
                    val = previous_predictions["Q"] * previous_predictions["P_given_Q"] + (1.0 - previous_predictions["Q"]) * previous_predictions["P_given_not_Q"]
                    lower, upper = val, val
            elif target_key == "Q":
                if "P" in previous_predictions and "P_given_Q" in previous_predictions and "P_given_not_Q" in previous_predictions:
                    diff = previous_predictions["P_given_Q"] - previous_predictions["P_given_not_Q"]
                    if abs(diff) > 1e-6:
                        val = (previous_predictions["P"] - previous_predictions["P_given_not_Q"]) / diff
                        lower, upper = val, val
            elif target_key == "P_given_Q":
                if "P" in previous_predictions and "Q" in previous_predictions and "P_given_not_Q" in previous_predictions:
                    if previous_predictions["Q"] > 1e-6:
                        val = (previous_predictions["P"] - (1.0 - previous_predictions["Q"]) * previous_predictions["P_given_not_Q"]) / previous_predictions["Q"]
                        lower, upper = val, val
            elif target_key == "P_given_not_Q":
                if "P" in previous_predictions and "Q" in previous_predictions and "P_given_Q" in previous_predictions:
                    if previous_predictions["Q"] < 1.0 - 1e-6:
                        val = (previous_predictions["P"] - previous_predictions["Q"] * previous_predictions["P_given_Q"]) / (1.0 - previous_predictions["Q"])
                        lower, upper = val, val
                        
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))
        if lower > upper:
            lower, upper = upper, lower
            
        return lower, upper
