"""LLM adapter implementation using LiteLLM for multi-provider baselines.

Architectural boundary:
- Uses `litellm` to call standard LLM APIs (Gemini, OpenAI, Anthropic, etc.).
- Translates QuestionSpec into system/user prompts asking for structured JSON.
- Evaluates cost and captures raw text responses for audit.
"""

import json
import time
from typing import Any

import litellm

from src.adapters.base import BaseAdapter
from src.config import settings
from src.models import (
    ChoiceResponse,
    NoulResponse,
    PerturbedCase,
    QuestionSpec,
    RunResult,
    ScoreResponse,
)

# litellm can be noisy, suppress non-critical logs if needed
litellm.suppress_debug_info = True


class LLMAdapter(BaseAdapter):
    """Adapter for testing standard LLMs as baselines against Jev."""

    def __init__(self, model_name: str = "gemini/gemini-3.8-flash") -> None:
        """Initialize adapter with specific LiteLLM model string."""
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        # e.g., 'gemini/gemini-1.5-flash' -> 'google' (assuming litellm naming)
        if "/" in self._model_name:
            provider = self._model_name.split("/")[0]
            if provider == "gemini":
                return "google"
            return provider
        return "unknown"

    @property
    def model_version(self) -> str:
        # e.g., 'gemini/gemini-1.5-flash' -> 'gemini-1.5-flash'
        if "/" in self._model_name:
            return self._model_name.split("/")[1]
        return self._model_name

    def _build_prompt(self, question: QuestionSpec) -> str:
        """Translate QuestionSpec into an LLM system instructions prompt."""
        base = (
            f"You are an expert classification engine.\n"
            f"Question / Instructions: {question.instructions}\n\n"
        )

        if question.type == "noul":
            base += (
                "You must return ONLY a valid JSON object matching this schema:\n"
                "{\n"
                '  "probability": <float between 0.0 and 1.0 indicating likelihood of True>,\n'
                '  "judgment": <boolean True or False>\n'
                "}\n"
            )
        elif question.type == "choice":
            if isinstance(question.criteria, list):
                options = question.criteria
            elif isinstance(question.criteria, dict):
                options = list(question.criteria.keys())
            else:
                options = []

            base += (
                f"Options: {options}\n\n"
                "You must return ONLY a valid JSON object matching this schema:\n"
                "{\n"
                '  "selected": "<string, MUST be exactly one of the options>",\n'
                '  "probabilities": {"<opt>": <float between 0.0 and 1.0>}, (sum to 1)\n'
                '  "confidence": <float between 0.0 and 1.0>\n'
                "}\n"
            )
        elif question.type == "score":
            base += (
                "You must return ONLY a valid JSON object matching this schema:\n"
                "{\n"
                '  "score": <float representing the final score>,\n'
                '  "distribution": {"<bucket_name>": <float>, ...},\n'
                '  "confidence": <float between 0.0 and 1.0>\n'
                "}\n"
            )
        else:
            raise ValueError(f"Unsupported question type: {question.type}")

        return base

    async def evaluate(
        self,
        case: PerturbedCase,
        question: QuestionSpec,
        run_id: str,
        batch_id: str,
        use_case: str,
    ) -> RunResult:
        """Execute evaluation using litellm and parse structured JSON response."""
        system_prompt = self._build_prompt(question)
        user_message = f"Input to evaluate:\n{case.perturbed_state}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        start_time = time.perf_counter()

        from typing import cast
        from litellm import ModelResponse

        provider_config = settings.get_provider_config(self.provider_name)
        max_tokens = provider_config.max_tokens or 250

        # We request structured JSON from supported models
        try:
            response_raw = await litellm.acompletion(
                model=self._model_name,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
            response = cast(ModelResponse, response_raw)
        except Exception:
            # Fallback for models that don't support response_format=json_object natively
            response_raw = await litellm.acompletion(
                model=self._model_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            response = cast(ModelResponse, response_raw)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        raw_api_response: dict[str, Any]
        if hasattr(response, "model_dump"):
            raw_api_response = response.model_dump()
        else:
            raw_api_response = dict(response)  # type: ignore

        # Extract content
        content = response.choices[0].message.content or "{}"

        # Very basic extraction if the model wrapped it in markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            parsed_data = json.loads(content)
        except json.JSONDecodeError:
            # Failed to parse JSON, create fallback empty structs
            parsed_data = {}

        parsed_response: NoulResponse | ChoiceResponse | ScoreResponse
        if question.type == "noul":
            prob = float(parsed_data.get("probability", 0.0))
            judgment = bool(parsed_data.get("judgment", prob >= 0.5))
            parsed_response = NoulResponse(probability=prob, judgment=judgment)

        elif question.type == "choice":
            selected = str(parsed_data.get("selected", ""))
            probs = dict(parsed_data.get("probabilities", {}))
            conf = float(parsed_data.get("confidence", 1.0))
            parsed_response = ChoiceResponse(
                selected=selected,
                probabilities=probs,
                confidence=conf,
            )

        elif question.type == "score":
            score_val = float(parsed_data.get("score", 0.0))
            dist = dict(parsed_data.get("distribution", {}))
            conf = float(parsed_data.get("confidence", 1.0))
            parsed_response = ScoreResponse(
                score=score_val,
                distribution=dist,
                confidence=conf,
            )
        else:
            raise ValueError(f"Unknown question type: {question.type}")

        # Compute token usage and dollar cost
        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", len(case.perturbed_state.split()) + 50)
        out_tokens = getattr(usage, "completion_tokens", 20)

        pricing = settings.get_pricing(self.provider_name, self.model_version)
        cost_usd = pricing.calculate_cost(input_tokens=in_tokens, output_tokens=out_tokens)

        return RunResult(
            run_id=run_id,
            batch_id=batch_id,
            provider=self.provider_name,
            model_version=self.model_version,
            use_case=use_case,
            question_key=question.key,
            original_state=case.original_state,
            perturbed_state=case.perturbed_state,
            perturbation_category=case.perturbation_category,
            transform_name=case.transform_name,
            expectation=case.expectation,
            raw_api_response=raw_api_response,
            parsed_response=parsed_response,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost_usd,
            http_status=200,
            retry_count=0,
        )
