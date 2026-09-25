"""Jev adapter implementation for TypeSafe AI's System One model.

Architectural boundary:
- Thin wrapper translating QuestionSpec/PerturbedCase into `typesafe-sdk` calls.
- Preserves raw API responses, tracks millisecond latency and token pricing.
- Contains zero business or evaluation logic.
"""

import time
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

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


class JevAdapter(BaseAdapter):
    """Adapter for Jev System One probabilistic decisions."""

    def __init__(self, client: AsyncTypeSafeClient | None = None) -> None:
        """Initialize with an optional AsyncTypeSafeClient instance."""
        self._client = client or AsyncTypeSafeClient()

    @property
    def provider_name(self) -> str:
        return "jev"

    @property
    def model_version(self) -> str:
        return settings.default_jev_model

    def _build_question(self, question: QuestionSpec) -> Noul | Choice | Score:
        """Translate domain QuestionSpec into typesafe-sdk primitive."""
        if question.type == "noul":
            return Noul(instructions=question.instructions)

        if question.type == "choice":
            criteria_raw = question.criteria
            if isinstance(criteria_raw, list):
                criteria = {opt: None for opt in criteria_raw}
            elif isinstance(criteria_raw, dict):
                criteria = criteria_raw
            else:
                criteria = {}
            return Choice(instructions=question.instructions, criteria=criteria)

        if question.type == "score":
            criteria_list = question.criteria if isinstance(question.criteria, list) else []
            return Score(instructions=question.instructions, criteria=criteria_list)

        raise ValueError(f"Unsupported question type: {question.type}")

    async def evaluate(
        self,
        case: PerturbedCase,
        question: QuestionSpec,
        run_id: str,
        batch_id: str,
        use_case: str,
    ) -> RunResult:
        """Execute a single evaluation call against Jev and return an immutable RunResult."""
        sdk_question = self._build_question(question)
        state_payload = {"input": case.perturbed_state}

        start_time = time.perf_counter()
        response = await self._client.system_one(
            state=state_payload,
            questions={question.key: sdk_question},
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract raw response dictionary for audit preservation
        raw_api_response: dict[str, Any]
        if hasattr(response, "model_dump"):
            raw_api_response = response.model_dump()
        elif hasattr(response, "__dict__"):
            raw_api_response = {
                k: v for k, v in response.__dict__.items() if not k.startswith("_")
            }
        else:
            raw_api_response = {"state": state_payload, "key": question.key}

        # Parse into typed domain response
        parsed_response: NoulResponse | ChoiceResponse | ScoreResponse
        if question.type == "noul":
            noul_obj = response.nouls[question.key]
            prob = float(getattr(noul_obj, "noul", getattr(noul_obj, "probability", 0.0)))
            judgment = prob >= 0.5
            parsed_response = NoulResponse(probability=prob, judgment=judgment)

        elif question.type == "choice":
            choice_obj = response.choices[question.key]
            selected = str(getattr(choice_obj, "choice", getattr(choice_obj, "selected", "")))
            probs = dict(getattr(choice_obj, "probabilities", {}))
            conf = float(getattr(choice_obj, "confidence", 1.0))
            parsed_response = ChoiceResponse(
                selected=selected,
                probabilities=probs,
                confidence=conf,
            )

        elif question.type == "score":
            score_obj = response.scores[question.key]
            score_val = float(getattr(score_obj, "score", 0.0))
            dist = dict(getattr(score_obj, "distribution", {}))
            conf = float(getattr(score_obj, "confidence", 1.0))
            parsed_response = ScoreResponse(
                score=score_val,
                distribution=dist,
                confidence=conf,
            )
        else:
            raise ValueError(f"Unknown question type: {question.type}")

        # Compute token usage and dollar cost
        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", len(case.perturbed_state.split()))
        out_tokens = getattr(usage, "output_tokens", 0)
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
