"""Configuration and pricing registry.

Architectural boundary:
- Centralizes runtime settings (database paths, default timeouts, batch sizes).
- Houses official pricing tables for token and request cost calculation.
- Never reads or hardcodes API keys or sensitive secrets.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ModelPricing:
    """Pricing rates per million tokens and per request."""

    input_cost_per_million: float
    output_cost_per_million: float = 0.0
    request_fee: float = 0.0

    def calculate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        """Calculate total dollar cost for given token counts."""
        token_cost = (
            (input_tokens / 1_000_000.0) * self.input_cost_per_million
            + (output_tokens / 1_000_000.0) * self.output_cost_per_million
        )
        return round(token_cost + self.request_fee, 8)


# Default fallback pricing (empirically sourced from provider official registries)
DEFAULT_PRICING: dict[str, dict[str, ModelPricing]] = {
    "jev": {
        "jev-1": ModelPricing(input_cost_per_million=0.042, output_cost_per_million=0.0),
    },
    "google": {
        "gemini-1.5-flash": ModelPricing(
            input_cost_per_million=0.075, output_cost_per_million=0.30
        ),
        "gemini-2.0-flash": ModelPricing(
            input_cost_per_million=0.10, output_cost_per_million=0.40
        ),
        "gemini-1.5-pro": ModelPricing(
            input_cost_per_million=1.25, output_cost_per_million=5.00
        ),
    },
    "groq": {
        "gemma-2-9b-it": ModelPricing(
            input_cost_per_million=0.20, output_cost_per_million=0.20
        ),
        "llama-3.1-8b-instant": ModelPricing(
            input_cost_per_million=0.05, output_cost_per_million=0.08
        ),
        "llama-3.3-70b-versatile": ModelPricing(
            input_cost_per_million=0.59, output_cost_per_million=0.79
        ),
    },
    "openai": {
        "gpt-4o-mini": ModelPricing(
            input_cost_per_million=0.15, output_cost_per_million=0.60
        ),
        "gpt-4o": ModelPricing(
            input_cost_per_million=2.50, output_cost_per_million=10.00
        ),
    },
    "anthropic": {
        "claude-3-5-haiku": ModelPricing(
            input_cost_per_million=0.80, output_cost_per_million=4.00
        ),
        "claude-3-5-sonnet": ModelPricing(
            input_cost_per_million=3.00, output_cost_per_million=15.00
        ),
    },
}


@dataclass(slots=True)
class Settings:
    """System runtime settings and defaults."""

    default_jev_model: str = "jev-1"
    default_concurrency_limit: int = 5
    default_timeout_seconds: float = 30.0
    default_max_retries: int = 3
    duckdb_path: str = "data/runs.duckdb"
    pricing_catalog: dict[str, dict[str, ModelPricing]] = field(
        default_factory=lambda: DEFAULT_PRICING
    )

    def get_pricing(self, provider: str, model: str) -> ModelPricing:
        """Lookup pricing for a provider and model, falling back to zero-cost if unknown."""
        provider_catalog = self.pricing_catalog.get(provider.lower())
        if provider_catalog:
            pricing = provider_catalog.get(model.lower())
            if pricing:
                return pricing
        return ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0)

    @classmethod
    def load(cls, pricing_file: Path | str | None = None) -> "Settings":
        """Load settings and optional external pricing catalog."""
        catalog = {**DEFAULT_PRICING}
        if pricing_file is None:
            default_path = Path(__file__).resolve().parent.parent / "pricing" / "providers.json"
            if default_path.is_file():
                pricing_file = default_path

        if pricing_file:
            path = Path(pricing_file)
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                for prov, models in raw_data.items():
                    if prov not in catalog:
                        catalog[prov] = {}
                    for mod, rates in models.items():
                        catalog[prov][mod] = ModelPricing(
                            input_cost_per_million=rates.get("input_cost_per_million", 0.0),
                            output_cost_per_million=rates.get("output_cost_per_million", 0.0),
                            request_fee=rates.get("request_fee", 0.0),
                        )
        return cls(pricing_catalog=catalog)


settings = Settings.load()
