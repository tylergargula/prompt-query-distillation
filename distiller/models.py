"""Pydantic models enforcing structured output from the distillation LLM call."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

B2B_INDUSTRIES = (
    "saas_software",
    "professional_services",
    "manufacturing_industrial",
    "marketing_advertising",
    "logistics_supply_chain",
    "cybersecurity",
    "hr_staffing",
    "financial_technology",
    "insurance_b2b",
    "commercial_real_estate",
)

B2C_INDUSTRIES = (
    "home_services",
    "health_wellness",
    "fitness_recreation",
    "ecommerce_retail",
    "travel_hospitality",
    "personal_finance",
    "real_estate_residential",
    "automotive",
    "education_learning",
    "beauty_personal_care",
    "food_beverage_delivery",
)

INDUSTRIES_BY_INTENT = {
    "B2B": B2B_INDUSTRIES,
    "B2C": B2C_INDUSTRIES,
}


class DistillationResult(BaseModel):
    """The LLM's structured judgment of a prompt's intent, industry, and keyword universe."""

    intent: Literal["B2B", "B2C"]
    industry: str
    keywords: list[str] = Field(min_length=3, max_length=6)

    @field_validator("industry")
    @classmethod
    def _normalize_industry(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("keywords")
    @classmethod
    def _clean_keywords(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for kw in v:
            norm = " ".join(kw.strip().lower().split())
            if norm and norm not in seen:
                seen.add(norm)
                cleaned.append(norm)
        if len(cleaned) < 3:
            raise ValueError(
                f"expected at least 3 distinct non-empty keywords after cleaning, got {cleaned}"
            )
        return cleaned

    @model_validator(mode="after")
    def _industry_matches_intent(self) -> Self:
        allowed = INDUSTRIES_BY_INTENT[self.intent]
        if self.industry not in allowed:
            raise ValueError(
                f"industry '{self.industry}' is not a valid {self.intent} industry. "
                f"Must be one of: {', '.join(allowed)}"
            )
        return self
