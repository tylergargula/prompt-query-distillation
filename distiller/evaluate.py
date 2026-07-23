"""Compares candidate LLMs on keyword-distillation quality.

"Quality" here means: of the keywords a model distills from a prompt, how many
does Ahrefs actually have monthly search volume (MSV) data for? The model with
the highest MSV hit rate (tie-broken by total MSV captured) wins.

Uses reference-files/reference_distillation.json as the prompt sample, skipping
the example each industry contributes as a few-shot (see few_shots.py) so the
eval isn't just measuring memorization of the exact worked example.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from distiller.ahrefs_client import AhrefsClient
from distiller.few_shots import REFERENCE_FILE
from distiller.llm_client import CANDIDATE_MODELS, DistillationError, distill_prompt

FEW_SHOT_COUNT = 1  # must match few_shots.load_few_shots(per_industry=...) used at inference time


@dataclass
class PromptEvalResult:
    prompt: str
    intent_expected: str
    industry_expected: str
    model: str
    ok: bool
    error: str | None = None
    intent_predicted: str | None = None
    industry_predicted: str | None = None
    keywords: list[str] = field(default_factory=list)
    keyword_volumes: dict[str, int] = field(default_factory=dict)


@dataclass
class ModelEvalResult:
    model: str
    prompts_tested: int = 0
    llm_failures: int = 0
    intent_correct: int = 0
    industry_correct: int = 0
    keywords_generated: int = 0
    keywords_with_msv: int = 0
    total_msv: int = 0

    @property
    def hit_rate(self) -> float:
        return self.keywords_with_msv / self.keywords_generated if self.keywords_generated else 0.0

    @property
    def intent_accuracy(self) -> float:
        return self.intent_correct / self.prompts_tested if self.prompts_tested else 0.0

    @property
    def industry_accuracy(self) -> float:
        return self.industry_correct / self.prompts_tested if self.prompts_tested else 0.0


def load_eval_prompts(per_industry: int = 2, skip: int = FEW_SHOT_COUNT) -> list[dict]:
    """Sample prompts per industry, skipping the ones used as few-shot examples."""
    data = json.loads(Path(REFERENCE_FILE).read_text())
    prompts = []
    for intent in ("B2B", "B2C"):
        for industry, entries in data[intent].items():
            for entry in entries[skip : skip + per_industry]:
                prompts.append({"intent": intent, "industry": industry, "prompt": entry["prompt"]})
    return prompts


def evaluate_model(
    model: str,
    prompts: list[dict],
    country: str = "us",
    ahrefs_client: AhrefsClient | None = None,
) -> tuple[ModelEvalResult, list[PromptEvalResult]]:
    ahrefs = ahrefs_client or AhrefsClient()
    prompt_results: list[PromptEvalResult] = []

    for item in prompts:
        try:
            distilled = distill_prompt(item["prompt"], model=model)
            prompt_results.append(
                PromptEvalResult(
                    prompt=item["prompt"],
                    intent_expected=item["intent"],
                    industry_expected=item["industry"],
                    model=model,
                    ok=True,
                    intent_predicted=distilled.intent,
                    industry_predicted=distilled.industry,
                    keywords=distilled.keywords,
                )
            )
        except DistillationError as exc:
            prompt_results.append(
                PromptEvalResult(
                    prompt=item["prompt"],
                    intent_expected=item["intent"],
                    industry_expected=item["industry"],
                    model=model,
                    ok=False,
                    error=str(exc),
                )
            )

    # Batch every distilled keyword across all prompts into one (auto-chunked) MSV lookup.
    all_keywords = sorted({kw for r in prompt_results if r.ok for kw in r.keywords})
    volume_rows = ahrefs.keywords_overview(all_keywords, country=country) if all_keywords else []
    volume_by_keyword = {
        " ".join(row["keyword"].strip().lower().split()): row.get("volume", 0) or 0
        for row in volume_rows
    }

    summary = ModelEvalResult(model=model)
    for r in prompt_results:
        summary.prompts_tested += 1
        if not r.ok:
            summary.llm_failures += 1
            continue
        if r.intent_predicted == r.intent_expected:
            summary.intent_correct += 1
        if r.industry_predicted == r.industry_expected:
            summary.industry_correct += 1
        for kw in r.keywords:
            vol = volume_by_keyword.get(kw, 0)
            r.keyword_volumes[kw] = vol
            summary.keywords_generated += 1
            if vol > 0:
                summary.keywords_with_msv += 1
                summary.total_msv += vol

    return summary, prompt_results


def run_evaluation(
    models: list[str] | None = None,
    per_industry: int = 2,
    country: str = "us",
) -> tuple[list[ModelEvalResult], list[PromptEvalResult]]:
    models = models or list(CANDIDATE_MODELS)
    ahrefs = AhrefsClient()
    prompts = load_eval_prompts(per_industry=per_industry)

    summaries: list[ModelEvalResult] = []
    all_details: list[PromptEvalResult] = []
    for model in models:
        summary, details = evaluate_model(model, prompts, country=country, ahrefs_client=ahrefs)
        summaries.append(summary)
        all_details.extend(details)

    summaries.sort(key=lambda s: (s.hit_rate, s.total_msv), reverse=True)
    return summaries, all_details
