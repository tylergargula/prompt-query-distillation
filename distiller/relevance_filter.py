"""Post-hoc LLM relevance filter for Ahrefs-expanded related keywords.

Ahrefs' matching-terms expansion is lexical: it surfaces any keyword sharing
terms with a seed, including phrases that share a word or two but are actually
about a different brand, a different entity, or are garbled compound phrases
(e.g. "amazon pricing strategy" or "xlemic+ go to market strategy" for a seed
of "pricing strategy" / "go to market strategy"). This runs ONE batched
classification call per pipeline run - not one call per keyword - to flag
those before they pollute the related MSV total.
"""

from __future__ import annotations

import json
import os

import httpx
from pydantic import BaseModel, ValidationError

from distiller.llm_client import OPENROUTER_URL
from distiller.models import DistillationResult


class RelevanceFilterError(RuntimeError):
    """Raised when the relevance-filter LLM call fails to produce a valid, complete response."""


class _KeywordRelevance(BaseModel):
    keyword: str
    relevant: bool


class _RelevanceResponse(BaseModel):
    results: list[_KeywordRelevance]


RESPONSE_JSON_SCHEMA = {
    "name": "relevance_filter_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string"},
                        "relevant": {"type": "boolean"},
                    },
                    "required": ["keyword", "relevant"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


def _build_system_prompt(prompt: str, distillation: DistillationResult) -> str:
    keyword_list = ", ".join(distillation.keywords)
    return f"""You are a keyword-relevance filter for an SEO keyword research tool.

The user's original search prompt was: "{prompt}"
It was classified as intent={distillation.intent}, industry={distillation.industry}.
The core distilled keywords for this prompt are: {keyword_list}.

You will be given a JSON object {{"keywords": [...]}} of "related keywords" pulled from a keyword
expansion API because they lexically share terms with the core keywords above. Some are genuinely
relevant search variants a real person researching the same topic would use. Others merely share a
word or two but are actually about a different topic, a specific brand/competitor's own strategy or
product that isn't what the user is researching, a different industry, or are garbled/nonsensical
compound phrases.

Mark relevant=true ONLY for keywords a person researching "{prompt}" would plausibly also search
for. Mark relevant=false for anything off-topic, oddly brand-specific, or garbled.

Respond with ONLY a JSON object: {{"results": [{{"keyword": "...", "relevant": true|false}}, ...]}}
Include exactly one entry per input keyword, in the same order, with no additions or omissions."""


def filter_related_keywords(
    prompt: str,
    distillation: DistillationResult,
    keywords: list[str],
    model: str,
    max_retries: int = 3,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> tuple[list[str], list[str]]:
    """Classify `keywords` as relevant/irrelevant to `prompt` via one batched LLM call.

    Returns (relevant_keywords, flagged_keywords), both subsets of `keywords` in original order.
    Raises RelevanceFilterError if the LLM fails to return a valid, complete classification after
    `max_retries` attempts - callers should fail open (keep all keywords) rather than propagate
    this as a hard pipeline failure, since filtering is a quality layer, not the source of truth.
    """
    if not keywords:
        return [], []

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RelevanceFilterError("OPENROUTER_API_KEY not set. Add it to .env or pass api_key=.")

    system_prompt = _build_system_prompt(prompt, distillation)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"keywords": keywords})},
    ]

    last_error: Exception | None = None
    use_schema = True
    with httpx.Client(timeout=timeout) as client:
        for _attempt in range(1, max_retries + 1):
            response_format = (
                {"type": "json_schema", "json_schema": RESPONSE_JSON_SCHEMA}
                if use_schema
                else {"type": "json_object"}
            )
            resp = client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "response_format": response_format,
                },
            )

            if resp.status_code != 200:
                last_error = RelevanceFilterError(
                    f"OpenRouter request failed ({resp.status_code}): {resp.text[:500]}"
                )
                if use_schema and "response_format" in resp.text:
                    use_schema = False
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response caused an API error. "
                            "Re-emit ONLY the required JSON object."
                        ),
                    }
                )
                continue

            body = resp.json()
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                last_error = RelevanceFilterError(f"Unexpected OpenRouter response shape: {body}")
                continue

            try:
                parsed = json.loads(content)
                result = _RelevanceResponse.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"That response was invalid ({exc}). Re-emit ONLY the JSON object described.",
                    }
                )
                continue

            by_keyword = {r.keyword: r.relevant for r in result.results}
            if set(by_keyword) != set(keywords):
                last_error = RelevanceFilterError(
                    "Relevance filter response keyword set didn't match the input keyword set."
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your response's keywords didn't exactly match the input list. "
                            "Re-emit ONLY the JSON object, with exactly one entry per input keyword."
                        ),
                    }
                )
                continue

            relevant = [kw for kw in keywords if by_keyword[kw]]
            flagged = [kw for kw in keywords if not by_keyword[kw]]
            return relevant, flagged

    raise RelevanceFilterError(
        f"Relevance filter failed to produce a valid classification after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
