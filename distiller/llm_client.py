"""OpenRouter-backed keyword distillation with few-shot prompting and strict
pydantic validation. Any hallucinated/malformed output is retried with the
validation error fed back to the model; exhausting retries raises DistillationError.
"""

from __future__ import annotations

import json
import os

import httpx
from pydantic import ValidationError

from distiller.few_shots import load_few_shots
from distiller.models import B2B_INDUSTRIES, B2C_INDUSTRIES, DistillationResult

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# gemini-2.5-flash edged out gpt-5.4-mini only slightly on MSV hit rate (66% vs
# 78% in a small eval) but is substantially cheaper per call, so it's the
# default. Re-run evaluate_models.py if that cost/quality tradeoff should be
# revisited.
DEFAULT_MODEL = "google/gemini-2.5-flash"

# meta-llama/llama-4-maverick and mistralai/mistral-small-24b-instruct-2501 are
# excluded: this account's OpenRouter provider allowlist (openai, deepseek,
# anthropic, google-ai-studio) doesn't include any provider that hosts them,
# so every call 404s with "No allowed providers are available for the selected model."
CANDIDATE_MODELS = (
    "google/gemini-2.5-flash",
    "openai/gpt-5.4-mini",
    "anthropic/claude-opus-4.7-fast",
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-v4-flash",
)

RESPONSE_JSON_SCHEMA = {
    "name": "distillation_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["B2B", "B2C"]},
            "industry": {
                "type": "string",
                "enum": list(B2B_INDUSTRIES) + list(B2C_INDUSTRIES),
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["intent", "industry", "keywords"],
        "additionalProperties": False,
    },
}


class DistillationError(RuntimeError):
    """Raised when the LLM fails to produce a valid DistillationResult after retries."""


def _build_system_prompt() -> str:
    b2b_list = ", ".join(B2B_INDUSTRIES)
    b2c_list = ", ".join(B2C_INDUSTRIES)
    examples = load_few_shots(per_industry=1)
    example_lines = []
    for ex in examples:
        example_lines.append(
            json.dumps(
                {
                    "prompt": ex["prompt"],
                    "intent": ex["intent"],
                    "industry": ex["industry"],
                    "keywords": ex["keywords"],
                }
            )
        )
    examples_block = "\n".join(example_lines)

    return f"""You are a search-intent and keyword distillation engine for a B2B/B2C SEO tool.

Given a user prompt (a natural-language question someone might ask a search engine or an AI assistant),
you must:
1. Classify the prompt's commercial intent as exactly one of: B2B, B2C.
2. Assign the single best-fit industry for that intent from its fixed list:
   - B2B industries: {b2b_list}
   - B2C industries: {b2c_list}
   Only ever choose an industry from the list matching the intent you chose.
3. Distill the prompt into 3 to 6 short, search-engine-style keywords: the core head term(s),
   qualifier variants, and the way a real user would type this into a search bar. Keywords should
   be lowercase, concise (1-6 words), and should NOT simply restate the full prompt as one long string.
   Only include a keyword if it is a distinct, natural search variant — do NOT pad the list with
   redundant or low-value filler just to reach 6. Fewer strong keywords (as few as 3) are better
   than extra weak ones.

Here are worked examples (one per industry), each as a JSON object with prompt/intent/industry/keywords:
{examples_block}

Respond with ONLY a JSON object matching this exact shape: {{"intent": "B2B"|"B2C", "industry": "<one of the listed keys>", "keywords": ["...", "..."]}}
No prose, no markdown fences, no explanation."""


def distill_prompt(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> DistillationResult:
    """Call the LLM to distill `prompt` into a validated DistillationResult.

    Retries on malformed JSON or pydantic validation failure, feeding the error
    back to the model so it can self-correct. Raises DistillationError if all
    attempts fail.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise DistillationError(
            "OPENROUTER_API_KEY not set. Add it to .env or pass api_key=."
        )

    system_prompt = _build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Prompt to distill:\n{prompt}"},
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
                last_error = DistillationError(
                    f"OpenRouter request failed ({resp.status_code}): {resp.text[:500]}"
                )
                if use_schema and "response_format" in resp.text:
                    # Some providers (e.g. DeepSeek) don't support strict json_schema
                    # mode yet; fall back to plain JSON mode and rely on the
                    # prompt instructions + pydantic validation/retry instead.
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
                last_error = DistillationError(f"Unexpected OpenRouter response shape: {body}")
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That was not valid JSON ({exc}). "
                            "Re-emit ONLY the JSON object described in the instructions."
                        ),
                    }
                )
                continue

            try:
                return DistillationResult.model_validate(parsed)
            except ValidationError as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your JSON failed schema validation: {exc}. "
                            "Fix the fields and re-emit ONLY the corrected JSON object."
                        ),
                    }
                )
                continue

    raise DistillationError(
        f"LLM failed to produce a valid DistillationResult after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
