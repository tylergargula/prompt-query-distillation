"""Loads few-shot keyword-distillation examples from the reference file."""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE_FILE = (
    Path(__file__).resolve().parent.parent / "reference-files" / "reference_distillation.json"
)


def load_few_shots(per_industry: int = 1) -> list[dict]:
    """Return a flat list of {intent, industry, prompt, keywords} examples.

    Takes the first `per_industry` examples from each industry bucket so the
    few-shot set is deterministic and covers every known industry.
    """
    data = json.loads(REFERENCE_FILE.read_text())
    examples = []
    for intent in ("B2B", "B2C"):
        for industry, entries in data[intent].items():
            for entry in entries[:per_industry]:
                examples.append(
                    {
                        "intent": intent,
                        "industry": industry,
                        "prompt": entry["prompt"],
                        "keywords": entry["keywords"],
                    }
                )
    return examples
