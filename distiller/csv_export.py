"""CSV export of a DistillationReport's target and related keyword data."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from distiller.pipeline import DistillationReport

OUTPUT_DIR = Path("outputs")
_SLUG_MAX_LEN = 60


def _slugify_prompt(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_")
    return slug[:_SLUG_MAX_LEN].rstrip("_") or "prompt"


def build_csv_path(prompt: str, date: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """date must be a pre-formatted yyyymmdd string (callers stamp the date, not this module)."""
    return output_dir / f"{_slugify_prompt(prompt)}_distillation_{date}.csv"


def export_csv(report: DistillationReport, date: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Write all target and related keyword rows for `report` to a CSV file, returning its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = build_csv_path(report.prompt, date, output_dir=output_dir)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt",
                "intent",
                "industry",
                "country",
                "model",
                "keyword",
                "keyword_type",
                "volume",
                "global_volume",
            ]
        )
        for row in report.target_rows:
            writer.writerow(
                [
                    report.prompt,
                    report.distillation.intent,
                    report.distillation.industry,
                    report.country,
                    report.model,
                    row.keyword,
                    "target",
                    row.volume,
                    row.global_volume,
                ]
            )
        for row in report.related_rows:
            writer.writerow(
                [
                    report.prompt,
                    report.distillation.intent,
                    report.distillation.industry,
                    report.country,
                    report.model,
                    row.keyword,
                    "related",
                    row.volume,
                    row.global_volume,
                ]
            )
        for row in report.related_flagged_rows:
            writer.writerow(
                [
                    report.prompt,
                    report.distillation.intent,
                    report.distillation.industry,
                    report.country,
                    report.model,
                    row.keyword,
                    "related_flagged_irrelevant",
                    row.volume,
                    row.global_volume,
                ]
            )

    return path
