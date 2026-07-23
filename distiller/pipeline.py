"""Orchestrates: LLM distillation -> Ahrefs target MSV -> Ahrefs related MSV -> universe MSV."""

from __future__ import annotations

from dataclasses import dataclass, field

from distiller.ahrefs_client import AhrefsClient
from distiller.llm_client import DEFAULT_MODEL, distill_prompt
from distiller.models import DistillationResult
from distiller.relevance_filter import RelevanceFilterError, filter_related_keywords


@dataclass
class KeywordVolume:
    keyword: str
    volume: int
    global_volume: int


@dataclass
class DistillationReport:
    prompt: str
    country: str
    model: str
    distillation: DistillationResult
    target_rows: list[KeywordVolume]
    related_rows: list[KeywordVolume]
    related_display_limit: int
    related_total_available: int
    filter_related: bool = True
    related_flagged_rows: list[KeywordVolume] = field(default_factory=list)
    relevance_filter_error: str | None = None

    @property
    def target_msv(self) -> int:
        return sum(row.volume for row in self.target_rows)

    @property
    def related_msv(self) -> int:
        return sum(row.volume for row in self.related_rows)

    @property
    def universe_msv(self) -> int:
        return self.target_msv + self.related_msv


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = " ".join(item.strip().lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def run_pipeline(
    prompt: str,
    country: str = "us",
    model: str = DEFAULT_MODEL,
    related_limit: int = 100,
    related_display_limit: int = 25,
    filter_related: bool = True,
    ahrefs_client: AhrefsClient | None = None,
) -> DistillationReport:
    """Run the full prompt -> keyword -> MSV distillation pipeline for one prompt."""
    result = distill_prompt(prompt, model=model)

    ahrefs = ahrefs_client or AhrefsClient()

    # 1. Target MSV: the base prompt itself + every distilled keyword.
    target_keywords = _dedup_keep_order([prompt] + result.keywords)
    target_volumes_raw = ahrefs.keywords_overview(target_keywords, country=country)
    volume_by_keyword = {
        " ".join(r["keyword"].strip().lower().split()): r
        for r in target_volumes_raw
    }

    target_rows = [
        KeywordVolume(
            keyword=kw,
            volume=volume_by_keyword.get(kw, {}).get("volume", 0) or 0,
            global_volume=volume_by_keyword.get(kw, {}).get("global_volume", 0) or 0,
        )
        for kw in target_keywords
    ]

    # 2. Related MSV: expand the distilled keywords (the prompt's searchable
    # representation) into matching terms, deduped against the target set.
    related_raw = ahrefs.matching_terms(
        result.keywords,
        country=country,
        match_mode="terms",
        limit=related_limit,
    )
    target_set = set(target_keywords)
    related_rows_all = []
    seen_related: set[str] = set()
    for r in related_raw:
        kw = " ".join(r["keyword"].strip().lower().split())
        if kw in target_set or kw in seen_related:
            continue
        seen_related.add(kw)
        related_rows_all.append(
            KeywordVolume(
                keyword=kw,
                volume=r.get("volume", 0) or 0,
                global_volume=r.get("global_volume", 0) or 0,
            )
        )

    related_rows_all.sort(key=lambda row: row.volume, reverse=True)

    related_flagged_rows: list[KeywordVolume] = []
    relevance_filter_error: str | None = None
    if filter_related and related_rows_all:
        try:
            _relevant, flagged = filter_related_keywords(
                prompt, result, [row.keyword for row in related_rows_all], model=model
            )
            flagged_set = set(flagged)
            related_flagged_rows = [r for r in related_rows_all if r.keyword in flagged_set]
            related_rows_all = [r for r in related_rows_all if r.keyword not in flagged_set]
        except RelevanceFilterError as exc:
            relevance_filter_error = str(exc)

    return DistillationReport(
        prompt=prompt,
        country=country,
        model=model,
        distillation=result,
        target_rows=target_rows,
        related_rows=related_rows_all,
        related_display_limit=related_display_limit,
        related_total_available=len(related_rows_all),
        filter_related=filter_related,
        related_flagged_rows=related_flagged_rows,
        relevance_filter_error=relevance_filter_error,
    )
