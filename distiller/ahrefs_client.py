"""Minimal Ahrefs API v3 client for keyword volume (MSV) lookups.

Only implements the two endpoints this project needs: keywords-overview
(known-keyword volume) and matching-terms (related-keyword expansion).
See /Users/tylergargula/.claude/skills/ahrefs-api for the fuller reference client.
"""

from __future__ import annotations

import os
import time

import httpx

BASE_URL = "https://api.ahrefs.com/v3"


class AhrefsClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        retry_delay: float = 5.0,
        courtesy_delay: float = 0.5,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("AHREFS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AHREFS_API_KEY not set. Pass api_key= or set the environment variable."
            )
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.courtesy_delay = courtesy_delay
        self.max_retries = max_retries
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get(self, path: str, params: dict) -> dict:
        url = f"{BASE_URL}/{path}"
        for attempt in range(self.max_retries + 1):
            resp = httpx.get(url, headers=self.headers, params=params, timeout=self.timeout)

            if resp.status_code == 429 and attempt < self.max_retries:
                time.sleep(self.retry_delay)
                continue

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ahrefs API error {resp.status_code} on {path}: {resp.text[:500]}"
                )

            return resp.json()

        raise RuntimeError(f"Ahrefs API rate limited after {self.max_retries} retries on {path}")

    def keywords_overview(
        self,
        keywords: list[str],
        country: str = "us",
        select: str = "keyword,volume,global_volume",
        chunk_size: int = 200,
    ) -> list[dict]:
        """Volume lookup for a known list of keywords. Auto-chunks large lists."""
        rows: list[dict] = []
        for i in range(0, len(keywords), chunk_size):
            chunk = keywords[i : i + chunk_size]
            data = self._get(
                "keywords-explorer/overview",
                {
                    "country": country,
                    "keywords": ",".join(chunk),
                    "select": select,
                },
            )
            rows.extend(data.get("keywords", []))
            if i + chunk_size < len(keywords):
                time.sleep(self.courtesy_delay)
        return rows

    def matching_terms(
        self,
        keywords: list[str],
        country: str = "us",
        match_mode: str = "terms",
        order_by: str = "volume:desc",
        select: str = "keyword,volume,global_volume",
        limit: int = 100,
    ) -> list[dict]:
        """Expand a set of seed keywords into related/matching terms with volumes."""
        kw_param = ",".join(keywords[:50])
        data = self._get(
            "keywords-explorer/matching-terms",
            {
                "keywords": kw_param,
                "country": country,
                "match_mode": match_mode,
                "order_by": order_by,
                "select": select,
                "limit": limit,
            },
        )
        return data.get("keywords", [])
