#!/usr/bin/env python3
"""CLI entry point for prompt-keyword distillation.

Usage:
    python main.py "What is the best CRM for a small agency" --country us
    python main.py "..." --model google/gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from dotenv import load_dotenv
from rich.console import Console

from distiller.ahrefs_client import AhrefsClient
from distiller.csv_export import export_csv
from distiller.display import render_report
from distiller.llm_client import CANDIDATE_MODELS, DEFAULT_MODEL, DistillationError
from distiller.pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill a prompt into B2B/B2C keywords and score them by Ahrefs search volume.",
    )
    parser.add_argument("prompt", help="The natural-language prompt to distill.")
    parser.add_argument(
        "--country",
        default="us",
        help="ISO 3166-1 alpha-2 country code for MSV lookup (default: us).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id to use for distillation (default: {DEFAULT_MODEL}). "
        f"Known-good candidates: {', '.join(CANDIDATE_MODELS)}.",
    )
    parser.add_argument(
        "--related-limit",
        type=int,
        default=100,
        help="Max related/matching terms to fetch from Ahrefs (default: 100).",
    )
    parser.add_argument(
        "--related-display-limit",
        type=int,
        default=25,
        help="Max related terms to print in the table (default: 25).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export all target and related keyword data to a CSV file in outputs/.",
    )
    parser.add_argument(
        "--filter-related",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run a batched LLM relevance check on related/matching-terms keywords and exclude "
            "off-topic ones from the related MSV total (default: on; use --no-filter-related to disable)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        ahrefs = AhrefsClient()
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    try:
        report = run_pipeline(
            prompt=args.prompt,
            country=args.country,
            model=args.model,
            related_limit=args.related_limit,
            related_display_limit=args.related_display_limit,
            filter_related=args.filter_related,
            ahrefs_client=ahrefs,
        )
    except DistillationError as exc:
        console.print(f"[bold red]Distillation failed:[/bold red] {exc}")
        return 1
    except RuntimeError as exc:
        console.print(f"[bold red]Ahrefs API error:[/bold red] {exc}")
        return 1

    render_report(report, console=console)

    if args.csv:
        csv_path = export_csv(report, date.today().strftime("%Y%m%d"))
        console.print(f"\n[bold green]CSV exported:[/bold green] {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
