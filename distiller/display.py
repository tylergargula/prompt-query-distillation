"""Renders a DistillationReport to the terminal via rich tables + a bar chart."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from distiller.pipeline import DistillationReport

BAR_WIDTH = 40


def _bar(volume: int, max_volume: int, width: int = BAR_WIDTH) -> str:
    if max_volume <= 0:
        return ""
    filled = round((volume / max_volume) * width)
    return "█" * filled


def render_report(report: DistillationReport, console: Console | None = None) -> None:
    console = console or Console()
    d = report.distillation

    header = Text()
    header.append("Prompt: ", style="bold")
    header.append(f"{report.prompt}\n")
    header.append("Intent: ", style="bold")
    header.append(f"{d.intent}  ")
    header.append("Industry: ", style="bold")
    header.append(f"{d.industry}  ")
    header.append("Country: ", style="bold")
    header.append(f"{report.country.upper()}  ")
    header.append("Model: ", style="bold")
    header.append(f"{report.model}")
    console.print(Panel(header, title="Distillation Summary", border_style="cyan"))

    # Table 1: base prompt + keywords, MSV each
    kw_table = Table(title="Prompt & Distilled Keywords — Monthly Search Volume (target)")
    kw_table.add_column("#", justify="right")
    kw_table.add_column("Keyword")
    kw_table.add_column("Type")
    kw_table.add_column("Volume (US MSV)" if report.country == "us" else "Volume (MSV)", justify="right")
    kw_table.add_column("Global Volume", justify="right")

    for i, row in enumerate(report.target_rows, start=1):
        kw_type = "base prompt" if row.keyword == " ".join(report.prompt.strip().lower().split()) else "keyword"
        kw_table.add_row(
            str(i),
            row.keyword,
            kw_type,
            f"{row.volume:,}",
            f"{row.global_volume:,}",
        )
    console.print(kw_table)

    # Table 2: related keywords (top N shown)
    if report.related_rows:
        shown = report.related_rows[: report.related_display_limit]
        rel_table = Table(
            title=(
                f"Related Keywords (top {len(shown)} of {report.related_total_available} "
                "found, all counted toward related MSV)"
            )
        )
        rel_table.add_column("#", justify="right")
        rel_table.add_column("Keyword")
        rel_table.add_column("Volume", justify="right")
        for i, row in enumerate(shown, start=1):
            rel_table.add_row(str(i), row.keyword, f"{row.volume:,}")
        console.print(rel_table)
    else:
        console.print("[yellow]No related keywords found via Ahrefs matching-terms.[/yellow]")

    if report.relevance_filter_error:
        console.print(
            f"[yellow]Relevance filtering skipped (kept all related keywords): "
            f"{report.relevance_filter_error}[/yellow]"
        )
    elif report.filter_related and report.related_flagged_rows:
        flagged_table = Table(
            title=(
                f"Related Keywords Flagged as Irrelevant "
                f"({len(report.related_flagged_rows)} excluded from related MSV)"
            ),
            border_style="red",
        )
        flagged_table.add_column("#", justify="right")
        flagged_table.add_column("Keyword")
        flagged_table.add_column("Volume", justify="right")
        for i, row in enumerate(report.related_flagged_rows, start=1):
            flagged_table.add_row(str(i), row.keyword, f"{row.volume:,}")
        console.print(flagged_table)

    # Table 3: target vs related vs universe breakdown
    breakdown = Table(title="Keyword Universe MSV Breakdown")
    breakdown.add_column("Segment")
    breakdown.add_column("Keyword Count", justify="right")
    breakdown.add_column("MSV", justify="right")
    breakdown.add_column("Share", justify="right")

    universe = report.universe_msv or 1  # avoid div-by-zero for share calc
    breakdown.add_row(
        "Target (prompt + distilled keywords)",
        str(len(report.target_rows)),
        f"{report.target_msv:,}",
        f"{report.target_msv / universe:.0%}",
    )
    breakdown.add_row(
        "Related (expanded matching terms)",
        str(len(report.related_rows)),
        f"{report.related_msv:,}",
        f"{report.related_msv / universe:.0%}",
    )
    breakdown.add_row(
        "[bold]Keyword Universe Total[/bold]",
        str(len(report.target_rows) + len(report.related_rows)),
        f"[bold]{report.universe_msv:,}[/bold]",
        "100%",
    )
    console.print(breakdown)

    # Bonus: simple bar chart comparing target vs related MSV
    max_v = max(report.target_msv, report.related_msv, 1)
    chart_lines = [
        f"{'Target':<10} {_bar(report.target_msv, max_v):<{BAR_WIDTH}} {report.target_msv:,}",
        f"{'Related':<10} {_bar(report.related_msv, max_v):<{BAR_WIDTH}} {report.related_msv:,}",
    ]
    console.print(Panel("\n".join(chart_lines), title="Target vs Related MSV", border_style="magenta"))
