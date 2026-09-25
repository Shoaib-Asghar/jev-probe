"""CLI presentation layer: Rich-formatted evaluation terminal reports.

Architectural boundary:
- Formats CategoryVerdict and RunResult records into Rich tables and panels.
- Provides interactive terminal commands for running evaluation benchmarks.
- Backend-first presentation before Phase 2 web dashboard.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.engine import (
    CategoryVerdict,
    ComparisonResult,
    compare_runs,
    extract_distribution,
)
from src.models import RunResult

CATEGORY_LABELS: dict[str, str] = {
    "A": "Category A (Surface Noise: Casing & Typos)",
    "B": "Category B (Position Bias: Option Ordering)",
    "C": "Category C (Semantic Perturbations)",
    "D": "Category D (Position & Padding)",
    "E": "Category E (Realistic Mess & OCR)",
    "F": "Category F (Adversarial Probing)",
    "G": "Category G (Repeat-Call Variance)",
}


@dataclass(slots=True)
class CaseComparison:
    """Paired baseline and perturbed RunResult along with their computed ComparisonResult."""

    original: RunResult
    perturbed: RunResult
    comparison: ComparisonResult


def pair_and_compare_results(
    results: list[RunResult],
) -> dict[str, list[CaseComparison]]:
    """Group RunResults by question and seed, pairing baseline runs with perturbed runs."""
    baselines: dict[tuple[str, str], RunResult] = {}
    for r in results:
        if r.perturbation_category == "baseline":
            baselines[(r.question_key, r.original_state)] = r

    comparisons_by_cat: dict[str, list[CaseComparison]] = {}
    for r in results:
        if r.perturbation_category == "baseline":
            continue
        key = (r.question_key, r.original_state)
        base = baselines.get(key)
        if base is not None:
            comp = compare_runs(base, r)
            cat = r.perturbation_category.upper()
            if cat not in comparisons_by_cat:
                comparisons_by_cat[cat] = []
            comparisons_by_cat[cat].append(
                CaseComparison(original=base, perturbed=r, comparison=comp)
            )

    return comparisons_by_cat


def render_report_header(
    console: Console,
    provider: str,
    model: str,
    use_case: str,
    duration_sec: float,
    total_cost: float,
    total_calls: int,
) -> None:
    """Render a styled Rich header panel summarizing benchmark execution metadata."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    header_content = (
        f"[bold cyan]Target Provider:[/bold cyan] {provider.upper()} ({model})\n"
        f"[bold cyan]Scenario / Use Case:[/bold cyan] {use_case}\n"
        f"[bold cyan]Execution Timestamp:[/bold cyan] {timestamp}\n"
        f"[bold cyan]Total API Calls:[/bold cyan] {total_calls:,} calls\n"
        f"[bold cyan]Wall-Clock Latency:[/bold cyan] {duration_sec:.2f}s "
        f"({(duration_sec * 1000.0) / max(1, total_calls):.1f}ms/call avg)\n"
        f"[bold cyan]Total Operational Cost:[/bold cyan] [green]${total_cost:.6f}[/green]"
    )
    console.print(
        Panel(
            header_content,
            title="[bold yellow]Jev Proving Ground — Empirical Evaluation Report[/bold yellow]",
            border_style="cyan",
        )
    )


def render_verdicts_table(
    console: Console,
    verdicts: list[CategoryVerdict],
) -> None:
    """Render a Rich summary table displaying objective pass/warn/fail category verdicts."""
    table = Table(
        title="Robustness & Invariance Quality Gate Summary",
        border_style="dim",
        header_style="bold magenta",
    )

    table.add_column("Category", style="bold", min_width=32)
    table.add_column("Verdict", justify="center", min_width=10)
    table.add_column("Cases", justify="right", min_width=8)
    table.add_column("Flip Rate", justify="right", min_width=12)
    table.add_column("Mean Conf Drift", justify="right", min_width=16)
    table.add_column("Mean Margin Collapse", justify="right", min_width=20)
    table.add_column("Audit Findings / Notes", style="dim", min_width=35)

    status_styles = {
        "pass": "[bold green]PASS[/bold green]",
        "warn": "[bold yellow]WARN[/bold yellow]",
        "fail": "[bold red]FAIL[/bold red]",
    }

    for v in verdicts:
        cat_title = CATEGORY_LABELS.get(v.category, f"Category {v.category}")
        verdict_badge = status_styles.get(v.status, v.status.upper())
        flip_text = f"{v.flip_count}/{v.total_cases} ({v.flip_rate:.1%})"
        if v.flip_rate > 0.0:
            flip_text = (
                f"[yellow]{flip_text}[/yellow]"
                if v.status == "warn"
                else f"[red]{flip_text}[/red]"
            )

        drift_sign = "+" if v.mean_confidence_drift > 0 else ""
        drift_text = f"{drift_sign}{v.mean_confidence_drift:.2%}"
        if v.mean_confidence_drift < -0.10:
            drift_text = f"[red]{drift_text}[/red]"

        margin_text = f"{v.mean_margin_collapse:.2%}"
        if v.mean_margin_collapse > 0.15:
            margin_text = f"[red]{margin_text}[/red]"

        reasons_summary = "; ".join(v.reasons[:2])

        table.add_row(
            cat_title,
            verdict_badge,
            str(v.total_cases),
            flip_text,
            drift_text,
            margin_text,
            reasons_summary,
        )

    console.print(table)


def render_drilldown_reports(
    console: Console,
    paired_by_cat: dict[str, list[CaseComparison]],
) -> None:
    """Render detailed side-by-side probability tables and deltas for every perturbed case."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]Full Probability Distributions, Posterior Deltas & Stability"
            " Drill-Down[/bold yellow]",
            border_style="magenta",
        )
    )

    for cat, cases in paired_by_cat.items():
        cat_label = CATEGORY_LABELS.get(cat, f"Category {cat}")
        console.print(f"\n[bold magenta]--- {cat_label} ({len(cases)} Cases) ---[/bold magenta]\n")

        for idx, item in enumerate(cases, 1):
            comp = item.comparison
            orig = item.original
            pert = item.perturbed

            orig_dist = extract_distribution(orig)
            pert_dist = extract_distribution(pert)
            deltas = comp.distribution_delta

            status_badge = (
                "[bold red]FLIPPED (Failure)[/bold red]"
                if comp.flipped
                else "[bold green]STABLE (Pass)[/bold green]"
            )

            # Format input preview
            orig_text = orig.original_state.strip().replace("\n", " ")
            pert_text = pert.perturbed_state.strip().replace("\n", " ")
            orig_preview = orig_text if len(orig_text) <= 75 else f"{orig_text[:72]}..."
            pert_preview = pert_text if len(pert_text) <= 75 else f"{pert_text[:72]}..."

            case_table = Table(
                title=(
                    f"Case #{idx}: [bold]{pert.transform_name}[/bold] "
                    f"({orig.question_key}) - {status_badge}"
                ),
                caption=(
                    f"Margin: {comp.margin_original:.4f} -> {comp.margin_perturbed:.4f} "
                    f"(collapse: {comp.margin_collapse:+.4f}) | "
                    f"Conf Δ: {comp.confidence_delta:+.4f} | "
                    f"Latency: {pert.latency_ms:.1f}ms (orig: {orig.latency_ms:.1f}ms)"
                ),
                caption_justify="left",
                caption_style="dim",
                border_style="dim",
                header_style="bold cyan",
            )

            case_table.add_column("Option / Class", style="bold", min_width=22)
            case_table.add_column("Original P(x)", justify="right", min_width=18)
            case_table.add_column("Perturbed P(x)", justify="right", min_width=18)
            case_table.add_column("Delta (Δ P)", justify="right", min_width=20)
            case_table.add_column("Distribution Shift", justify="center", min_width=18)

            all_options = sorted(set(orig_dist.keys()) | set(pert_dist.keys()))
            for opt in all_options:
                p_orig = orig_dist.get(opt, 0.0)
                p_pert = pert_dist.get(opt, 0.0)
                delta = deltas.get(opt, round(p_pert - p_orig, 6))

                p_orig_str = f"{p_orig:.4f} ({p_orig:5.1%})"
                p_pert_str = f"{p_pert:.4f} ({p_pert:5.1%})"

                if delta > 0.0005:
                    delta_str = f"[green]+{delta:.4f} (+{delta:5.1%})[/green]"
                    shift_indicator = "[green]+ GAINED MASS[/green]"
                elif delta < -0.0005:
                    delta_str = f"[red]{delta:.4f} ({delta:5.1%})[/red]"
                    shift_indicator = "[red]- LOST MASS[/red]"
                else:
                    delta_str = "[dim] 0.0000 ( 0.0%)[/dim]"
                    shift_indicator = "[dim]- UNCHANGED[/dim]"

                case_table.add_row(opt, p_orig_str, p_pert_str, delta_str, shift_indicator)

            console.print(f"  [dim]Original  Input:[/dim] {orig_preview}")
            console.print(f"  [dim]Perturbed Input:[/dim] {pert_preview}")
            console.print(case_table)
            console.print()



