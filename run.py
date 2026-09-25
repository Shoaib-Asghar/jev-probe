"""Single entry point for the Jev Proving Ground.

Architectural boundary:
- This file contains ZERO business logic.
- It parses arguments, wires dependencies (adapter, store, runner), and delegates to the engine.
"""

import argparse
import sys
import time

from rich.console import Console

from src.adapters.jev_adapter import JevAdapter
from src.cli import (
    pair_and_compare_results,
    render_drilldown_reports,
    render_report_header,
    render_verdicts_table,
)
from src.engine import judge_category, run_sequential
from src.engine.store import RunStore
from src.use_cases.spam import get_spam_use_case


def main() -> None:
    """Command-line entrypoint for running benchmark suites."""
    parser = argparse.ArgumentParser(
        description="jev-probe: Empirical robustness and consistency benchmarking tool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="Run an evaluation pipeline.")
    test_parser.add_argument(
        "--use-case",
        default="spam",
        choices=["spam"],
        help="Target domain scenario to benchmark (default: spam).",
    )
    test_parser.add_argument(
        "--categories",
        nargs="+",
        default=["A", "B"],
        help="Perturbation categories to evaluate (default: A B).",
    )
    test_parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Limit number of seed inputs evaluated (default: all seeds).",
    )
    test_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed drill-down output showing probability distributions and deltas.",
    )
    test_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate calls and cost without hitting the API.",
    )

    args = parser.parse_args()
    console = Console()

    if args.command == "test":
        # 1. Wire Dependencies
        if args.use_case == "spam":
            use_case = get_spam_use_case()
        else:
            console.print(f"[bold red]Unknown use case: {args.use_case}[/bold red]")
            sys.exit(1)

        adapter = JevAdapter()
        store = RunStore()
        store.create_tables()

        if args.dry_run:
            from src.config import settings
            from src.engine.runner import get_perturbation_cases

            active_categories = args.categories or use_case.applicable_perturbations or ["A", "B"]
            seeds = use_case.seed_states[:args.seeds] if args.seeds else use_case.seed_states
            total_calls = 0
            total_input_tokens = 0

            for seed_text in seeds:
                for question in use_case.questions:
                    total_calls += 1
                    total_input_tokens += len(seed_text.split())
                    for cat in active_categories:
                        p_cases = get_perturbation_cases(cat, seed_text, question=question, seed=42)
                        for p_case in p_cases:
                            total_calls += 1
                            total_input_tokens += len(p_case.perturbed_state.split())

            pricing = settings.get_pricing(adapter.provider_name, adapter.model_version)
            estimated_cost = pricing.calculate_cost(
                input_tokens=total_input_tokens, output_tokens=0
            )

            console.print("\n[bold yellow]Dry-Run Execution Plan[/bold yellow]")
            console.print(
                f"[bold cyan]Target Provider:[/bold cyan] {adapter.provider_name.upper()} "
                f"({adapter.model_version})"
            )
            console.print(f"[bold cyan]Scenario:[/bold cyan] {use_case.name}")
            console.print(f"[bold cyan]Seeds:[/bold cyan] {len(seeds)}")
            console.print(f"[bold cyan]Categories:[/bold cyan] {', '.join(active_categories)}")
            console.print(f"[bold cyan]Total API Calls:[/bold cyan] {total_calls:,}")
            console.print(
                f"[bold cyan]Estimated Cost:[/bold cyan] [green]${estimated_cost:.6f}[/green]\n"
            )
            return

        # 2. Execute
        start_wall = time.perf_counter()
        results = run_sequential(
            use_case=use_case,
            adapter=adapter,
            categories=args.categories,
            max_seeds=args.seeds,
        )
        duration_sec = time.perf_counter() - start_wall

        if not results:
            console.print("[yellow]No results returned. Exiting.[/yellow]")
            sys.exit(0)

        # 3. Persist
        batch_id = results[0].batch_id
        store.save_batch(batch_id, results, use_case.name)

        # 4. Presentation
        total_cost = sum(r.cost_usd for r in results)
        total_calls = len(results)

        render_report_header(
            console=console,
            provider=adapter.provider_name,
            model=adapter.model_version,
            use_case=use_case.name,
            duration_sec=duration_sec,
            total_cost=total_cost,
            total_calls=total_calls,
        )

        paired_by_cat = pair_and_compare_results(results)
        verdicts = []
        active_cats = args.categories or ["A", "B"]

        for cat in active_cats:
            cat_key = cat.upper()
            cases = paired_by_cat.get(cat_key, [])
            verdict = judge_category(cat_key, [c.comparison for c in cases])
            verdicts.append(verdict)

        render_verdicts_table(console, verdicts)

        if args.verbose:
            render_drilldown_reports(console, paired_by_cat)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    try:
        main()
    except Exception as e:
        Console(stderr=True).print(f"[bold red]Execution Error:[/bold red] {e}")
        sys.exit(1)
