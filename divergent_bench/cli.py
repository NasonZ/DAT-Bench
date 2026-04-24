"""Command-line interface for DAT-Bench."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer

from .config.strategies import DAT_STRATEGIES, DEFAULT_TEMPERATURES

app = typer.Typer(
    name="divergent-bench",
    help="DAT-Bench: run Divergent Association Task experiments.",
    no_args_is_help=True,
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def run(
    provider: str = typer.Option("openai", help="LLM provider: openai, anthropic, gemini, ollama, openrouter."),
    model: str | None = typer.Option(None, help="Model name to use."),
    strategy: str = typer.Option("none", help=f"DAT strategy. Available: {', '.join(DAT_STRATEGIES)}"),
    temperature: float | None = typer.Option(None, help="Generation temperature; defaults to the strategy temperature."),
    samples: int = typer.Option(10, min=1, help="Number of samples to generate."),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run one DAT experiment condition."""
    if strategy not in DAT_STRATEGIES:
        raise typer.BadParameter(f"Unknown strategy '{strategy}'. Available: {', '.join(DAT_STRATEGIES)}")

    _configure_logging(verbose)

    # Lazy import keeps `divergent-bench --help` fast and independent of API/GloVe setup.
    from .experiments.runner import ExperimentRunner

    runner = ExperimentRunner(provider=provider, model=model)
    results = asyncio.run(
        runner.run_dat_experiment(
            strategy=strategy,
            temperature=temperature,
            num_samples=samples,
            save_incrementally=output is None,
        )
    )
    analysis = runner.analyze_results(results)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"results": results, "analysis": analysis}, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"Results saved to {output}")

    typer.echo("\n" + "=" * 60)
    typer.echo(f"DAT EXPERIMENT RESULTS - {strategy.upper()}")
    typer.echo("=" * 60)
    typer.echo(f"Provider: {provider}")
    typer.echo(f"Model: {model or 'default'}")
    typer.echo(f"Strategy: {strategy}")
    typer.echo(f"Temperature: {temperature if temperature is not None else DEFAULT_TEMPERATURES.get(strategy, 0.7)}")
    typer.echo(f"Samples: {samples}")
    typer.echo("-" * 60)

    if "error" in analysis:
        typer.echo(f"Error: {analysis['error']}")
        raise typer.Exit(code=1)

    typer.echo(f"Valid samples: {analysis['num_valid']}/{analysis['num_samples']}")
    typer.echo(f"Mean DAT score: {analysis['mean_score']:.2f}")
    typer.echo(f"Std deviation: {analysis['std_score']:.2f}")
    typer.echo(f"Min score: {analysis['min_score']:.2f}")
    typer.echo(f"Max score: {analysis['max_score']:.2f}")
    typer.echo(f"Median score: {analysis['median_score']:.2f}")


@app.command()
def strategies() -> None:
    """List available prompting strategies."""
    for name in DAT_STRATEGIES:
        typer.echo(f"{name}\ttemperature={DEFAULT_TEMPERATURES.get(name, 0.7)}")
