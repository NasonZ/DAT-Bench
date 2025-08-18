#!/usr/bin/env python
"""
CLI for running DAT experiments.
Based on DAT_GPT API scripts but simplified for divergent_bench.
"""

import asyncio
import click
import json
import logging
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from divergent_bench.experiments.runner import ExperimentRunner
from divergent_bench.config.strategies import DAT_STRATEGIES, DEFAULT_TEMPERATURES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--provider", 
    default="openai",
    help="LLM provider (openai, anthropic, gemini, ollama)"
)
@click.option(
    "--model",
    default=None,
    help="Specific model to use (e.g., gpt-4, claude-3-5-sonnet)"
)
@click.option(
    "--strategy",
    default="none",
    type=click.Choice(list(DAT_STRATEGIES.keys())),
    help="DAT strategy to use"
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Generation temperature (uses strategy default if not specified)"
)
@click.option(
    "--samples",
    default=10,
    type=int,
    help="Number of samples to generate"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file path (default: results/<provider>_<strategy>_<timestamp>.json)"
)
@click.option(
    "--batch",
    is_flag=True,
    help="Run batch experiments with multiple strategies"
)
def main(provider, model, strategy, temperature, samples, output, batch):
    """
    Run DAT experiments with specified LLM and strategy.
    
    Examples:
    
    \b
    # Run single experiment with OpenAI
    python scripts/run_dat.py --provider openai --strategy none --samples 10
    
    \b
    # Run with specific model and temperature
    python scripts/run_dat.py --provider anthropic --model claude-3-5-sonnet --temperature 0.8
    
    \b
    # Run batch experiments
    python scripts/run_dat.py --batch --provider openai --samples 5
    """
    
    # Initialize runner
    runner = ExperimentRunner(provider=provider, model=model)
    
    if batch:
        # Run batch experiments with all strategies
        logger.info("Running batch experiments with all strategies...")
        strategies = list(DAT_STRATEGIES.keys())
        temperatures = [0.5, 0.7, 1.0]
        
        results = asyncio.run(
            runner.run_batch_experiments(
                strategies=strategies,
                temperatures=temperatures,
                samples_per_condition=samples
            )
        )
        
        # Save batch results
        output_path = output or f"results/batch_{provider}_{Path().stem}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch results saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH EXPERIMENT SUMMARY")
        print("="*60)
        for key, condition_results in results.items():
            analysis = runner.analyze_results(condition_results)
            print(f"\n{key}:")
            print(f"  Valid samples: {analysis.get('num_valid', 0)}/{analysis.get('num_samples', 0)}")
            if 'mean_score' in analysis:
                print(f"  Mean DAT score: {analysis['mean_score']:.2f}")
                print(f"  Std deviation: {analysis['std_score']:.2f}")
                print(f"  Range: {analysis['min_score']:.2f} - {analysis['max_score']:.2f}")
    
    else:
        # Run single experiment
        logger.info(f"Running DAT experiment: {strategy=}, {temperature=}, {samples=}")
        
        results = asyncio.run(
            runner.run_dat_experiment(
                strategy=strategy,
                temperature=temperature,
                num_samples=samples,
                save_incrementally=True
            )
        )
        
        # Analyze results
        analysis = runner.analyze_results(results)
        
        # Save if output path specified
        if output:
            with open(output, 'w') as f:
                json.dump({
                    "results": results,
                    "analysis": analysis
                }, f, indent=2)
            logger.info(f"Results saved to {output}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"DAT EXPERIMENT RESULTS - {strategy.upper()}")
        print("="*60)
        print(f"Provider: {provider}")
        print(f"Model: {model or 'default'}")
        print(f"Strategy: {strategy}")
        print(f"Temperature: {temperature or DEFAULT_TEMPERATURES.get(strategy, 0.7)}")
        print(f"Samples: {samples}")
        print("-"*60)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
        else:
            print(f"Valid samples: {analysis['num_valid']}/{analysis['num_samples']}")
            print(f"Mean DAT score: {analysis['mean_score']:.2f}")
            print(f"Std deviation: {analysis['std_score']:.2f}")
            print(f"Min score: {analysis['min_score']:.2f}")
            print(f"Max score: {analysis['max_score']:.2f}")
            print(f"Median score: {analysis['median_score']:.2f}")
        
        # Show sample outputs
        if results:
            print("\nSample outputs:")
            for i, result in enumerate(results[:3]):  # Show first 3
                print(f"\n  Sample {i+1}:")
                print(f"    Words: {', '.join(result['words'][:5])}...")
                if result['score']:
                    print(f"    Score: {result['score']:.2f}")


if __name__ == "__main__":
    main()