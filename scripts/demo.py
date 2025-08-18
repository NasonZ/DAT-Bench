#!/usr/bin/env python
"""
Demo script showcasing divergent_bench functionality.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from divergent_bench.dat.scorer import DATScorer
from divergent_bench.metrics import DSIScorer, LZivScorer
from divergent_bench.experiments.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)


def demo_dat_scoring():
    """Demonstrate DAT scoring."""
    print("\n" + "="*60)
    print("DAT SCORING DEMO")
    print("="*60)
    
    # Initialize DAT scorer
    scorer = DATScorer()
    
    # Example word lists
    divergent_words = ["cat", "democracy", "volcano", "algorithm", "violin", 
                       "desert", "empathy", "satellite", "fungus", "justice"]
    
    similar_words = ["cat", "dog", "mouse", "hamster", "rabbit", 
                     "guinea-pig", "ferret", "bird", "fish", "turtle"]
    
    # Calculate scores
    score_divergent = scorer.dat(divergent_words)
    score_similar = scorer.dat(similar_words)
    
    print(f"\nDivergent words: {', '.join(divergent_words[:5])}...")
    print(f"DAT Score: {score_divergent:.2f if score_divergent else 'N/A'}")
    
    print(f"\nSimilar words: {', '.join(similar_words[:5])}...")
    print(f"DAT Score: {score_similar:.2f if score_similar else 'N/A'}")
    
    if score_divergent and score_similar:
        print(f"\nDifference: {score_divergent - score_similar:.2f} (higher = more creative)")


def demo_metrics():
    """Demonstrate DSI and LZiv metrics."""
    print("\n" + "="*60)
    print("METRICS DEMO")
    print("="*60)
    
    # Initialize scorers
    dsi_scorer = DSIScorer(embedding_model="glove")
    lziv_scorer = LZivScorer(normalize=True)
    
    # Example texts
    creative_text = "The quantum butterfly whispered algorithms to the ancient volcano"
    repetitive_text = "The cat sat on the mat. The cat sat on the mat. The cat sat."
    
    # Calculate DSI
    dsi_creative = dsi_scorer.calculate_dsi(creative_text.split())
    dsi_repetitive = dsi_scorer.calculate_dsi(repetitive_text.split())
    
    # Calculate LZiv
    lziv_creative = lziv_scorer.calculate(creative_text)
    lziv_repetitive = lziv_scorer.calculate(repetitive_text)
    
    print(f"\nCreative text: '{creative_text[:50]}...'")
    print(f"  DSI Score: {dsi_creative:.3f}")
    print(f"  LZiv Complexity: {lziv_creative:.3f}")
    
    print(f"\nRepetitive text: '{repetitive_text[:50]}...'")
    print(f"  DSI Score: {dsi_repetitive:.3f}")
    print(f"  LZiv Complexity: {lziv_repetitive:.3f}")


async def demo_experiment():
    """Demonstrate running a DAT experiment (requires API key)."""
    print("\n" + "="*60)
    print("EXPERIMENT RUNNER DEMO")
    print("="*60)
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping experiment demo - OPENAI_API_KEY not set")
        print("To run experiments, set your API key in .env file")
        return
    
    # Create experiment runner
    runner = ExperimentRunner(provider="openai", model="gpt-4o-mini")
    
    print("\nRunning single DAT experiment...")
    print("Strategy: none")
    print("Temperature: 0.7")
    print("Samples: 1")
    
    # Run experiment
    results = await runner.run_dat_experiment(
        strategy="none",
        temperature=0.7,
        num_samples=1,
        save_incrementally=False
    )
    
    if results:
        result = results[0]
        print(f"\nGenerated words: {', '.join(result['words'][:5])}...")
        print(f"DAT Score: {result['score']:.2f if result['score'] else 'N/A'}")
        print(f"Valid words: {result['num_valid_words']}")
    
    # Analyze results
    analysis = runner.analyze_results(results)
    print(f"\nAnalysis: {analysis}")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("# DIVERGENT_BENCH DEMO")
    print("#"*60)
    
    # Run DAT scoring demo
    try:
        demo_dat_scoring()
    except Exception as e:
        print(f"DAT demo error: {e}")
    
    # Run metrics demo
    try:
        demo_metrics()
    except Exception as e:
        print(f"Metrics demo error: {e}")
    
    # Run experiment demo (async)
    try:
        asyncio.run(demo_experiment())
    except Exception as e:
        print(f"Experiment demo error: {e}")
    
    print("\n" + "#"*60)
    print("# Demo complete!")
    print("# Run 'python scripts/run_dat.py --help' for CLI usage")
    print("#"*60)


if __name__ == "__main__":
    main()