#!/usr/bin/env python
"""
Test script for visualization module.
Tests with actual results from divergent_bench.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from divergent_bench.visualization import (
    load_results,
    prepare_data,
    ridge_plot,
    statistical_heatmap,
    word_frequency_stacked,
    triangular_matrix,
    apply_plot_style,
    get_model_summary,
    get_best_runs
)
from divergent_bench.dat.scorer import DATScorer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_visualizations():
    """Test all visualization functions with real data."""
    
    # 1. Load results
    logger.info("Loading results...")
    results_dir = Path("results")
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    df = load_results(results_dir)
    logger.info(f"Loaded {len(df)} results")
    
    if df.empty:
        logger.error("No results loaded")
        return
    
    # Show what we have
    logger.info("\nAvailable models:")
    if 'display_model' in df.columns:
        models = df['display_model'].unique()
    else:
        models = df['model'].unique()
    
    for model in models:
        count = len(df[df.get('display_model', df.get('model')) == model])
        logger.info(f"  {model}: {count} samples")
    
    # 2. Prepare data
    logger.info("\nPreparing data...")
    df_clean = prepare_data(df, filter_outliers=True, sample_size=None)
    logger.info(f"After preparation: {len(df_clean)} results")
    
    # Get summary statistics
    summary = get_model_summary(df_clean)
    logger.info("\nModel Summary:")
    print(summary)
    
    # 3. Apply plot styling
    apply_plot_style(style='clean')
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 4. Test Ridge Plot
    logger.info("\nCreating ridge plot...")
    try:
        fig = ridge_plot(
            df_clean,
            title='DAT Score Distributions',
            show_mean=True,
            figsize=(12, 8),
            overlap=0.0,            # no overlap in the test output
            left_margin=0.26        # a bit more room for long model names
        )
        
        if fig:
            fig.savefig(output_dir / 'test_ridge_plot.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {output_dir / 'test_ridge_plot.png'}")
            plt.close(fig)
    except Exception as e:
        logger.error(f"  Ridge plot failed: {e}")
    
    # 5. Test Statistical Heatmap
    logger.info("\nCreating statistical heatmap...")
    try:
        # Use only models with enough data
        model_counts = df_clean['display_model' if 'display_model' in df_clean.columns else 'model'].value_counts()
        top_models = model_counts.head(5).index.tolist()
        
        df_subset = df_clean[df_clean.get('display_model', df_clean.get('model')).isin(top_models)]
        
        fig = statistical_heatmap(
            df_subset,
            title='Model Comparison',
            figsize=(14, 6)
        )
        
        if fig:
            fig.savefig(output_dir / 'test_statistical_heatmap.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {output_dir / 'test_statistical_heatmap.png'}")
            plt.close(fig)
    except Exception as e:
        logger.error(f"  Statistical heatmap failed: {e}")
    
    # 6. Test Word Frequency Stacked Chart
    logger.info("\nCreating word frequency stacked chart...")
    try:
        # Create stacked bar chart with model attribution
        fig = word_frequency_stacked(
            df_clean,
            top_n=20,
            title='Top Words by Model Attribution',
            normalize="total",                # Shows relative frequency while keeping attribution
            sort_by="total",
            show_total_labels=True,
            annotate_models_using=True
        )
        
        if fig:
            fig.savefig(output_dir / 'test_word_frequency_stacked.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {output_dir / 'test_word_frequency_stacked.png'}")
            plt.close(fig)
            
        # Also create non-normalized version to show raw counts
        fig = word_frequency_stacked(
            df_clean,
            top_n=15,
            title='Top Words by Model (Raw Counts)',
            normalize=False,
            sort_by="total",
            show_total_labels=True,
            annotate_models_using=True
        )
        
        if fig:
            fig.savefig(output_dir / 'test_word_frequency_raw.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {output_dir / 'test_word_frequency_raw.png'}")
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"  Word frequency stacked chart failed: {e}")
    
    # 7. Test Triangular Matrix for Best Runs
    logger.info("\nCreating triangular matrix visualizations for best runs...")
    try:
        # Get best run for each model
        best_runs = get_best_runs(df_clean, top_n=1)
        
        if not best_runs.empty:
            # Initialize DAT scorer for distance calculations
            scorer = DATScorer()
            
            # Create triangular matrix for top models
            model_col = 'display_model' if 'display_model' in best_runs.columns else 'model'
            
            # Get top 3 models by score
            top_models = best_runs.nlargest(3, 'score')[model_col].unique()[:3]
            
            for model in top_models:
                model_run = best_runs[best_runs[model_col] == model].iloc[0]
                
                if 'words' in model_run and model_run['words']:
                    words = model_run['words']
                    score = model_run['score']
                    
                    # Create triangular matrix
                    fig = triangular_matrix(
                        words=words[:10],  # Use first 10 words
                        scorer=scorer,
                        title=f'{model} - Best Run (Score: {score:.2f})',
                        figsize=(8, 8),
                        show_values=True
                    )
                    
                    if fig:
                        safe_model_name = model.replace('/', '_').replace(':', '_')
                        fig.savefig(output_dir / f'test_triangular_matrix_{safe_model_name}.png',
                                   dpi=300, bbox_inches='tight')
                        logger.info(f"  Saved triangular matrix for {model}")
                        plt.close(fig)
        else:
            logger.warning("  No best runs found for triangular matrix")
            
    except Exception as e:
        logger.error(f"  Triangular matrix visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. Test filtering by strategy
    if 'strategy' in df_clean.columns:
        strategies = df_clean['strategy'].unique()
        logger.info(f"\nAvailable strategies: {strategies}")
        
        if 'random' in strategies:
            logger.info("\nCreating ridge plot for 'random' strategy...")
            try:
                df_random = df_clean[df_clean['strategy'] == 'random']
                
                fig = ridge_plot(
                    df_random,
                    title='DAT Scores - Random Strategy',
                    show_mean=True,
                    overlap=0.0,         # no overlap for clarity
                    left_margin=0.26
                )
                
                if fig:
                    fig.savefig(output_dir / 'test_ridge_random_strategy.png', 
                               dpi=300, bbox_inches='tight')
                    logger.info(f"  Saved strategy-specific plot")
                    plt.close(fig)
            except Exception as e:
                logger.error(f"  Strategy-specific plot failed: {e}")
    
    logger.info(f"\n✅ Visualization tests complete! Check the '{output_dir}' directory for outputs.")


if __name__ == "__main__":
    test_visualizations()