"""
Core visualization functions for divergent_bench.
Direct implementations based on DAT_GPT notebooks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import logging

from .styles import (
    get_model_colors, 
    PLOT_DEFAULTS,
    style_ridge_plot,
    style_heatmap
)

logger = logging.getLogger(__name__)


# NOTE: ridge plot improvements: tighter layout, axis-anchored labels, gentler overlap
def ridge_plot(df: pd.DataFrame, 
               score_col: str = 'score',
               model_col: str = None,
               order: Optional[List[str]] = None,
               colors: Optional[Dict[str, str]] = None,
               show_mean: bool = True,
               show_median: bool = False,
               xlim: Optional[Tuple[float, float]] = None,
               figsize: Tuple[int, int] = (12, 8),
               title: Optional[str] = None,
               overlap: float = -0.15,
               *,
               label_offset: float = -0.02,
               left_margin: float = 0.25,
               show_rug: bool = True,
               rug_max_n: int = 10) -> plt.Figure:
    """
    Create ridge plot (overlapping density plots) for model comparison.
    
    Ridge plots are ideal for:
    - Comparing score distributions across 3-8 models
    - Showing shape differences (multimodality, skewness) 
    - Identifying outliers and distribution tails
    - Small sample sizes (rug plot shows actual data points when n≤10)
    
    Statistical features:
    - Robust x-axis limits (1st-99th percentile) prevent outlier compression
    - Rug plots for n≤10 show actual observations
    - Sample size warnings for n<10 (red) and n<30 (orange)
    - KDE bandwidth adapts to sample size
    
    Args:
        df: DataFrame with scores
        score_col: Column name for scores
        model_col: Column name for models (auto-detected if None) 
        order: Order of models (top to bottom). If None, sorted by mean score
        colors: Dict mapping models to colors. If None, auto-generated
        show_mean: Whether to show mean lines
        show_median: Whether to show median lines
        xlim: Explicit x-axis limits. If None, uses robust percentile-based limits
        figsize: Figure size as (width, height)
        title: Plot title
        overlap: Vertical overlap between distributions (-1 to 0, higher = more overlap)
        label_offset: Y-axis label positioning offset
        left_margin: Left margin for model names
        show_rug: Show rug plot for small samples (n≤rug_max_n)
        rug_max_n: Maximum sample size to show rug plot
    
    Returns:
        Figure object or None if error
        
    Example:
        >>> fig = ridge_plot(df, overlap=-0.15, show_rug=True)  # Production view
        >>> fig = ridge_plot(df, overlap=0.0, show_rug=False)  # Separated view
    """
    # Auto-detect model column
    if model_col is None:
        if 'display_model' in df.columns:
            model_col = 'display_model'
        elif 'model' in df.columns:
            model_col = 'model'
        else:
            raise ValueError("No model column found")
    
    # Get model order
    if order is None:
        # Sort by mean score (DESC so best is at top of plot)
        order = df.groupby(model_col)[score_col].mean().sort_values(ascending=False).index.tolist()
    
    # Filter to models in order
    df_plot = df[df[model_col].isin(order)]
    
    # Get colors
    if colors is None:
        colors = get_model_colors(order)
    
    # Create FacetGrid
    g = sns.FacetGrid(
        df_plot, 
        row=model_col, 
        hue=model_col,
        aspect=PLOT_DEFAULTS['ridge_aspect'], 
        height=PLOT_DEFAULTS['ridge_height'],
        row_order=order, 
        palette=colors, 
        hue_order=order
    )
    
    # Remove redundant titles (Fix from VISUALIZATION_IMPROVEMENTS.md)
    g.set_titles("")
    
    # Make backgrounds transparent
    for ax in g.axes.flat:
        style_ridge_plot(ax, transparent=True)
    
    # Draw the densities
    g.map(sns.kdeplot, score_col, 
          bw_adjust=PLOT_DEFAULTS['kde_bandwidth'],
          clip_on=False,
          fill=PLOT_DEFAULTS['kde_fill'], 
          alpha=PLOT_DEFAULTS['kde_alpha'], 
          linewidth=1.5)
    
    # Add white outline for contrast
    g.map(sns.kdeplot, score_col, 
          clip_on=False, 
          color="white", 
          lw=2, 
          bw_adjust=PLOT_DEFAULTS['kde_bandwidth'])
    
    # Add reference line at y=0
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    
    # Add mean or median lines and statistical warnings
    model_counts = df_plot.groupby(model_col).size()  # noqa: F841
    for ax, model in zip(g.axes.flat, order):
        model_data = df_plot[df_plot[model_col] == model][score_col]
        n = len(model_data)
        
        if show_mean and n > 0:
            mean_val = model_data.mean()
            ax.axvline(mean_val, color='black', linestyle='--', 
                      alpha=0.7, ymin=0, ymax=0.5)
        
        if show_median and n > 0:
            median_val = model_data.median()
            ax.axvline(median_val, color='black', linestyle=':', 
                      alpha=0.7, ymin=0, ymax=0.5)
        
        # Add statistical context warnings (from VISUALIZATION_IMPROVEMENTS.md)
        if n < 10 and n > 0:
            ax.text(0.95, 0.85, '⚠️ n<10', transform=ax.transAxes,
                   color='red', fontweight='bold', fontsize=9,
                   ha='right', va='top', alpha=0.8)
            # Show rug plot for very small samples so raw data is visible
            if show_rug and n <= rug_max_n:
                sns.rugplot(x=model_data, ax=ax, height=0.06, lw=0.8,
                           color='black', alpha=0.45, clip_on=False)
        elif n < 30:
            ax.text(0.95, 0.85, '⚠️ Limited data', transform=ax.transAxes,
                   color='orange', fontsize=9,
                   ha='right', va='top', alpha=0.8)
            # Still show rug for small enough samples
            if show_rug and n <= rug_max_n:
                sns.rugplot(x=model_data, ax=ax, height=0.06, lw=0.8,
                           color='black', alpha=0.45, clip_on=False)
    
    # Overlap the subplots
    # Also add a bit more left margin so y-labels never sit over the plot.
    g.figure.subplots_adjust(hspace=overlap, left=left_margin)
    
    # Clean up axes
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    # Light x-grid for readability
    for ax in g.axes.flat:
        ax.grid(axis='x', alpha=0.10)
        ax.set_axisbelow(True)
    
    # Set x limits (robust default: 1st-99th percentile with small padding)
    if xlim:
        g.set(xlim=xlim)
    else:
        try:
            # Use robust percentiles to handle outliers gracefully
            lo, hi = np.nanpercentile(df_plot[score_col], [1, 99])
            pad = max(0.5, (hi - lo) * 0.05)  # 5% padding or minimum 0.5
            g.set(xlim=(lo - pad, hi + pad))
        except Exception:
            pass  # Fall back to matplotlib's auto-scaling
    
    # Add labels
    g.set_xlabels("Score", fontsize=PLOT_DEFAULTS['label_size'])
    
    # Add title
    if title:
        g.figure.suptitle(title, fontsize=PLOT_DEFAULTS['title_size'], y=0.99)
    
    # Add model names with sample sizes as y-labels (Fix from VISUALIZATION_IMPROVEMENTS.md)
    model_counts = df_plot.groupby(model_col).size()
    for ax, model in zip(g.axes.flat, order):
        count = model_counts.get(model, 0)
        label = f"{model} (n={count})"
        
        # Add warning emoji for small samples
        if count < 10:
            label += " ⚠️"
        
        # Add y-axis label using text annotation anchored to the axis
        # This works better with FacetGrid than set_ylabel
        ax.text(label_offset, 0.5, label, 
                transform=ax.transAxes,
                fontsize=PLOT_DEFAULTS['label_size'], 
                ha='right', 
                va='center',
                rotation=0)
    
    return g.figure


def statistical_heatmap(df: pd.DataFrame,
                       score_col: str = 'score',
                       model_col: str = None,
                       order: Optional[List[str]] = None,
                       colors: Optional[Dict[str, str]] = None,
                       bar_xlim: Optional[Tuple[float, float]] = None,
                       figsize: Tuple[int, int] = (14, 8),
                       title: Optional[str] = None,
                       p_correction: Optional[str] = "holm",
                       heat_metric: str = "t") -> plt.Figure:
    """
    Create dual-panel visualization: bar plot + statistical significance matrix.
    
    Statistical heatmaps are ideal for:
    - Pairwise model comparisons with proper multiple testing correction
    - Showing both effect sizes and statistical significance
    - Identifying clusters of similar/different models
    - Publication-ready statistical summaries
    
    Statistical features:
    - Welch's t-test (robust to unequal variances)
    - Multiple comparison correction (Holm step-down by default)
    - Choice of heat metric: t-values or Cohen's d effect sizes
    - Clear significance indicators (*, **, ***)
    
    Args:
        df: DataFrame with scores
        score_col: Column name for scores
        model_col: Column name for models
        order: Order of models. If None, sorted by mean score
        colors: Dict mapping models to colors
        bar_xlim: X-axis limits for bar plot
        figsize: Figure size
        title: Overall title
        p_correction: Multiple comparison correction method:
                      "holm" (default) - Holm step-down, controls FWER
                      "bonferroni" - Bonferroni, controls FWER (more conservative)
                      "bh" or "fdr" - Benjamini-Hochberg, controls FDR
                      None - No correction (not recommended)
        heat_metric: What to show in heatmap colors:
                     "t" (default) - t-statistics 
                     "d" - Cohen's d effect sizes (more interpretable)
    
    Returns:
        Matplotlib figure with two subplots: performance bars + significance matrix
        
    Example:
        >>> # Standard view with effect sizes
        >>> fig = statistical_heatmap(df, heat_metric="d", p_correction="holm")
        >>> # Diagnostic view with t-values
        >>> fig = statistical_heatmap(df, heat_metric="t", p_correction=None)
    """
    # Auto-detect model column
    if model_col is None:
        if 'display_model' in df.columns:
            model_col = 'display_model'
        elif 'model' in df.columns:
            model_col = 'model'
        else:
            raise ValueError("No model column found")
    
    # Calculate statistics
    stats_df = df.groupby(model_col)[score_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('sem', lambda x: x.std() / np.sqrt(len(x))),
        ('count', 'count')
    ]).reset_index()
    
    # Get model order
    if order is None:
        order = stats_df.sort_values('mean')[model_col].tolist()
    else:
        # Filter to models that exist
        order = [m for m in order if m in stats_df[model_col].values]
    
    stats_df = stats_df[stats_df[model_col].isin(order)]
    
    # Get colors
    if colors is None:
        colors = get_model_colors(order)
    
    # Calculate pairwise t-tests and effect sizes
    n_models = len(order)
    t_values = np.zeros((n_models, n_models))
    p_values = np.ones((n_models, n_models))
    effect_sizes = np.zeros((n_models, n_models))  # Cohen's d
    
    for i, model1 in enumerate(order):
        for j, model2 in enumerate(order):
            if i != j:
                scores1 = df[df[model_col] == model1][score_col].dropna()
                scores2 = df[df[model_col] == model2][score_col].dropna()
                
                if len(scores1) > 1 and len(scores2) > 1:
                    # Welch's t-test (unequal variances)
                    t_stat, p_val = stats.ttest_ind(scores1, scores2, equal_var=False)
                    t_values[i, j] = t_stat
                    p_values[i, j] = p_val
                    
                    # Calculate Cohen's d effect size
                    pooled_std = np.sqrt((scores1.std()**2 + scores2.std()**2) / 2)
                    if pooled_std > 0:
                        effect_sizes[i, j] = (scores1.mean() - scores2.mean()) / pooled_std
    
    # Apply multiple comparison correction
    if p_correction:
        if p_correction.lower() in {"holm", "holm-bonferroni"}:
            # Holm step-down correction (controls FWER)
            pairs = []
            for i in range(n_models):
                for j in range(i+1, n_models):
                    pairs.append(((i, j), p_values[i, j]))
            
            if pairs:
                m = len(pairs)  # Number of comparisons
                # Sort by p-value
                order_idx = sorted(range(m), key=lambda k: pairs[k][1])
                adj_p = [None] * m
                prev = 0.0
                
                for rank, idx in enumerate(order_idx):
                    _, p = pairs[idx]
                    # Holm correction: multiply by decreasing factor
                    val = min(1.0, (m - rank) * p)
                    # Enforce monotonicity
                    val = max(prev, val)
                    adj_p[idx] = val
                    prev = val
                
                # Write back adjusted p-values (both triangles for convenience)
                for idx, ((i, j), _) in enumerate(pairs):
                    p_values[i, j] = adj_p[idx]
                    p_values[j, i] = adj_p[idx]
                    
        elif p_correction.lower() == "bonferroni":
            # Simple Bonferroni correction
            m = n_models * (n_models - 1) / 2
            p_values = np.minimum(p_values * m, 1.0)
            
        elif p_correction.lower() in {"bh", "fdr", "benjamini-hochberg"}:
            # Benjamini-Hochberg FDR correction
            pairs = []
            for i in range(n_models):
                for j in range(i+1, n_models):
                    pairs.append(((i, j), p_values[i, j]))
                    
            if pairs:
                m = len(pairs)
                # Sort by p-value
                order_idx = sorted(range(m), key=lambda k: pairs[k][1])
                adj_p = [None] * m
                prev = 1.0
                
                for rank_rev, idx in enumerate(reversed(order_idx)):
                    rank = m - rank_rev
                    _, p = pairs[idx]
                    # BH correction
                    val = min(1.0, p * m / rank)
                    # Enforce monotonicity (non-increasing from right)
                    val = min(prev, val)
                    adj_p[idx] = val
                    prev = val
                
                # Write back adjusted p-values
                for idx, ((i, j), _) in enumerate(pairs):
                    p_values[i, j] = adj_p[idx]
                    p_values[j, i] = adj_p[idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Left panel: Horizontal bar plot
    y_positions = range(len(order))
    
    # Get stats in order
    means = [stats_df[stats_df[model_col] == m]['mean'].values[0] for m in order]
    sems = [stats_df[stats_df[model_col] == m]['sem'].values[0] for m in order]
    bar_colors = [colors.get(m, '#888888') for m in order]
    
    # Create bars
    bars = ax1.barh(y_positions, means, 
                    color=bar_colors, 
                    alpha=PLOT_DEFAULTS['bar_alpha'])
    
    # Add error bars
    ax1.errorbar(means, y_positions,
                xerr=sems, 
                fmt='none', 
                color='black', 
                capsize=PLOT_DEFAULTS['error_capsize'],
                linewidth=PLOT_DEFAULTS['error_linewidth'])
    
    # Customize bar plot with sample sizes (from VISUALIZATION_IMPROVEMENTS.md)
    ax1.set_yticks(y_positions)
    # Add sample sizes to labels
    counts = [stats_df[stats_df[model_col] == m]['count'].values[0] for m in order]
    labels_with_n = [f'{model} (n={int(count)})' for model, count in zip(order, counts)]
    ax1.set_yticklabels(labels_with_n)
    ax1.set_xlabel('Mean Score', fontsize=PLOT_DEFAULTS['label_size'])
    ax1.set_title('Performance', fontsize=PLOT_DEFAULTS['title_size'])
    
    if bar_xlim:
        ax1.set_xlim(bar_xlim)
    
    # Add value labels on bars (positioned beyond error bars to avoid overlap)
    for i, (mean, sem) in enumerate(zip(means, sems)):
        # Position text to the right of the error bar
        text_x = mean + sem + 1.5  # Add offset beyond the error bar
        ax1.text(text_x, i, f'{mean:.1f}', 
                va='center', fontsize=9)
    
    # Right panel: Statistical significance heatmap
    # Create masks for upper triangle
    mask = np.triu(np.ones_like(t_values, dtype=bool))
    
    # Create significance annotations with effect sizes (from VISUALIZATION_IMPROVEMENTS.md)
    annot = np.full_like(p_values, '', dtype=object)
    for i in range(n_models):
        for j in range(n_models):
            if not mask[i, j] and i != j:
                # Add effect size (Cohen's d)
                d = effect_sizes[i, j]
                if abs(d) > 0.01:  # Only show if meaningful
                    annot[i, j] = f'd={d:.2f}\n'
                else:
                    annot[i, j] = ''
                    
                # Add significance stars
                if p_values[i, j] < 0.001:
                    annot[i, j] += '***'
                elif p_values[i, j] < 0.01:
                    annot[i, j] += '**'
                elif p_values[i, j] < 0.05:
                    annot[i, j] += '*'
    
    # Choose heat metric (t-values or Cohen's d)
    if heat_metric.lower() == "d":
        heat_data = effect_sizes
        cbar_label = "Cohen's d"
    else:
        heat_data = t_values
        cbar_label = "t-values"
    
    # Plot heatmap
    vmax = np.abs(heat_data).max() if np.abs(heat_data).max() > 0 else 1
    
    sns.heatmap(heat_data, 
                mask=mask, 
                annot=annot, 
                fmt='',
                cmap=PLOT_DEFAULTS['heatmap_cmap'], 
                center=PLOT_DEFAULTS['heatmap_center'],
                vmin=-vmax, 
                vmax=vmax,
                xticklabels=order, 
                yticklabels=order,
                cbar_kws={'label': cbar_label},
                linewidths=PLOT_DEFAULTS['heatmap_linewidth'], 
                ax=ax2)
    
    style_heatmap(ax2, 'Statistical Significance')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=PLOT_DEFAULTS['title_size'], y=1.02)
    
    plt.tight_layout()
    
    return fig



def word_frequency_stacked(df: pd.DataFrame,
                           words_col: str = 'words',
                           model_col: str = None,
                           top_n: int = 20,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8),
                           normalize: Union[bool, str] = "per_word",
                           sort_by: str = "total",
                           show_total_labels: bool = True,
                           annotate_models_using: bool = True) -> plt.Figure:
    """
    Create stacked bar chart showing word frequency with model attribution.
    
    Word frequency charts are ideal for:
    - Analyzing which models contribute to popular responses
    - Identifying convergent vs divergent thinking patterns
    - Detecting model-specific vocabulary preferences
    - Understanding response diversity across models
    
    Normalization modes:
    - False/None: Raw counts - shows absolute frequency
    - "per_word": 100% stacked - shows model share per word (recommended)
    - "total": % of all mentions - preserves relative frequency between words
    - "within_model": Legacy per-model normalization (can be misleading)
    
    Visual interpretation:
    - Full rainbow stack = all models converged on this word
    - Single color = unique to that model  
    - Bar length = popularity (in "total" mode) or equal (in "per_word" mode)
    - Y-axis badges [n/m] = n models of m total used this word
    
    Args:
        df: DataFrame with words and model columns
        words_col: Column name containing word lists
        model_col: Column name for models (auto-detected if None)
        top_n: Number of top words to show
        title: Chart title
        figsize: Figure size
        normalize: Normalization mode:
            - False/None: Raw counts
            - "per_word": 100% stacked per word (shows model attribution)  
            - "total": % of total mentions (shows relative frequency)
            - True/"within_model": Legacy per-model normalization
        sort_by: "total" (default), "alphabetic", or "max_model"
        show_total_labels: Add total count labels at end of bars
        annotate_models_using: Add [n/m] badges to y-axis labels
    
    Returns:
        Figure object or None if error
        
    Example:
        >>> # Standard view showing model attribution
        >>> fig = word_frequency_stacked(df, normalize="per_word", top_n=20)
        >>> # Raw counts view  
        >>> fig = word_frequency_stacked(df, normalize=False, top_n=15)
        >>> # Relative frequency with attribution
        >>> fig = word_frequency_stacked(df, normalize="total", top_n=20)
    """
    # Auto-detect model column
    if model_col is None:
        if 'display_model' in df.columns:
            model_col = 'display_model'
        elif 'model' in df.columns:
            model_col = 'model'
        else:
            raise ValueError("No model column found")
    
    # Build word-model matrix
    word_model_counts = {}
    model_totals = {}
    
    for _, row in df.iterrows():
        model = row[model_col]
        words = row[words_col]
        
        if model not in model_totals:
            model_totals[model] = 0
        model_totals[model] += 1
        
        if isinstance(words, list):
            word_list = [w.lower() for w in words if isinstance(w, str)]
        elif isinstance(words, str):
            try:
                import ast
                parsed = ast.literal_eval(words)
                word_list = [w.lower() for w in parsed if isinstance(w, str)]
            except:
                continue
        else:
            continue
            
        for word in word_list:
            if word not in word_model_counts:
                word_model_counts[word] = {}
            if model not in word_model_counts[word]:
                word_model_counts[word][model] = 0
            word_model_counts[word][model] += 1
    
    # Totals per word and optional sorting
    word_totals = {word: sum(counts.values()) for word, counts in word_model_counts.items()}

    if sort_by == "alphabetic":
        sorted_words = sorted(word_totals.keys())
    elif sort_by == "max_model":
        # sort by the max count contributed by any single model
        def max_contrib(w):
            c = word_model_counts.get(w, {})
            return max(c.values()) if c else 0
        sorted_words = sorted(word_totals.keys(), key=lambda w: max_contrib(w), reverse=True)
    else:  # "total"
        sorted_words = [w for w, _ in sorted(word_totals.items(), key=lambda x: x[1], reverse=True)]

    top_words = [(w, word_totals[w]) for w in sorted_words[:top_n]]
    
    if not top_words:
        logger.warning("No words found for stacked frequency plot")
        return None
    
    # Prepare data for stacking
    words = [w for w, _ in top_words]
    models = sorted(model_totals.keys())
    
    # Get colors for models
    colors = get_model_colors(models)
    
    # Resolve normalization mode (back-compat with bool)
    if isinstance(normalize, bool):
        norm_mode = "within_model" if normalize else None
    else:
        norm_mode = normalize  # None / "per_word" / "within_model" / "total"

    # Calculate total mentions for "total" normalization
    if norm_mode == "total":
        grand_total = sum(sum(counts.values()) for counts in word_model_counts.values())
    
    # Build matrix for plotting
    data_matrix = []
    for model in models:
        model_data = []
        for word in words:
            count = word_model_counts.get(word, {}).get(model, 0)
            if norm_mode == "within_model" and model_totals[model] > 0:
                # Legacy behavior: normalize within each model
                value = (count / model_totals[model]) * 100
            elif norm_mode == "per_word":
                denom = sum(word_model_counts.get(word, {}).values())
                value = (count / denom * 100) if denom > 0 else 0
            elif norm_mode == "total" and grand_total > 0:
                # Normalize by total mentions across all words
                value = (count / grand_total * 100)
            else:
                value = count  # raw counts
            model_data.append(value)
        data_matrix.append(model_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create stacked bars
    y_positions = np.arange(len(words))
    left = np.zeros(len(words))
    
    for i, (model, data) in enumerate(zip(models, data_matrix)):
        bars = ax.barh(y_positions, data, left=left,
                      label=f'{model} (n={model_totals[model]})',
                      color=colors[model],
                      alpha=0.88, edgecolor="white", linewidth=0.6)
        left += data

    # Compose y tick labels once (no floating text on canvas)
    if annotate_models_using:
        models_using_by_word = {
            w: sum(1 for m in models if word_model_counts.get(w, {}).get(m, 0) > 0)
            for w in words
        }
        ytick_labels = [
            f"{w}  [{models_using_by_word[w]}/{len(models)}]" if models_using_by_word[w] > 1 else w
            for w in words
        ]
    else:
        ytick_labels = words
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ytick_labels, fontsize=PLOT_DEFAULTS['label_size'])
    
    # Axis labels by mode
    if norm_mode == "within_model":
        xlabel = 'Mentions per 100 samples of each model (can sum >100 across models)'
    elif norm_mode == "per_word":
        xlabel = 'Share of mentions per word (%, sums to 100% across models)'
    elif norm_mode == "total":
        xlabel = 'Percentage of total word mentions (%)'
    else:
        xlabel = 'Word Count'
    ax.set_xlabel(xlabel, fontsize=PLOT_DEFAULTS['label_size'])

    # Optional total labels at end of each bar (always raw totals)
    if show_total_labels:
        for yi, w in enumerate(words):
            total = word_totals.get(w, 0)
            # right edge of the stack for this row
            right_edge = left[yi] if norm_mode else sum(word_model_counts.get(w, {}).values())
            # small padding
            pad = 1.0 if norm_mode else 0.15
            ax.text(right_edge + pad, yi, f'{total}',
                    va='center', ha='left', fontsize=9, color='dimgray')

    # Helpful grid + axisbelow for readability
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    
    # Title with context
    if not title:
        title = f'Top {top_n} Words - Model Attribution'
    
    # Add subtitle with insights
    total_samples = len(df)
    subtitle = f'n={total_samples} total samples | {len(models)} models | '
    
    # Check convergence
    high_convergence_words = [w for w in words 
                             if len(word_model_counts.get(w, {})) >= len(models)*0.7]
    if high_convergence_words:
        subtitle += f'{len(high_convergence_words)} converged words'
    else:
        subtitle += 'Low vocabulary overlap'
    
    ax.set_title(title, fontsize=PLOT_DEFAULTS['title_size'], pad=18)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
           fontsize=10, ha='center', color='gray')
    
    # Add legend with sample sizes
    ax.legend(loc='lower right',
              frameon=False,
              ncol=(1 if len(models) <= 5 else 2),
              fontsize=PLOT_DEFAULTS['legend_size'],
              title="Models")
    
    # Tight x limits with a bit of right padding so labels never clip
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.05)
    
    # Add vertical line at 0 for reference
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    return fig


def score_distribution(df: pd.DataFrame,
                       score_col: str = 'score',
                       model_col: str = None,
                       models: Optional[List[str]] = None,
                       bins: int = 30,
                       figsize: Tuple[int, int] = (10, 6),
                       title: Optional[str] = None) -> plt.Figure:
    """
    Simple histogram of score distributions.
    
    Args:
        df: DataFrame with scores
        score_col: Column name for scores
        model_col: Column name for models
        models: List of models to include (None for all)
        bins: Number of histogram bins
        figsize: Figure size
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    # Auto-detect model column
    if model_col is None:
        if 'display_model' in df.columns:
            model_col = 'display_model'
        elif 'model' in df.columns:
            model_col = 'model'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if models and model_col:
        # Plot specific models
        colors_dict = get_model_colors(models)
        for model in models:
            model_data = df[df[model_col] == model][score_col].dropna()
            if len(model_data) > 0:
                ax.hist(model_data, bins=bins, alpha=0.5, 
                       label=f'{model} (n={len(model_data)})',
                       color=colors_dict.get(model))
        ax.legend()
    else:
        # Plot all data
        scores = df[score_col].dropna()
        ax.hist(scores, bins=bins, alpha=0.7, color='#2E86AB')
        ax.set_ylabel('Count')
    
    ax.set_xlabel('Score', fontsize=PLOT_DEFAULTS['label_size'])
    
    if title:
        ax.set_title(title, fontsize=PLOT_DEFAULTS['title_size'])
    else:
        ax.set_title('Score Distribution', fontsize=PLOT_DEFAULTS['title_size'])
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def triangular_matrix(words: List[str],
                     distances: Optional[List[float]] = None,
                     scorer: Optional['DATScorer'] = None,
                     title: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 10),
                     cmap: str = 'Greens',
                     show_values: bool = True,
                     value_format: str = '{:.0f}',
                     fontsize_values: int = 11,
                     fontsize_labels: int = 12) -> plt.Figure:
    """
    Create triangular matrix visualization of pairwise semantic distances.
    
    This visualization mimics the DAT website's triangular display showing
    all pairwise distances between words in a clean, readable format.
    
    Triangular matrices are ideal for:
    - Showing all pairwise relationships in a compact format
    - Understanding which word pairs contribute most/least to the score
    - Identifying semantic clusters or outliers
    - Replicating the classic DAT visualization format
    
    Args:
        words: List of words to display
        distances: Pre-computed pairwise distances (optional)
                  If None, will compute using scorer
        scorer: DATScorer instance for computing distances (required if distances=None)
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap for the matrix cells
        show_values: Whether to display distance values in cells
        value_format: Format string for distance values
        fontsize_values: Font size for distance values
        fontsize_labels: Font size for word labels
    
    Returns:
        Matplotlib figure with triangular matrix
        
    Example:
        >>> # From a result with words and pre-computed distances
        >>> fig = triangular_matrix(result['words'], result['distances'])
        >>> # Or compute distances on the fly
        >>> scorer = DATScorer()
        >>> fig = triangular_matrix(words, scorer=scorer)
    """
    import itertools
    
    n_words = len(words)
    
    # Compute distances if not provided
    if distances is None:
        if scorer is None:
            raise ValueError("Either distances or scorer must be provided")
        
        # Validate and clean words
        valid_words = []
        for word in words:
            valid = scorer.validate(word)
            if valid:
                valid_words.append(valid)
        
        # Compute pairwise distances
        distances = []
        for word1, word2 in itertools.combinations(valid_words, 2):
            dist = scorer.distance(word1, word2) * 100  # Convert to 0-200 scale
            distances.append(dist)
    
    # Create distance matrix (lower triangular)
    dist_matrix = np.zeros((n_words, n_words))
    dist_idx = 0
    
    for i in range(n_words):
        for j in range(i):
            if dist_idx < len(distances):
                dist_matrix[i, j] = distances[dist_idx]
                dist_idx += 1
    
    # Create figure with light background like the original
    fig, ax = plt.subplots(figsize=figsize, facecolor='#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Create the triangular visualization
    # We'll use a custom approach to match the DAT website style
    
    # Clear the axes
    ax.clear()
    ax.set_facecolor('#f8f9fa')
    ax.set_xlim(-1.2, n_words - 0.5)
    ax.set_ylim(-1.5, n_words)  # Reduce top whitespace
    ax.set_aspect('equal')
    
    # Remove all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add word labels on the left (vertical)
    # Align with the CENTER of each row
    for i, word in enumerate(words[1:], 1):  # Skip first word (no row for it)
        y_pos = n_words - i - 1  # Exact center of the row where cells are positioned
        ax.text(-0.7, y_pos, word, 
                ha='right', va='center', fontsize=fontsize_labels,
                fontweight='normal')
    
    # Add word labels on the bottom (horizontal) - BELOW the matrix
    for j, word in enumerate(words[:-1]):  # Skip last word (no column for it)
        # Position below the bottom row of the matrix
        ax.text(j, -0.7, word,
                ha='center', va='top', fontsize=fontsize_labels,
                fontweight='normal', rotation=0)
    
    # Normalize distances for coloring
    if len(distances) > 0:
        vmin, vmax = min(distances), max(distances)
    else:
        vmin, vmax = 0, 200
    
    # Draw the triangular matrix cells
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import get_cmap
    
    # Create custom colormap matching the original's subtle green shading
    # Light green for low values, darker green for high values
    colors = ['#f0f8f0', '#c8e6c9', '#81c784', '#4caf50', '#388e3c', '#2e7d32']
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('custom_green', colors, N=n_bins)
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = custom_cmap if cmap == 'Greens' else get_cmap(cmap)
    
    # Draw cells and values
    dist_idx = 0
    for i in range(1, n_words):  # Rows (starting from second word)
        for j in range(i):  # Columns (up to diagonal)
            if dist_idx < len(distances):
                dist_val = distances[dist_idx]
                
                # Calculate position (flip y-axis for top-to-bottom)
                x = j
                y = n_words - i - 1
                
                # Draw cell background with no border, like the original
                # Use a lighter green base with darker for higher values
                # Normalize so higher distances get darker green (better)
                color = colormap(norm(dist_val))
                rect = Rectangle((x - 0.5, y - 0.5), 1.0, 1.0,
                               facecolor=color, edgecolor='none', linewidth=0)
                ax.add_patch(rect)
                
                # Add distance value - always black text
                if show_values:
                    ax.text(x, y, value_format.format(dist_val),
                           ha='center', va='center', fontsize=fontsize_values,
                           color='black',
                           fontweight='normal')
                
                dist_idx += 1
    
    # Add title at the bottom  
    if title:
        ax.text(n_words / 2 - 0.5, -1.2, title,
               ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    return fig