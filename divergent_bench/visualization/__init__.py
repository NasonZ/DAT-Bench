"""
Visualization module for divergent_bench.

Quick start:
    from divergent_bench.visualization import load_results, prepare_data, apply_theme
    from divergent_bench.visualization import ranked_dot_plot, distribution_plot

    apply_theme()
    df = prepare_data(load_results("results/"))
    fig = ranked_dot_plot(df)
"""

from .loader import (
    load_results,
    load_verifiers_results,
    prepare_data,
    get_model_summary,
    get_best_runs,
)
from .plots import (
    ranked_dot_plot,
    distribution_plot,
    significance_matrix,
    word_model_heatmap,
    triangular_matrix,
    # Backward-compatible aliases
    ridge_plot,
    statistical_heatmap,
    word_frequency_stacked,
)
from .styles import apply_theme, get_model_color, get_model_colors

# Legacy alias
apply_plot_style = apply_theme

__all__ = [
    # Data
    "load_results",
    "load_verifiers_results",
    "prepare_data",
    "get_model_summary",
    "get_best_runs",
    # Primary visualizations
    "ranked_dot_plot",
    "distribution_plot",
    "significance_matrix",
    "word_model_heatmap",
    "triangular_matrix",
    # Theme
    "apply_theme",
    "apply_plot_style",
    "get_model_color",
    "get_model_colors",
    # Deprecated aliases (still importable)
    "ridge_plot",
    "statistical_heatmap",
    "word_frequency_stacked",
]
