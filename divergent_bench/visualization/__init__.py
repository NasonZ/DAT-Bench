"""
Visualization module for divergent_bench.
Simple, direct plotting functions based on DAT_GPT notebooks.
"""

from .loader import load_results, prepare_data, get_model_summary, get_best_runs
from .plots import ridge_plot, statistical_heatmap, word_frequency_stacked, triangular_matrix
from .styles import apply_plot_style, get_model_color

__all__ = [
    # Data loading
    'load_results',
    'prepare_data',
    'get_model_summary',
    'get_best_runs',
    
    # Visualizations
    'ridge_plot',
    'statistical_heatmap', 
    'word_frequency_stacked',
    'triangular_matrix',
    
    # Styling
    'apply_plot_style',
    'get_model_color',
]