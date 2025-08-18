"""
Flexible styling and colors for visualizations.
Provides sensible defaults while allowing customization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import hashlib


# Default color palette for models
# Uses a curated set of distinct colors
DEFAULT_COLORS = [
    '#2E86AB',  # Blue
    '#A23B72',  # Purple
    '#F18F01',  # Orange
    '#C73E1D',  # Red
    '#6A994E',  # Green
    '#BC4B51',  # Rose
    '#5D576B',  # Gray purple
    '#F4D35E',  # Yellow
    '#264653',  # Dark teal
    '#E76F51',  # Coral
]

# Model-specific colors (optional overrides)
# These are from the DAT notebook for known models
MODEL_SPECIFIC_COLORS = {
    # OpenAI
    'gpt-4': '#008080',
    'gpt-5': '#008080',
    'gpt-5-nano': '#00A3A3',
    'gpt-3.5-turbo': '#006400',
    'gpt-3.5': '#006400',
    
    # Anthropic
    'claude': '#FF8C00',
    'claude-3': '#FF8C00',
    'claude-3-opus': '#FF8C00',
    'claude-3.5-sonnet': '#FFA500',
    
    # Google
    'gemini': '#1E90FF',
    'gemini-pro': '#1E90FF',
    'gemini-1.5-pro': '#1E90FF',
    
    # Meta/Ollama
    'llama': '#6A0DAD',
    'llama3': '#6A0DAD',
    'llama3.2': '#8B008B',
    'llama3.2:3b': '#8B008B',
    
    # Human baseline
    'human': '#4D4D4D',
}


def get_model_color(model_name: str, 
                    color_index: Optional[int] = None) -> str:
    """
    Get color for a model with flexible fallback.
    
    Args:
        model_name: Name of the model
        color_index: Optional index for color palette
    
    Returns:
        Hex color string
    """
    if not model_name:
        return '#888888'
    
    model_lower = model_name.lower()
    
    # Check for exact match first
    if model_lower in MODEL_SPECIFIC_COLORS:
        return MODEL_SPECIFIC_COLORS[model_lower]
    
    # Check for partial matches (e.g., "gpt-5-nano" matches "gpt-5")
    for key, color in MODEL_SPECIFIC_COLORS.items():
        if key in model_lower:
            return color
    
    # Use color index if provided
    if color_index is not None:
        return DEFAULT_COLORS[color_index % len(DEFAULT_COLORS)]
    
    # Generate consistent color from model name
    # This ensures the same model always gets the same color
    hash_val = int(hashlib.md5(model_name.encode()).hexdigest()[:8], 16)
    return DEFAULT_COLORS[hash_val % len(DEFAULT_COLORS)]


def get_model_colors(model_names: List[str], 
                     use_specific: bool = True) -> Dict[str, str]:
    """
    Get colors for a list of models.
    
    Args:
        model_names: List of model names
        use_specific: Whether to use model-specific colors
    
    Returns:
        Dictionary mapping model names to colors
    """
    colors = {}
    
    for i, model in enumerate(model_names):
        if use_specific:
            colors[model] = get_model_color(model)
        else:
            # Use palette order for consistent coloring
            colors[model] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    
    return colors


# Plot style settings
PLOT_DEFAULTS = {
    'font_size': 12,
    'title_size': 16,
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 10,
    'dpi': 100,
    'save_dpi': 300,
    
    # Figure settings
    'figure_facecolor': 'white',
    'axes_facecolor': 'white',
    'grid': False,
    
    # Ridge plot specific (updated per VISUALIZATION_IMPROVEMENTS.md)
    'ridge_height': 1.5,  # Increased from 1 for better readability
    'ridge_aspect': 4,    # Changed from 9 for better scaling with many models
    'ridge_overlap': -0.3,  # Less overlap for readability
    'kde_bandwidth': 1,
    'kde_alpha': 0.7,
    'kde_fill': True,
    
    # Heatmap specific
    'heatmap_cmap': 'coolwarm',
    'heatmap_center': 0,
    'heatmap_linewidth': 0.5,
    
    # Bar plot specific
    'bar_alpha': 0.8,
    'error_capsize': 5,
    'error_linewidth': 1.5,
}


def apply_plot_style(style: str = 'clean', 
                     font_scale: float = 1.0,
                     rc_params: Optional[Dict] = None):
    """
    Apply consistent plot styling.
    
    Args:
        style: Style preset ('clean', 'paper', 'notebook')
        font_scale: Scale factor for all fonts
        rc_params: Additional matplotlib rcParams
    """
    # Set seaborn style
    if style == 'clean':
        sns.set_style('whitegrid')
    elif style == 'paper':
        sns.set_style('white')
    elif style == 'notebook':
        sns.set_style('notebook')
    else:
        sns.set_style(style)
    
    # Apply font scaling
    scaled_defaults = {
        'font.size': PLOT_DEFAULTS['font_size'] * font_scale,
        'axes.titlesize': PLOT_DEFAULTS['title_size'] * font_scale,
        'axes.labelsize': PLOT_DEFAULTS['label_size'] * font_scale,
        'xtick.labelsize': PLOT_DEFAULTS['tick_size'] * font_scale,
        'ytick.labelsize': PLOT_DEFAULTS['tick_size'] * font_scale,
        'legend.fontsize': PLOT_DEFAULTS['legend_size'] * font_scale,
        'figure.dpi': PLOT_DEFAULTS['dpi'],
        'savefig.dpi': PLOT_DEFAULTS['save_dpi'],
        'figure.facecolor': PLOT_DEFAULTS['figure_facecolor'],
        'axes.facecolor': PLOT_DEFAULTS['axes_facecolor'],
        'axes.grid': PLOT_DEFAULTS['grid'],
    }
    
    plt.rcParams.update(scaled_defaults)
    
    # Apply any additional custom params
    if rc_params:
        plt.rcParams.update(rc_params)


def get_colormap(n_colors: int, cmap_name: str = 'Set2') -> List[str]:
    """
    Get a list of colors from a matplotlib colormap.
    
    Args:
        n_colors: Number of colors needed
        cmap_name: Name of matplotlib colormap
    
    Returns:
        List of hex color strings
    """
    cmap = plt.cm.get_cmap(cmap_name)
    colors = [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
    return ['#%02x%02x%02x' % tuple(int(c * 255) for c in color[:3]) for color in colors]


def style_ridge_plot(ax, transparent: bool = True):
    """
    Apply specific styling for ridge plots.
    
    Args:
        ax: Matplotlib axes object
        transparent: Whether to make background transparent
    """
    if transparent:
        ax.set_facecolor('none')
    
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def style_heatmap(ax, title: Optional[str] = None):
    """
    Apply specific styling for heatmaps.
    
    Args:
        ax: Matplotlib axes object
        title: Optional title
    """
    if title:
        ax.set_title(title, fontsize=PLOT_DEFAULTS['title_size'], pad=10)
    
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)