# Visualization Module

Simple, flexible visualization tools for divergent_bench results.

## Quick Start

```python
from divergent_bench.visualization import (
    load_results,
    prepare_data,
    ridge_plot,
    statistical_heatmap,
    word_frequency_chart,
    apply_plot_style
)

# Load results from JSON files
df = load_results("results/")

# Prepare data (outlier removal, sampling)
df = prepare_data(df)

# Apply consistent styling
apply_plot_style()

# Create visualizations
fig = ridge_plot(df, title="DAT Score Distributions")
fig.savefig("dat_scores.png", dpi=300, bbox_inches='tight')
```

## Core Functions

### Data Loading
- `load_results(path)` - Load JSON results from divergent_bench
- `prepare_data(df)` - Clean and prepare data for visualization
- `get_model_summary(df)` - Get summary statistics per model

### Visualizations
- `ridge_plot()` - Overlapping density plots (primary visualization)
- `statistical_heatmap()` - Dual-panel bar chart + significance matrix
- `word_frequency_chart()` - Most common words analysis
- `score_distribution()` - Simple histogram of scores

### Styling
- `apply_plot_style()` - Apply consistent plot styling
- `get_model_color()` - Get consistent colors for models

## Design Philosophy

1. **Simple** - Just functions that make plots, no complex abstractions
2. **Flexible** - Works with whatever model names and data you have
3. **Direct** - Based directly on proven DAT_GPT notebook code
4. **Pragmatic** - ~500 lines total, easy to understand and modify

## Examples

### Filter by Strategy
```python
df_random = df[df['strategy'] == 'random']
fig = ridge_plot(df_random, title="Random Strategy Results")
```

### Compare Specific Models
```python
models = ['gpt-5-nano', 'llama3.2:3b']
df_subset = df[df['display_model'].isin(models)]
fig = statistical_heatmap(df_subset)
```

### Model-Specific Word Analysis
```python
fig = word_frequency_chart(
    df,
    model_filter='gpt-5-nano',
    top_n=20
)
```

## File Structure

- `loader.py` - Load and transform JSON results (150 lines)
- `plots.py` - Core visualization functions (400 lines)
- `styles.py` - Colors and styling (150 lines)

Total: ~700 lines of straightforward code.

## Notes

- Handles malformed JSON files gracefully
- Auto-detects model names and strategies
- Flexible color system with fallbacks
- Works with partial data (missing columns handled)
- No rigid model name mappings required