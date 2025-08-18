# Technical Notes - Divergent Bench

This document consolidates key technical decisions, architectural patterns, and implementation details for ongoing reference.

## Table of Contents
- [Visualization Standards](#visualization-standards)
- [Statistical Methods](#statistical-methods)
- [Architecture Decisions](#architecture-decisions)
- [Model Detection Patterns](#model-detection-patterns)

---

## Visualization Standards

### Production Requirements
- **Statistical Context**: All visualizations must include sample size warnings (n<10 red, n<30 orange)
- **Small Sample Transparency**: Show actual data points via rug plots when n≤10
- **Multiple Comparison Correction**: Use Holm step-down by default for pairwise comparisons
- **Effect Sizes**: Prefer Cohen's d over t-statistics for interpretability

### Color Semantics
- **Model Families**: Consistent colors across all plots (purple for llama, teal for GPT variants)
- **Statistical Significance**: Red-blue diverging for effect sizes, centered at zero
- **Warnings**: Red for critical (n<10), orange for caution (n<30)

### Typography & Layout
- **Font Sizes**: Title 14pt, labels 11pt, annotations 9pt
- **Aspect Ratios**: Ridge plots 4:1 (not 9:1), heatmaps square
- **Margins**: Left margin 0.25+ for long model names
- **DPI**: Always 300 for publication quality

---

## Statistical Methods

### Multiple Comparison Correction

**Holm Step-Down Procedure** (Default)
```python
# For m comparisons with p-values p₁ ≤ p₂ ≤ ... ≤ pₘ
for i in range(m):
    p_adjusted[i] = min(1.0, (m - i) * p[i])
    # Enforce monotonicity
    p_adjusted[i] = max(p_adjusted[i-1], p_adjusted[i])
```

**When to use each method:**
- **Holm**: Standard model comparisons (controls FWER, more powerful than Bonferroni)
- **Bonferroni**: When being very conservative
- **Benjamini-Hochberg**: Exploratory analysis with many models (controls FDR)

### Effect Size Calculations

**Cohen's d**
```python
d = (mean₁ - mean₂) / pooled_std
pooled_std = sqrt((std₁² + std₂²) / 2)
```

**Interpretation:**
- d ≈ 0.2: Small effect
- d ≈ 0.5: Medium effect
- d ≈ 0.8: Large effect
- d > 1.0: Very large effect

### Robust Statistics

**Percentile-based limits** for outlier-resistant visualizations:
- Use 1st and 99th percentiles instead of min/max
- Add 5% padding for visual clarity
- Prevents single outliers from compressing the view

---

## Architecture Decisions

### Model Capabilities Registry

Centralized configuration to eliminate redundancy:

```python
MODEL_CAPABILITIES = {
    "o1": {
        "supports_temperature": False,
        "supports_system": False,
        "supports_tools": False,
        "supports_streaming": False,
        "supports_structured": True,
        "supports_prefill": False,
        "supports_json_schema": True,
        "force_json": False
    },
    # ... other models
}
```

**Benefits:**
- Single source of truth for model capabilities
- Easier to add new models
- Reduces code duplication across providers

### Response Parsing Architecture

**Dual-method approach for structured output:**
1. **Native parse mode**: Direct provider support (OpenAI json_schema)
2. **Instructor fallback**: For providers without native support

**Provider routing logic:**
```python
if provider == "openai" and parse_as:
    if is_o1_model(model):
        return _handle_o1_structured()
    else:
        return _handle_standard_structured()
```

### Model Detection Patterns

**Critical ordering** - most specific patterns first:
```python
MODEL_PATTERNS = [
    ("o1-mini", "o1-mini"),      # Most specific
    ("o1-preview", "o1-preview"),
    ("o1-", "o1"),                # More general
    ("o1", "o1"),                 # Fallback
    # ...
]
```

**Bug fixed**: o1-mini was incorrectly matching "o1" pattern due to wrong ordering.

---

## Performance Optimizations

### Visualization Data Loading
- Cache GloVe embeddings after first load
- Use DataFrame operations instead of loops where possible
- Batch file I/O operations

### Statistical Computations
- Vectorize pairwise comparisons using NumPy
- Pre-compute means/stds once and reuse
- Use Welch's t-test (more robust than Student's t)

---

## Known Limitations

1. **FutureWarning in loader.py**: pandas groupby deprecation at lines 136, 147
2. **Long model names**: May need truncation in plots
3. **Memory usage**: GloVe embeddings load ~2GB into memory

---

## Best Practices

### Adding New Visualizations
1. Follow the 3-file structure: plots.py, styles.py, loader.py
2. Always include statistical context (sample sizes, warnings)
3. Use consistent color scheme from `get_model_colors()`
4. Export at 300 DPI with `bbox_inches='tight'`

### Testing Visualizations
1. Test with small samples (n<10) to verify rug plots
2. Test with outliers to verify robust limits
3. Test with 5+ models to verify multiple comparison correction
4. Always generate both normalized and raw versions

### Code Quality
1. Use type hints for all public functions
2. Document statistical assumptions in docstrings
3. Validate input data shapes early
4. Handle edge cases explicitly (n=0, n=1, missing data)

---

## Future Enhancements

### Planned
- Bootstrap confidence intervals for very small samples (n<10)
- Power analysis annotations for underpowered comparisons
- Pagination helper for >8 models in ridge plots

### Under Consideration
- Bayesian alternatives for small samples
- Interactive HTML exports with Plotly
- Real-time visualization updates during experiments