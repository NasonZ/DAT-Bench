"""
Core visualizations for divergent_bench.

Design principles (see styles.py for full rationale):
  - Statistical honesty: representation matches data quality
  - Data-ink ratio: every mark encodes information
  - Accessibility: works in grayscale and for colorblind viewers

Plot catalogue
--------------
  ranked_dot_plot     Primary "leaderboard" view — dot + 95% CI per model
  distribution_plot   Adaptive per-model distributions (dots / box / KDE by n)
  significance_matrix Pairwise effect sizes with significance stars
  word_model_heatmap  Word × model frequency heatmap
  triangular_matrix   Pairwise semantic distances for a single word set
"""

import itertools
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy import stats

from .styles import (
    FONT,
    SPACING,
    PALETTE,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    get_model_colors,
    truncate_label,
    _light_grid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_model_col(df: pd.DataFrame) -> str:
    for c in ("display_model", "model"):
        if c in df.columns:
            return c
    raise ValueError("No model column found in DataFrame")


def _order_by_mean(df: pd.DataFrame, model_col: str, score_col: str) -> list:
    return (
        df.groupby(model_col)[score_col]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _ci95(data):
    """Bootstrap-free 95% CI using t-distribution (correct for small n)."""
    n = len(data)
    if n < 2:
        m = data.mean()
        return m, m, m
    m = data.mean()
    se = data.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return m, m - t_crit * se, m + t_crit * se


# ===================================================================
# 1.  RANKED DOT PLOT  —  the "leaderboard" view
# ===================================================================

def ranked_dot_plot(
    df: pd.DataFrame,
    score_col: str = "score",
    model_col: str = None,
    order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, None),
    title: str = "Model Performance",
) -> plt.Figure:
    """Cleveland dot plot with 95% CI whiskers.

    Replaces the old bar chart: x-axis starts near the data range so
    differences between models are visually honest, and CIs use the
    t-distribution (correct for small n).

    Args:
        df: DataFrame with scores.
        score_col: Score column.
        model_col: Model column (auto-detected).
        order: Model display order (default: descending mean).
        figsize: (width, height). Height auto-scales with model count.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    model_col = model_col or _detect_model_col(df)
    if order is None:
        order = _order_by_mean(df, model_col, score_col)

    n_models = len(order)
    height = figsize[1] or max(3, 0.42 * n_models + 1.2)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    colors = get_model_colors(order)

    means, ci_lo, ci_hi, ns, labels = [], [], [], [], []
    for model in order:
        data = df.loc[df[model_col] == model, score_col].dropna()
        m, lo, hi = _ci95(data)
        n = len(data)
        means.append(m)
        ci_lo.append(lo)
        ci_hi.append(hi)
        ns.append(n)
        labels.append(f"{truncate_label(model)} (n={n})")

    y = np.arange(n_models)
    xerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
    xerr_hi = [hi - m for m, hi in zip(means, ci_hi)]

    # Compute a reasonable max CI width to cap extreme whiskers (n<5)
    widths = [hi - lo for lo, hi in zip(ci_lo, ci_hi)]
    median_width = np.median(widths) if widths else 1
    cap_width = max(median_width * 4, 5)  # don't let any bar exceed 4× median

    for i, model in enumerate(order):
        # Cap extreme CI widths for visual balance (still label honestly)
        lo_err = min(xerr_lo[i], cap_width / 2)
        hi_err = min(xerr_hi[i], cap_width / 2)
        capped = (lo_err < xerr_lo[i]) or (hi_err < xerr_hi[i])

        ax.errorbar(
            means[i], y[i],
            xerr=[[lo_err], [hi_err]],
            fmt="o",
            color=colors[model],
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            ecolor=colors[model],
            alpha=0.4 if ns[i] < 5 else 1.0,
            zorder=3,
        )

        # Score label — offset from the (possibly capped) whisker end
        label_x = means[i] + hi_err + 0.6
        label_text = f"{means[i]:.1f}"
        if capped:
            label_text += " *"  # flag that CI was capped

        ax.text(
            label_x, y[i], label_text,
            va="center", ha="left",
            fontsize=FONT["annotation"],
            color="#444",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    # Remove left spine for cleaner look
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # X-axis starts near the data, not at zero
    all_lo = min(ci_lo)
    all_hi = max(ci_hi)
    # Use capped range for limits
    display_lo = max(all_lo, min(means) - cap_width / 2)
    display_hi = min(all_hi, max(means) + cap_width / 2)
    pad = max(1, (display_hi - display_lo) * 0.12)
    ax.set_xlim(display_lo - pad, display_hi + pad + 4)  # room for labels
    ax.set_xlabel("Mean Score (95% CI)", fontsize=FONT["label"])

    _light_grid(ax, axis="x")
    ax.set_title(title, fontsize=FONT["title"], pad=SPACING["pad_title"])

    # Footnote for capped CIs
    any_capped = any(
        (xerr_lo[i] > cap_width / 2 or xerr_hi[i] > cap_width / 2)
        for i in range(n_models)
    )
    if any_capped:
        fig.text(
            0.98, 0.01, "* CI whisker capped for readability (n < 5)",
            ha="right", va="bottom",
            fontsize=FONT["annotation"], color="#888", fontstyle="italic",
        )

    fig.tight_layout(rect=[0, 0.03, 1, 1] if any_capped else None)
    return fig


# ===================================================================
# 2.  DISTRIBUTION PLOT  —  adaptive by sample size
# ===================================================================

def distribution_plot(
    df: pd.DataFrame,
    score_col: str = "score",
    model_col: str = None,
    order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, None),
    title: str = "Score Distributions",
) -> plt.Figure:
    """Adaptive distribution plot: strip → box+strip → violin+strip by n.

    Thresholds:
      n <  20 → strip plot only (raw dots, no smoothing)
      n 20-49 → box plot + strip overlay
      n >= 50 → violin + strip overlay

    Args:
        df: DataFrame with scores.
        score_col: Score column.
        model_col: Model column (auto-detected).
        order: Model display order (default: descending mean).
        figsize: (width, height). Height auto-scales.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    model_col = model_col or _detect_model_col(df)
    if order is None:
        order = _order_by_mean(df, model_col, score_col)

    n_models = len(order)
    height = figsize[1] or max(4, 0.6 * n_models + 1.2)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    colors = get_model_colors(order)
    df_plot = df[df[model_col].isin(order)].copy()

    # Category ordering for seaborn
    df_plot[model_col] = pd.Categorical(df_plot[model_col], categories=order, ordered=True)

    # Build labels with n
    counts = df_plot.groupby(model_col, observed=True).size()
    label_map = {
        m: f"{truncate_label(m)} (n={counts.get(m, 0)})"
        for m in order
    }

    # Draw per-model, adaptive
    for i, model in enumerate(order):
        data = df_plot.loc[df_plot[model_col] == model, score_col].dropna()
        n = len(data)
        c = colors[model]

        if n >= 50:
            # Violin + strip
            parts = ax.violinplot(
                data, positions=[i], vert=False, showmedians=False,
                showextrema=False, widths=0.7,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(c)
                pc.set_alpha(0.35)
                pc.set_edgecolor(c)
                pc.set_linewidth(0.8)
        elif n >= 20:
            # Box + strip
            ax.boxplot(
                data, positions=[i], vert=False, widths=0.45,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c, alpha=0.3, edgecolor=c, linewidth=0.8),
                medianprops=dict(color=c, linewidth=1.5),
                whiskerprops=dict(color=c, linewidth=0.8),
                capprops=dict(color=c, linewidth=0.8),
            )

        # Always show raw points (jittered)
        rng = np.random.default_rng(42 + i)  # unique seed per row
        jitter_range = 0.2 if n >= 20 else 0.12
        jitter = rng.uniform(-jitter_range, jitter_range, size=n)
        ax.scatter(
            data, i + jitter,
            s=18 if n < 20 else (12 if n < 50 else 6),
            color=c, alpha=0.7 if n < 20 else 0.55,
            edgecolors="white", linewidths=0.4,
            zorder=4,
        )

        # Mean marker — larger diamond, outlined for visibility
        mean_val = data.mean()
        ax.plot(
            mean_val, i, "D",
            color="black", markersize=6,
            markeredgecolor="white", markeredgewidth=0.8,
            zorder=5,
        )

    ax.set_yticks(range(n_models))
    ax.set_yticklabels([label_map[m] for m in order])
    ax.invert_yaxis()

    # Remove left spine, cleaner
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # X limits: percentile-based with breathing room
    lo, hi = np.nanpercentile(df_plot[score_col].dropna(), [1, 99])
    pad = max(1, (hi - lo) * 0.1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Score", fontsize=FONT["label"])

    _light_grid(ax, axis="x")
    ax.set_title(title, fontsize=FONT["title"], pad=SPACING["pad_title"])

    fig.tight_layout()
    return fig


# ===================================================================
# 3.  SIGNIFICANCE MATRIX  —  effect size + stars only
# ===================================================================

def significance_matrix(
    df: pd.DataFrame,
    score_col: str = "score",
    model_col: str = None,
    order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 6),
    title: str = "Pairwise Comparisons",
    correction: str = "holm",
) -> plt.Figure:
    """Heatmap of Cohen's d with significance stars.

    Color encodes effect size (diverging blue–red). Cell text shows only
    significance stars (*, **, ***). Effect size values are readable from
    the color scale — no double-encoding.

    Args:
        df: DataFrame with scores.
        score_col: Score column.
        model_col: Model column (auto-detected).
        order: Model order (default: descending mean).
        figsize: Figure size.
        title: Title.
        correction: Multiple comparison correction: "holm", "bonferroni",
                    "bh"/"fdr", or None.

    Returns:
        matplotlib Figure.
    """
    import seaborn as sns

    model_col = model_col or _detect_model_col(df)
    if order is None:
        order = _order_by_mean(df, model_col, score_col)

    n = len(order)
    effect = np.zeros((n, n))
    pvals = np.ones((n, n))
    computable = np.ones((n, n), dtype=bool)  # track pairs with enough data

    for i, m1 in enumerate(order):
        s1 = df.loc[df[model_col] == m1, score_col].dropna()
        for j, m2 in enumerate(order):
            if i == j:
                continue
            s2 = df.loc[df[model_col] == m2, score_col].dropna()
            if len(s1) < 2 or len(s2) < 2:
                computable[i, j] = False
                continue
            _, p = stats.ttest_ind(s1, s2, equal_var=False)
            pooled = np.sqrt((s1.std() ** 2 + s2.std() ** 2) / 2)
            d = (s1.mean() - s2.mean()) / pooled if pooled > 0 else 0
            effect[i, j] = d
            pvals[i, j] = p

    # Correction on unique pairs
    if correction:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if computable[i, j]:
                    pairs.append(((i, j), pvals[i, j]))
        if pairs:
            _apply_correction(pairs, pvals, correction)

    # Mask upper triangle + diagonal
    mask = np.triu(np.ones((n, n), dtype=bool))

    # Annotations: stars for significant, "—" for insufficient data
    annot = np.full((n, n), "", dtype=object)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            if not computable[i, j]:
                annot[i, j] = "—"
                continue
            p = pvals[i, j]
            if p < 0.001:
                annot[i, j] = "***"
            elif p < 0.01:
                annot[i, j] = "**"
            elif p < 0.05:
                annot[i, j] = "*"

    # Scale figure to be properly square given the label space
    side = max(figsize[0], 0.7 * n + 2.5)
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    vmax = max(np.abs(effect).max(), 0.5)

    # Light fill for masked (upper triangle) region so it doesn't blend with bg
    mask_fill = np.where(mask, 0.0, np.nan)

    short_labels = [truncate_label(m) for m in order]

    sns.heatmap(
        effect,
        mask=mask,
        annot=annot,
        fmt="",
        cmap=DIVERGING_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": "Cohen's d (effect size)",
            "shrink": 0.7,
            "aspect": 25,
        },
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax,
        annot_kws={"fontsize": FONT["annotation"] + 2, "fontweight": "bold"},
        square=True,
    )

    # Fill upper triangle with very light grey so it's clearly "intentionally empty"
    for i in range(n):
        for j in range(i, n):
            ax.add_patch(Rectangle(
                (j, i), 1, 1, fill=True,
                facecolor="#f5f5f5", edgecolor="white", linewidth=0.8,
            ))

    # X-axis: rotate with right-aligned anchor so labels don't clip
    ax.set_xticklabels(short_labels, rotation=40, ha="right",
                       fontsize=FONT["tick"])
    ax.tick_params(axis="y", rotation=0, labelsize=FONT["tick"])

    # Colorbar label styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT["tick"])
    cbar.set_label("Cohen's d (effect size)", fontsize=FONT["label"] - 1)

    ax.set_title(title, fontsize=FONT["title"], pad=SPACING["pad_title"])

    fig.tight_layout()
    return fig


def _apply_correction(pairs, pvals, method):
    """Apply multiple-comparison correction in-place to pvals matrix."""
    m = len(pairs)
    if method.lower() in ("holm", "holm-bonferroni"):
        order_idx = sorted(range(m), key=lambda k: pairs[k][1])
        adj = [None] * m
        prev = 0.0
        for rank, idx in enumerate(order_idx):
            val = min(1.0, (m - rank) * pairs[idx][1])
            val = max(prev, val)
            adj[idx] = val
            prev = val
        for idx, ((i, j), _) in enumerate(pairs):
            pvals[i, j] = pvals[j, i] = adj[idx]
    elif method.lower() == "bonferroni":
        for idx, ((i, j), _) in enumerate(pairs):
            p = min(1.0, pairs[idx][1] * m)
            pvals[i, j] = pvals[j, i] = p
    elif method.lower() in ("bh", "fdr", "benjamini-hochberg"):
        order_idx = sorted(range(m), key=lambda k: pairs[k][1])
        adj = [None] * m
        prev = 1.0
        for rank_rev, idx in enumerate(reversed(order_idx)):
            rank = m - rank_rev
            val = min(1.0, pairs[idx][1] * m / rank)
            val = min(prev, val)
            adj[idx] = val
            prev = val
        for idx, ((i, j), _) in enumerate(pairs):
            pvals[i, j] = pvals[j, i] = adj[idx]


# ===================================================================
# 4.  WORD × MODEL HEATMAP
# ===================================================================

def word_model_heatmap(
    df: pd.DataFrame,
    words_col: str = "words",
    model_col: str = None,
    top_n: int = 25,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "Word Usage by Model",
) -> plt.Figure:
    """Heatmap of word frequency across models.

    Replaces stacked bars: far more readable at high model counts because
    each cell is an independent comparison rather than a stacked segment.
    Normalized per-model (% of that model's samples containing the word).

    Args:
        df: DataFrame with word lists and model column.
        words_col: Column containing word lists.
        model_col: Model column (auto-detected).
        top_n: Number of top words to show.
        figsize: Figure size.
        title: Title.

    Returns:
        matplotlib Figure.
    """
    import seaborn as sns

    model_col = model_col or _detect_model_col(df)
    models = sorted(df[model_col].unique())
    model_sample_counts = df[model_col].value_counts()

    # Count word occurrences per model
    word_model = {}  # word -> {model: count}
    for _, row in df.iterrows():
        model = row[model_col]
        words = row[words_col]
        if isinstance(words, str):
            try:
                import ast
                words = ast.literal_eval(words)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(words, list):
            continue
        for w in words:
            if isinstance(w, str):
                w = w.lower()
                if w not in word_model:
                    word_model[w] = Counter()
                word_model[w][model] += 1

    # Top words by total count
    word_totals = {w: sum(c.values()) for w, c in word_model.items()}
    top_words = sorted(word_totals, key=word_totals.get, reverse=True)[:top_n]

    # Build matrix: normalize by model sample count (% of samples)
    matrix = np.zeros((len(top_words), len(models)))
    for i, word in enumerate(top_words):
        for j, model in enumerate(models):
            raw = word_model.get(word, {}).get(model, 0)
            n_samples = model_sample_counts.get(model, 1)
            matrix[i, j] = (raw / n_samples) * 100  # percentage

    # Mask zeros so they render as white (clearly "not used") vs. low-freq warm
    mask_zeros = matrix == 0

    # Scale figure width to model count to avoid label cramming
    auto_w = max(figsize[0], len(models) * 0.9 + 3)
    auto_h = max(figsize[1], len(top_words) * 0.38 + 2)
    fig, ax = plt.subplots(figsize=(auto_w, auto_h))

    sns.heatmap(
        matrix,
        mask=mask_zeros,
        xticklabels=[truncate_label(m) for m in models],
        yticklabels=top_words,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={
            "label": "% of model's samples",
            "shrink": 0.7,
            "aspect": 25,
        },
        ax=ax,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": FONT["annotation"] + 1},
    )

    # Draw "0" text in masked (zero) cells so they still have a value shown
    for i in range(len(top_words)):
        for j in range(len(models)):
            if mask_zeros[i, j]:
                ax.text(
                    j + 0.5, i + 0.5, "0",
                    ha="center", va="center",
                    fontsize=FONT["annotation"],
                    color="#ccc",
                )

    # X-axis: rotate with right-aligned anchor
    ax.set_xticklabels(
        [truncate_label(m) for m in models],
        rotation=40, ha="right", fontsize=FONT["tick"],
    )
    ax.tick_params(axis="y", rotation=0, labelsize=FONT["tick"])

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT["tick"])
    cbar.set_label("% of model's samples", fontsize=FONT["label"] - 1)

    # Title + subtitle
    n_total = len(df)
    n_mod = len(models)
    ax.set_title(title, fontsize=FONT["title"], pad=SPACING["pad_title"] + 14)
    ax.text(
        0.5, 1.02,
        f"n={n_total} samples · {n_mod} models · top {top_n} words · normalized per model",
        transform=ax.transAxes,
        ha="center", fontsize=FONT["annotation"] + 1, color="#666",
    )

    fig.tight_layout()
    return fig


# ===================================================================
# 5.  TRIANGULAR MATRIX  —  pairwise semantic distances
#     Modelled on the datcreativity.com results display
# ===================================================================

def triangular_matrix(
    words: List[str],
    distances: Optional[List[float]] = None,
    scorer=None,
    title: Optional[str] = None,
    score: Optional[float] = None,
    percentile: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 8),
    show_values: bool = True,
) -> plt.Figure:
    """Triangular matrix of pairwise semantic distances.

    Matches the DAT website (datcreativity.com) style:
      - Lower triangle only, words on both axes
      - Red→yellow→green colormap (low distance = red/bad, high = green/good)
      - Clean white background, no borders, no grid
      - Score + percentile header above the matrix

    Args:
        words: List of words.
        distances: Pre-computed pairwise distances, or None to compute.
        scorer: DATScorer instance (required if distances is None).
        title: Optional title line (e.g. model name).
        score: DAT score to display in header.
        percentile: Percentile rank to display in header.
        figsize: Figure size.
        show_values: Show distance values in cells.

    Returns:
        matplotlib Figure.
    """
    n = len(words)

    if distances is None:
        if scorer is None:
            raise ValueError("Provide either distances or scorer")
        # Track which words are valid vs invalid in GloVe
        validity = [(w, scorer.validate(w)) for w in words]
        invalid_words = {w for w, v in validity if v is None}
        valid_lookup = {w: v for w, v in validity if v is not None}

        # Compute distances only for valid pairs; None for pairs with invalid words
        distances = []
        for i in range(n):
            for j in range(i):
                w_i, w_j = words[i], words[j]
                if w_i in invalid_words or w_j in invalid_words:
                    distances.append(None)
                else:
                    distances.append(
                        scorer.distance(valid_lookup[w_i], valid_lookup[w_j]) * 100
                    )
    else:
        invalid_words = set()

    # DAT website colormap: red (low/bad) → yellow (mid) → green (high/good)
    dat_colors = ["#f4a0a0", "#f5c6a0", "#f5e6a0", "#d4e6a0", "#a8d5a0", "#7bc47b"]
    dat_cmap = LinearSegmentedColormap.from_list("dat_rg", dat_colors, N=128)

    valid_dists = [d for d in distances if d is not None]
    vmin = min(valid_dists) if valid_dists else 0
    vmax = max(valid_dists) if valid_dists else 200
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Grid layout: n rows × (n-1) columns
    # Row i displays word[i]. Row 0 (top) has no cells. Row n-1 (bottom) has n-1 cells.
    # Column j corresponds to word[j]. Columns go 0..n-2.
    # Cell at grid position (row=i, col=j) where i > j.
    #
    # Coordinate mapping (data coords, 1 unit = 1 cell):
    #   cell center:  x = col,  y = (n-1) - row
    #   row label:    x = -0.7, y = (n-1) - row
    #   col label:    x = col,  y = -0.7

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    # Limits with padding for labels
    ax.set_xlim(-2.5, n - 0.5)
    ax.set_ylim(-2.5, n + 0.5)

    # --- Draw cells ---
    idx = 0
    for row in range(1, n):
        for col in range(row):
            if idx >= len(distances):
                break
            val = distances[idx]
            cx = col
            cy = (n - 1) - row

            if val is None:
                rect = Rectangle((cx - 0.5, cy - 0.5), 1, 1,
                                  facecolor="#e0e0e0", edgecolor="white", linewidth=1.0)
                ax.add_patch(rect)
                if show_values:
                    ax.text(cx, cy, "?", ha="center", va="center",
                            fontsize=FONT["tick"], color="#999")
            else:
                color = dat_cmap(norm(val))
                rect = Rectangle((cx - 0.5, cy - 0.5), 1, 1,
                                  facecolor=color, edgecolor="white", linewidth=1.0)
                ax.add_patch(rect)
                if show_values:
                    ax.text(cx, cy, f"{val:.0f}", ha="center", va="center",
                            fontsize=FONT["tick"] + 1, fontweight="bold", color="black")
            idx += 1

    # --- Row labels (left) ---
    for row, word in enumerate(words):
        cy = (n - 1) - row
        is_inv = word in invalid_words
        ax.text(-0.7, cy, word, ha="right", va="center",
                fontsize=FONT["label"], fontweight="bold",
                color="#cc0000" if is_inv else "black",
                fontstyle="italic" if is_inv else "normal")

    # --- Column labels (bottom) ---
    for col, word in enumerate(words[:-1]):
        is_inv = word in invalid_words
        ax.text(col, -0.7, word, ha="right", va="top",
                fontsize=FONT["tick"],
                color="#cc0000" if is_inv else "black",
                fontstyle="italic" if is_inv else "normal",
                rotation=45)

    # --- Header (use figure suptitle for clean separation) ---
    header_parts = []
    if title:
        header_parts.append(title)
    if score is not None:
        s = f"Score: {score:.2f}"
        if percentile is not None:
            s += f"  (higher than {percentile:.1f}% of people)"
        header_parts.append(s)

    if header_parts:
        fig.suptitle(
            "\n".join(header_parts),
            fontsize=FONT["title"], fontweight="bold",
            y=0.97,
        )

    # --- Footnote for invalid words ---
    if invalid_words:
        note = f"? = not in GloVe vocabulary: {', '.join(sorted(invalid_words))}"
        fig.text(0.5, 0.02, note, ha="center", va="bottom",
                 fontsize=FONT["annotation"], color="#cc0000", fontstyle="italic")

    fig.subplots_adjust(top=0.90, bottom=0.08)
    return fig


# ===================================================================
# Backward-compatible aliases (old names → new implementations)
# ===================================================================

def ridge_plot(df, **kwargs):
    """Deprecated: use distribution_plot() instead."""
    logger.warning("ridge_plot() is deprecated, use distribution_plot()")
    return distribution_plot(df, **kwargs)


def statistical_heatmap(df, **kwargs):
    """Deprecated: use ranked_dot_plot() + significance_matrix() instead."""
    logger.warning("statistical_heatmap() is deprecated, use ranked_dot_plot() + significance_matrix()")
    # Return the dot plot as the primary view
    title = kwargs.pop("title", "Model Performance")
    return ranked_dot_plot(df, title=title, **{
        k: v for k, v in kwargs.items()
        if k in ("score_col", "model_col", "order", "figsize")
    })


def word_frequency_stacked(df, **kwargs):
    """Deprecated: use word_model_heatmap() instead."""
    logger.warning("word_frequency_stacked() is deprecated, use word_model_heatmap()")
    return word_model_heatmap(df, **{
        k: v for k, v in kwargs.items()
        if k in ("words_col", "model_col", "top_n", "figsize", "title")
    })
