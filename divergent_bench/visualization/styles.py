"""
Visual foundations for divergent_bench.

Design principles:
  1. Statistical honesty — representation must match data quality (no KDE for n<20)
  2. Data-ink ratio — every pixel encodes information, nothing decorative
  3. Accessibility — colorblind-safe by default, legible at all sizes
  4. Consistency — one palette, one type system, one spacing scale
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Dict, List, Optional
import hashlib

# ---------------------------------------------------------------------------
# Palette — colorblind-safe categorical + sequential + diverging
# ---------------------------------------------------------------------------

# Primary categorical palette: seaborn "colorblind" (CB-safe, 6 colors)
# Extended with hand-picked additions that maintain CB safety via luminance contrast.
PALETTE = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#D55E00",  # vermillion
    "#CC78BC",  # pink/mauve
    "#CA9161",  # brown
    "#FBAFE4",  # light pink
    "#949494",  # grey
    "#ECE133",  # yellow
    "#56B4E9",  # sky blue
]

# Provider-family hue anchors (optional overrides, still CB-safe)
PROVIDER_HUES = {
    "openai":    "#0173B2",  # blue family
    "anthropic": "#DE8F05",  # orange family
    "google":    "#029E73",  # green family
    "meta":      "#CC78BC",  # mauve family
    "deepseek":  "#D55E00",  # vermillion
    "ollama":    "#CA9161",  # brown
    "human":     "#949494",  # grey
}

# Diverging colormap for significance matrices (CB-safe)
DIVERGING_CMAP = "RdBu_r"

# Sequential colormap for distance matrices
SEQUENTIAL_CMAP = "YlGnBu"


def get_model_color(model_name: str, index: int = 0) -> str:
    """Deterministic color for a model name.

    Priority: provider prefix match → hash-based from PALETTE.
    """
    if not model_name:
        return "#949494"

    lower = model_name.lower()

    # Try provider prefix
    for provider, color in PROVIDER_HUES.items():
        if lower.startswith(provider) or provider in lower:
            return color

    # Stable hash fallback — same name always gets same color
    h = int(hashlib.md5(model_name.encode()).hexdigest()[:8], 16)
    return PALETTE[h % len(PALETTE)]


def get_model_colors(model_names: List[str]) -> Dict[str, str]:
    """Map a list of model names to colors, avoiding duplicates where possible."""
    colors = {}
    used = set()

    for name in model_names:
        c = get_model_color(name)
        # If collision with a different model, walk the palette
        if c in used:
            h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            for offset in range(1, len(PALETTE)):
                candidate = PALETTE[(h + offset) % len(PALETTE)]
                if candidate not in used:
                    c = candidate
                    break
        colors[name] = c
        used.add(c)

    return colors


# ---------------------------------------------------------------------------
# Typography & spacing — one scale, applied everywhere
# ---------------------------------------------------------------------------

FONT = {
    "family": "sans-serif",
    "title": 14,
    "label": 11,
    "tick": 9,
    "annotation": 8,
    "legend": 9,
}

SPACING = {
    "pad_title": 12,
    "pad_label": 8,
    "grid_alpha": 0.18,
    "grid_lw": 0.5,
    "spine_lw": 0.6,
}

DPI = {"screen": 100, "save": 300}


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------

def apply_theme(context: str = "paper"):
    """Apply the divergent_bench visual theme globally.

    Args:
        context: "paper" for publication, "notebook" for interactive use.
    """
    sns.set_style("white")

    scale = 1.0 if context == "paper" else 1.15

    rc = {
        # Typography
        "font.family": FONT["family"],
        "font.size": FONT["tick"] * scale,
        "axes.titlesize": FONT["title"] * scale,
        "axes.labelsize": FONT["label"] * scale,
        "xtick.labelsize": FONT["tick"] * scale,
        "ytick.labelsize": FONT["tick"] * scale,
        "legend.fontsize": FONT["legend"] * scale,
        "legend.title_fontsize": FONT["legend"] * scale,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": SPACING["spine_lw"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        # Grid
        "axes.grid": False,
        # DPI
        "figure.dpi": DPI["screen"],
        "savefig.dpi": DPI["save"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }

    plt.rcParams.update(rc)


# ---------------------------------------------------------------------------
# Helpers used by plot functions
# ---------------------------------------------------------------------------

def truncate_label(name: str, max_len: int = 22) -> str:
    """Shorten long model names for axis labels."""
    if len(name) <= max_len:
        return name
    # Try dropping common prefixes
    for prefix in ("hf.co/", "huggingface/", "openai/", "anthropic/"):
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "\u2026"


def _light_grid(ax, axis="x"):
    """Add a subtle reference grid."""
    ax.grid(axis=axis, alpha=SPACING["grid_alpha"], linewidth=SPACING["grid_lw"])
    ax.set_axisbelow(True)
