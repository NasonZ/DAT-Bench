"""
Hand-curated DAT fixture dataset for pipeline validation.

Each entry has a known tier and expected score range, so we can verify
the full pipeline (word extraction → GloVe scoring → reward normalization)
produces sensible outputs. Mental model: if you read the words, you should
be able to predict roughly where the score lands.

Tiers (raw DAT score ranges, approximate):
  terrible:  35-50   (same category, near-synonyms)
  bad:       50-62   (slight variety but obvious clusters)
  average:   62-75   (lazy diversity, common words)
  good:      75-88   (genuinely diverse across domains)
  elite:     88-96   (maximally distant, rare/edge-case words)
  invalid:   N/A     (too few valid words → reward 0.0)
"""

from __future__ import annotations

from typing import TypedDict


class FixtureEntry(TypedDict):
    words: list[str]
    tier: str
    expected_score_low: float   # raw DAT score lower bound
    expected_score_high: float  # raw DAT score upper bound
    expected_reward_low: float  # normalized reward lower bound
    expected_reward_high: float # normalized reward upper bound
    note: str


# Normalization: reward = clamp((raw - 40) / 60, 0, 1)
FIXTURES: list[FixtureEntry] = [
    # --- TERRIBLE (same semantic cluster) ---
    {
        "words": ["cat", "dog", "fish", "bird", "horse", "cow", "sheep", "goat", "pig", "duck"],
        "tier": "terrible",
        "expected_score_low": 45.0,
        "expected_score_high": 55.0,
        "expected_reward_low": 0.05,
        "expected_reward_high": 0.25,
        "note": "All animals — minimal semantic distance",
    },
    {
        "words": ["apple", "banana", "orange", "grape", "mango", "peach", "plum", "cherry", "lemon", "melon"],
        "tier": "terrible",
        "expected_score_low": 35.0,
        "expected_score_high": 45.0,
        "expected_reward_low": 0.0,
        "expected_reward_high": 0.10,
        "note": "All fruits — even tighter cluster than animals",
    },
    {
        "words": ["freedom", "justice", "truth", "honor", "courage", "wisdom", "faith", "hope", "love", "peace"],
        "tier": "terrible",
        "expected_score_low": 42.0,
        "expected_score_high": 52.0,
        "expected_reward_low": 0.0,
        "expected_reward_high": 0.20,
        "note": "All abstract virtues — same semantic neighborhood",
    },

    # --- BAD (slight variety but lazy clusters) ---
    {
        "words": ["car", "truck", "boat", "plane", "train", "bicycle", "house", "garden", "river", "mountain"],
        "tier": "bad",
        "expected_score_low": 54.0,
        "expected_score_high": 65.0,
        "expected_reward_low": 0.20,
        "expected_reward_high": 0.42,
        "note": "6 vehicles + 4 outdoor/nature — two obvious clusters",
    },

    # --- AVERAGE (some diversity, still predictable) ---
    {
        "words": ["car", "tree", "music", "dog", "house", "rain", "book", "chair", "lake", "bread"],
        "tier": "average",
        "expected_score_low": 66.0,
        "expected_score_high": 78.0,
        "expected_reward_low": 0.40,
        "expected_reward_high": 0.63,
        "note": "Common everyday words with moderate spread",
    },
    {
        "words": ["table", "chair", "lamp", "idea", "dream", "hope", "rock", "sand", "wind", "fire"],
        "tier": "average",
        "expected_score_low": 63.0,
        "expected_score_high": 75.0,
        "expected_reward_low": 0.35,
        "expected_reward_high": 0.58,
        "note": "Physical + abstract mix but lazy picks",
    },

    # --- GOOD (genuinely diverse) ---
    {
        "words": ["whale", "hammer", "symphony", "cactus", "glacier", "umbrella", "passport", "volcano", "whistle", "tapestry"],
        "tier": "good",
        "expected_score_low": 80.0,
        "expected_score_high": 92.0,
        "expected_reward_low": 0.65,
        "expected_reward_high": 0.87,
        "note": "Strong diversity — animals, tools, music, plants, geography, objects, documents",
    },
    {
        "words": ["anchor", "melody", "fungus", "blueprint", "drought", "velvet", "algebra", "orphan", "turbine", "quartz"],
        "tier": "good",
        "expected_score_low": 85.0,
        "expected_score_high": 95.0,
        "expected_reward_low": 0.75,
        "expected_reward_high": 0.92,
        "note": "Spans maritime, music, biology, design, weather, texture, math, social, engineering, geology",
    },

    # --- ELITE (maximally distant) ---
    {
        "words": ["quarantine", "magma", "lullaby", "fossil", "bandwidth", "origami", "drought", "pupil", "anchor", "velvet"],
        "tier": "elite",
        "expected_score_low": 88.0,
        "expected_score_high": 98.0,
        "expected_reward_low": 0.80,
        "expected_reward_high": 0.97,
        "note": "Medical, geology, music, paleontology, tech, craft, weather, education, maritime, texture",
    },
    {
        "words": ["fjord", "pamphlet", "yawn", "cobalt", "treason", "plankton", "hymn", "debris", "elbow", "monarchy"],
        "tier": "elite",
        "expected_score_low": 87.0,
        "expected_score_high": 97.0,
        "expected_reward_low": 0.78,
        "expected_reward_high": 0.95,
        "note": "Geography, publishing, physiology, chemistry, law, biology, religion, destruction, anatomy, politics",
    },

    # --- INVALID (pipeline should handle gracefully → reward 0.0) ---
    {
        "words": ["xylqz", "brrpt", "fnord", "zxcvb", "whale", "hammer", "symphony", "cactus", "qwert", "asdfg"],
        "tier": "invalid",
        "expected_score_low": 0.0,
        "expected_score_high": 0.0,
        "expected_reward_low": 0.0,
        "expected_reward_high": 0.0,
        "note": "Only 4 valid words — below minimum of 7, should return None → reward 0.0",
    },
    {
        "words": [],
        "tier": "invalid",
        "expected_score_low": 0.0,
        "expected_score_high": 0.0,
        "expected_reward_low": 0.0,
        "expected_reward_high": 0.0,
        "note": "Empty input — should return reward 0.0",
    },
]


def get_fixtures_by_tier(tier: str) -> list[FixtureEntry]:
    """Get all fixtures for a given tier."""
    return [f for f in FIXTURES if f["tier"] == tier]


def get_all_tiers() -> list[str]:
    """Get ordered list of tiers."""
    return ["terrible", "bad", "average", "good", "elite", "invalid"]
