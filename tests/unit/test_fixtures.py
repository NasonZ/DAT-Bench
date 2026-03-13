"""
Validate that hand-curated fixtures score within their expected ranges.

This is the core pipeline validation test: if our mental model of
'these words are terrible/average/elite' doesn't match the actual DAT
scorer output, either the fixtures are wrong or the pipeline is broken.

Requires GloVe 840B embeddings (set GLOVE_PATH and WORDS_PATH env vars).
"""

import os
import pytest

from divergent_bench.data.fixtures import FIXTURES, get_fixtures_by_tier, get_all_tiers


needs_glove = pytest.mark.skipif(
    not os.getenv("GLOVE_PATH"),
    reason="GLOVE_PATH not set — GloVe 840B embeddings required",
)


@pytest.fixture(scope="module")
def scorer():
    """Lazy-load scorer once for all tests in this module."""
    from divergent_bench.dat.scorer import DATScorer
    return DATScorer()


@needs_glove
@pytest.mark.parametrize(
    "fixture",
    [f for f in FIXTURES if f["tier"] != "invalid"],
    ids=[f"{f['tier']}_{i}" for i, f in enumerate(f for f in FIXTURES if f["tier"] != "invalid")],
)
def test_fixture_score_in_expected_range(scorer, fixture):
    """Each fixture's raw DAT score should fall within its declared range."""
    score = scorer.dat(fixture["words"], minimum=7)
    assert score is not None, f"Fixture {fixture['tier']} returned None (not enough valid words?)"
    assert fixture["expected_score_low"] <= score <= fixture["expected_score_high"], (
        f"Fixture '{fixture['tier']}' ({fixture['note']}): "
        f"score {score:.2f} outside expected [{fixture['expected_score_low']}, {fixture['expected_score_high']}]"
    )


@needs_glove
@pytest.mark.parametrize(
    "fixture",
    [f for f in FIXTURES if f["tier"] != "invalid"],
    ids=[f"{f['tier']}_{i}" for i, f in enumerate(f for f in FIXTURES if f["tier"] != "invalid")],
)
def test_fixture_reward_in_expected_range(scorer, fixture):
    """Each fixture's normalized reward should fall within its declared range."""
    score = scorer.dat(fixture["words"], minimum=7)
    assert score is not None
    reward = max(0.0, min(1.0, (score - 40.0) / 60.0))
    assert fixture["expected_reward_low"] <= reward <= fixture["expected_reward_high"], (
        f"Fixture '{fixture['tier']}' ({fixture['note']}): "
        f"reward {reward:.3f} outside expected [{fixture['expected_reward_low']}, {fixture['expected_reward_high']}]"
    )


@needs_glove
@pytest.mark.parametrize(
    "fixture",
    [f for f in FIXTURES if f["tier"] == "invalid"],
    ids=[f"invalid_{i}" for i, _ in enumerate(f for f in FIXTURES if f["tier"] == "invalid")],
)
def test_invalid_fixtures_return_none(scorer, fixture):
    """Invalid fixtures should return None from the scorer."""
    score = scorer.dat(fixture["words"], minimum=7)
    assert score is None, f"Expected None for invalid fixture, got {score}"


@needs_glove
def test_tier_ordering(scorer):
    """Tier mean scores should be monotonically increasing: terrible < bad < average < good < elite."""
    tier_means = {}
    for tier in ["terrible", "bad", "average", "good", "elite"]:
        fixtures = get_fixtures_by_tier(tier)
        scores = []
        for f in fixtures:
            s = scorer.dat(f["words"], minimum=7)
            if s is not None:
                scores.append(s)
        if scores:
            tier_means[tier] = sum(scores) / len(scores)

    ordered = ["terrible", "bad", "average", "good", "elite"]
    for i in range(len(ordered) - 1):
        lo, hi = ordered[i], ordered[i + 1]
        if lo in tier_means and hi in tier_means:
            assert tier_means[lo] < tier_means[hi], (
                f"Tier ordering violated: {lo} ({tier_means[lo]:.2f}) >= {hi} ({tier_means[hi]:.2f})"
            )


def test_fixture_data_integrity():
    """Basic structural checks on fixture data (no GloVe needed)."""
    assert len(FIXTURES) >= 10, "Expected at least 10 fixtures"

    valid_tiers = set(get_all_tiers())
    for f in FIXTURES:
        assert f["tier"] in valid_tiers, f"Unknown tier: {f['tier']}"
        assert isinstance(f["words"], list)
        assert f["expected_score_low"] <= f["expected_score_high"]
        assert f["expected_reward_low"] <= f["expected_reward_high"]
        assert 0.0 <= f["expected_reward_low"] <= 1.0
        assert 0.0 <= f["expected_reward_high"] <= 1.0
        assert f["note"], "Each fixture should have a descriptive note"

    # Check all tiers are represented
    tiers_present = {f["tier"] for f in FIXTURES}
    assert tiers_present == valid_tiers, f"Missing tiers: {valid_tiers - tiers_present}"
