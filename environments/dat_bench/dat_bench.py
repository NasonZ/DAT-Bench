"""
DAT-Bench — Divergent Association Task as a verifiers environment.

Single-turn: one prompt -> model generates 10 maximally different words.
Scored via GloVe 840B cosine distances (DATScorer).

Datasets:
  - Training: N independent trials of the same strategy prompt (sampling creates variety).
  - Eval: hand-curated fixture dataset with known score tiers for pipeline validation.
    Models still generate freely — the fixtures provide *expected score ranges* to
    verify the pipeline is working, not ground-truth answers.

Usage:
    prime eval run dat-bench -m gpt-4.1-mini -n 30 -r 1 -s
    prime eval run dat-bench -m claude-sonnet -n 30 -r 1 -s -a '{"strategy": "competitive"}'
"""

import sys
from pathlib import Path

import verifiers as vf
from datasets import Dataset

# Ensure divergent_bench is importable (handles editable installs / dev setups)
_BENCH_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _BENCH_ROOT / "divergent_bench"
if _PACKAGE_ROOT.exists():
    parent = str(_BENCH_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)

from divergent_bench.config.strategies import DAT_STRATEGIES, DEFAULT_TEMPERATURES
from divergent_bench.data.fixtures import FIXTURES, get_all_tiers
from divergent_bench.rubrics import DATRubric


def _build_prompt_dataset(prompt_text: str, n: int, strategy: str) -> Dataset:
    """Build a dataset of N identical DAT trial prompts."""
    return Dataset.from_dict(
        {
            "question": [prompt_text] * n,
            "answer": [""] * n,
            "info": [{"strategy": strategy}] * n,
        }
    )


def _build_fixture_eval_dataset(strategy: str) -> Dataset:
    """Build eval dataset from hand-curated fixtures.

    Each fixture has a known tier and expected score range.
    The model still generates freely — the fixture metadata is stored in
    `info` so we can validate pipeline outputs against expectations.
    """
    prompt_text = DAT_STRATEGIES[strategy]
    tiers = get_all_tiers()

    questions = []
    answers = []
    infos = []

    for fixture in FIXTURES:
        # Skip invalid-tier fixtures for eval (they test error handling,
        # not model output — use them in unit tests instead)
        if fixture["tier"] == "invalid":
            continue

        questions.append(prompt_text)
        # Store the fixture words as "answer" — not for exact matching, but
        # so the fixture is visible in results for manual inspection
        answers.append(", ".join(fixture["words"]))
        infos.append(
            {
                "strategy": strategy,
                "fixture_tier": fixture["tier"],
                "fixture_words": fixture["words"],
                "expected_score_low": fixture["expected_score_low"],
                "expected_score_high": fixture["expected_score_high"],
                "expected_reward_low": fixture["expected_reward_low"],
                "expected_reward_high": fixture["expected_reward_high"],
                "fixture_note": fixture["note"],
            }
        )

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "info": infos,
        }
    )


def load_environment(
    strategy: str = "DAT_instructions",
    num_examples: int = 50,
    system_prompt: str | None = None,
    minimum_words: int = 7,
    use_fixture_eval: bool = True,
    num_eval_examples: int = 30,
) -> vf.Environment:
    """Load DAT benchmark as a verifiers environment.

    Args:
        strategy: DAT prompting strategy (none, random, competitive, DAT_instructions).
        num_examples: Number of training examples (each is one independent DAT trial).
        system_prompt: Optional system prompt override.
        minimum_words: Minimum valid words required for scoring (default 7).
        use_fixture_eval: If True, use hand-curated fixture dataset for eval.
            If False, use N synthetic prompt copies (same as training).
        num_eval_examples: Number of eval examples (only used if use_fixture_eval=False).
    """
    if strategy not in DAT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(DAT_STRATEGIES.keys())}"
        )

    prompt_text = DAT_STRATEGIES[strategy]

    dataset = _build_prompt_dataset(prompt_text, num_examples, strategy)

    if use_fixture_eval:
        eval_dataset = _build_fixture_eval_dataset(strategy)
    else:
        eval_dataset = _build_prompt_dataset(prompt_text, num_eval_examples, strategy)

    rubric = DATRubric(strategy=strategy, minimum_words=minimum_words)

    temperature = DEFAULT_TEMPERATURES.get(strategy, 0.7)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        sampling_args={"temperature": temperature, "max_tokens": 500},
    )
