"""
DAT-Bench — Divergent Association Task as a verifiers environment.

Single-turn: one prompt → model generates 10 maximally different words.
Scored via GloVe 840B cosine distances (DATScorer) through a composite
rubric with creativity, validity, and format signals.

Two environment loaders for different use cases:
  load_environment()      — Training. XMLParser + format reward signal.
  load_eval_environment() — Eval. Pydantic structured output (DATWords) for
                            deterministic parsing via response_format.

Structured output is supported by OpenAI, Anthropic, vLLM, and llama.cpp
via their OpenAI-compatible APIs. Verifiers ≤0.1.2 passes response_format
through to the API but only reads message.content (the JSON serialization),
not message.parsed. The rubric's JSON extraction handles this transparently.
See dat_rubric.py module docstring for details. When verifiers adds native
structured output support, load_eval_environment can be simplified.

Datasets:
  Training: N independent trials of the same strategy prompt.
    Sampling temperature creates variety — no ground-truth answer exists.
  Eval: hand-curated fixture dataset with known score tiers for pipeline
    validation. Models still generate freely; fixtures provide expected
    score ranges to verify the pipeline works.

Usage:
    prime eval run dat-bench -m gpt-4.1-mini -n 30 -r 1 -s
    prime eval run dat-bench -m claude-sonnet -n 30 -r 1 -s -a '{"strategy": "competitive"}'
"""

import sys
import uuid
from pathlib import Path
from typing import List

import verifiers as vf
from datasets import Dataset
from pydantic import BaseModel, Field

# Ensure divergent_bench is importable
_BENCH_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _BENCH_ROOT / "divergent_bench"
if _PACKAGE_ROOT.exists():
    parent = str(_BENCH_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)

from divergent_bench.config.strategies import DAT_STRATEGIES, DEFAULT_TEMPERATURES
from divergent_bench.data.fixtures import FIXTURES, get_all_tiers
from divergent_bench.rubrics import DATRubric


# --------------------------------------------------------------------------
# Structured output model (used for eval)
# --------------------------------------------------------------------------

class DATWords(BaseModel):
    """Pydantic model for structured output parsing of DAT responses."""
    words: List[str] = Field(
        description="List of exactly 10 single English nouns that are as different from each other as possible",
        min_length=10,
        max_length=10,
    )


# --------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------

# Training: instructs XML output format matching the XMLParser schema.
# The format reward signal trains the model to comply with this structure.
TRAIN_SYSTEM_PROMPT = """\
You are completing the Divergent Association Task — a measure of verbal creativity.

Generate exactly 10 English nouns that are as semantically different from each \
other as possible. Words that would never appear in the same context score highest.

Rules:
- Only single English nouns (things, objects, concepts)
- No proper nouns, no technical jargon
- Think broadly across domains: physical objects, abstract concepts, natural \
phenomena, human activities, etc.

Respond with your 10 words inside <words> tags, one word per line:
<words>
word1
word2
...
word10
</words>"""

# Eval: no format instructions — structured output handles parsing.
EVAL_SYSTEM_PROMPT = """\
You are completing the Divergent Association Task — a measure of verbal creativity.

Generate exactly 10 English nouns that are as semantically different from each \
other as possible. Words that would never appear in the same context score highest.

Rules:
- Only single English nouns (things, objects, concepts)
- No proper nouns, no technical jargon
- Think broadly across domains: physical objects, abstract concepts, natural \
phenomena, human activities, etc."""


# --------------------------------------------------------------------------
# Dataset builders
# --------------------------------------------------------------------------

def _build_prompt_dataset(prompt_text: str, n: int, strategy: str) -> Dataset:
    """Build a dataset of N independent DAT trial prompts.

    Each row gets a unique trial_id so downstream analysis can distinguish
    rollouts even though the prompt text is identical.
    """
    return Dataset.from_dict(
        {
            "question": [prompt_text] * n,
            "answer": [""] * n,
            "info": [
                {
                    "strategy": strategy,
                    "trial_id": str(uuid.uuid4()),
                }
                for _ in range(n)
            ],
        }
    )


def _build_fixture_eval_dataset(strategy: str) -> Dataset:
    """Build eval dataset from hand-curated fixtures.

    Each fixture has a known tier and expected score range. The model still
    generates freely — fixture metadata is stored in `info` for pipeline
    validation against expectations.
    """
    prompt_text = DAT_STRATEGIES[strategy]

    questions = []
    answers = []
    infos = []

    for fixture in FIXTURES:
        if fixture["tier"] == "invalid":
            continue

        questions.append(prompt_text)
        answers.append(", ".join(fixture["words"]))
        infos.append(
            {
                "strategy": strategy,
                "trial_id": str(uuid.uuid4()),
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


# --------------------------------------------------------------------------
# Environment loaders
# --------------------------------------------------------------------------

def load_environment(
    strategy: str = "DAT_instructions",
    num_examples: int = 50,
    system_prompt: str | None = None,
    minimum_words: int = 7,
    use_fixture_eval: bool = True,
    num_eval_examples: int = 30,
) -> vf.SingleTurnEnv:
    """Load DAT training environment with XMLParser + format reward.

    Uses XML-formatted prompts so the model learns <words>...</words> output
    structure. Format compliance is a weighted reward signal (0.1) alongside
    creativity (1.0) and validity (0.2).

    Args:
        strategy: DAT prompting strategy (none, random, competitive, DAT_instructions).
        num_examples: Number of training examples (each is one independent trial).
        system_prompt: Override default system prompt.
        minimum_words: Minimum valid words required for DAT scoring (default 7).
        use_fixture_eval: Use hand-curated fixture dataset for eval.
        num_eval_examples: Eval examples if use_fixture_eval is False.
    """
    if strategy not in DAT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(DAT_STRATEGIES.keys())}"
        )

    prompt_text = DAT_STRATEGIES[strategy]

    parser = vf.XMLParser(fields=["words"], answer_field="words")

    rubric = DATRubric(
        parser=parser,
        strategy=strategy,
        minimum_words=minimum_words,
    )

    dataset = _build_prompt_dataset(prompt_text, num_examples, strategy)

    if use_fixture_eval:
        eval_dataset = _build_fixture_eval_dataset(strategy)
    else:
        eval_dataset = _build_prompt_dataset(prompt_text, num_eval_examples, strategy)

    temperature = DEFAULT_TEMPERATURES.get(strategy, 0.7)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt or TRAIN_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        sampling_args={"temperature": temperature, "max_tokens": 500},
    )


def load_eval_environment(
    strategy: str = "DAT_instructions",
    num_examples: int = 50,
    system_prompt: str | None = None,
    minimum_words: int = 7,
    use_fixture_eval: bool = True,
    num_eval_examples: int = 30,
) -> vf.SingleTurnEnv:
    """Load DAT eval environment with Pydantic structured output.

    Uses response_format=DATWords for deterministic parsing — no XML tags
    needed, no format reward signal (irrelevant for eval). The structured
    output constraint is handled by the API provider, not the model.

    Supported by OpenAI, Anthropic, vLLM, and llama.cpp via their
    OpenAI-compatible APIs.

    Args:
        strategy: DAT prompting strategy (none, random, competitive, DAT_instructions).
        num_examples: Number of eval examples.
        system_prompt: Override default system prompt.
        minimum_words: Minimum valid words required for DAT scoring (default 7).
        use_fixture_eval: Use hand-curated fixture dataset for eval.
        num_eval_examples: Eval examples if use_fixture_eval is False.
    """
    if strategy not in DAT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(DAT_STRATEGIES.keys())}"
        )

    prompt_text = DAT_STRATEGIES[strategy]

    # Eval: no XMLParser needed, structured output handles parsing.
    # DATRubric still uses its extract_words() which handles both XML
    # and plain text, so it works with structured output responses too.
    rubric = DATRubric(
        strategy=strategy,
        minimum_words=minimum_words,
    )

    dataset = _build_prompt_dataset(prompt_text, num_examples, strategy)

    if use_fixture_eval:
        eval_dataset = _build_fixture_eval_dataset(strategy)
    else:
        eval_dataset = _build_prompt_dataset(prompt_text, num_eval_examples, strategy)

    temperature = DEFAULT_TEMPERATURES.get(strategy, 0.7)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt or EVAL_SYSTEM_PROMPT,
        rubric=rubric,
        sampling_args={
            "temperature": temperature,
            "max_tokens": 500,
            "response_format": DATWords,
        },
    )
