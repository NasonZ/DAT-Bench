# DAT-Bench 🧠

Benchmarks for divergent thinking in LLMs — starting with the Divergent Association Task, with decomposition tracks planned.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Motivation

Most benchmarks test facts, logic, or chain-of-thought. DAT-Bench asks a complementary question:

> Do models that think more divergently produce better decompositions — and therefore more thoughtful answers?

We start with the **Divergent Association Task (DAT)**, a validated measure of verbal creativity where subjects generate 10 words as semantically different from each other as possible. Scores are computed via GloVe 840B cosine distances. We then plan to extend to **diversity-primed decomposition** — standard query/task decomposition with an explicit orthogonality objective.

## Current state

**DAT benchmark** — implemented and integrated with [verifiers](https://github.com/willccbb/verifiers) for RL training and eval.

- DAT scorer (GloVe 840B cosine distances)
- Composite reward rubric (creativity, validity, format compliance)
- Training environment (XMLParser + format reward signal)
- Eval environment (Pydantic structured output for deterministic parsing)
- Multi-provider support (OpenAI, Anthropic, Ollama, OpenRouter)
- Statistical visualisations (ridge plots, significance matrices, effect sizes)
- Hand-curated fixture dataset for pipeline validation

**Planned** — QD-DP (query decomposition, diversity-primed) and TD-DP (task decomposition, diversity-primed). See [Decomposition tracks](#decomposition-tracks-planned) below.

## Quick start

### Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv)
- GloVe 840B embeddings (set `GLOVE_PATH` env var, or place in `data/embeddings/`)

### Install

```bash
git clone https://github.com/NasonZ/DAT-Bench.git
cd DAT-Bench

uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### CLI

```bash
# OpenAI
uv run python scripts/run_dat.py \
  --provider openai --model gpt-5-mini \
  --strategy competitive --samples 20

# Ollama (local)
uv run python scripts/run_dat.py \
  --provider ollama --model llama3.2:3b \
  --strategy random --samples 15

# OpenRouter
export OPENROUTER_API_KEY="your-key"
uv run python scripts/run_dat.py \
  --provider openrouter --model meta-llama/llama-3.1-8b-instruct --samples 10
```

### Verifiers integration

DAT-Bench is a verifiers environment. Two loaders for different use cases:

```python
from environments.dat_bench.dat_bench import load_environment, load_eval_environment

# Training: XMLParser + format reward signal
train_env = load_environment(strategy="competitive", num_examples=50)

# Eval: Pydantic structured output, deterministic parsing
eval_env = load_eval_environment(strategy="DAT_instructions", num_examples=30)
```

The rubric provides composite reward signals for RL training:

| Signal | Weight | Purpose |
|---|---|---|
| `creativity_reward` | 1.0 | Normalized DAT score (primary gradient) |
| `validity_reward` | 0.2 | Fraction of words in GloVe vocabulary |
| `format_reward` | 0.1 | XML format compliance (training only) |
| `raw_dat_score` | 0.0 | Raw score for analysis (0-200 scale) |
| `valid_word_count` | 0.0 | Count of valid words for analysis |

### Scorer directly

```python
from divergent_bench import DATScorer

scorer = DATScorer()
score = scorer.dat(["whale", "hammer", "symphony", "cactus", "glacier",
                     "umbrella", "passport", "volcano", "whistle", "tapestry"])
print(f"DAT score: {score:.1f}")  # ~90
```

## Strategies

Four prompting strategies, each testing a different framing:

| Strategy | Temperature | Description |
|---|---|---|
| `none` | 0.7 | Minimal instructions — baseline |
| `competitive` | 0.7 | Prize framing with tips for maximising distance |
| `DAT_instructions` | 0.7 | Full task context from the original DAT paper |
| `random` | 1.0 | Explicit randomness instruction |

## Scoring

DAT score = average cosine distance between all pairs of the first 7 valid words (out of 10 provided), multiplied by 100. Range: 0-200. Higher is more creative.

The rubric normalises raw scores to 0-1 using a linear map from the empirical range [40, 100].

## Decomposition tracks (planned)

These tracks keep familiar decomposition workflows but ask the model to make sub-parts as different as possible, improving coverage.

**QD-DP** (Query Decomposition, Diversity-Primed) — given a complex question, generate orthogonal sub-questions, answer each, synthesise. Scored on orthogonality, coverage against a topic map, and redundancy penalty.

**TD-DP** (Task Decomposition, Diversity-Primed) — given a complex task, generate distinct top-level approaches, pick one, build a plan tree. Scored on approach diversity, actionability, and risk assessment via LLM-as-judge rubrics.

Both tracks will include standard (non-diversity-primed) baselines and report effect sizes with multiple-comparison correction.

## Project structure

```
DAT-Bench/
├── divergent_bench/
│   ├── dat/                          # DAT scorer (GloVe cosine distances)
│   ├── rubrics/                      # Verifiers-compatible reward rubrics
│   ├── config/                       # Prompting strategies, model configs
│   ├── data/                         # Fixture datasets for validation
│   ├── experiments/                  # Experiment runner, structured output
│   ├── llm/                          # Multi-provider client adapters
│   ├── metrics/                      # Divergence metrics (DSI, LZiv)
│   ├── visualization/                # Ridge plots, heatmaps, stats
│   ├── decomposition/               # (planned) QD/TD implementations
│   └── utils/
├── environments/
│   └── dat_bench/                    # Verifiers environment definition
│       ├── dat_bench.py              # load_environment / load_eval_environment
│       └── pyproject.toml
├── configs/
│   └── eval/                         # Eval configuration (TOML)
├── docs/
│   ├── research/                     # Alignment plans, design docs
│   ├── api/                          # API integration notes
│   └── development/                  # Roadmap, technical notes
├── tests/
│   ├── unit/                         # Scorer, fixture, metric tests
│   └── integration/                  # API, e2e, provider tests
├── scripts/                          # CLI entry points
├── pyproject.toml
└── LICENSE
```

## Statistical features

- **Multiple-comparison control** — Holm (default), Bonferroni, Benjamini-Hochberg
- **Effect sizes** — Cohen's *d* with standard thresholds
- **Small-sample handling** — adaptive rendering (strip/box/violin by *n*), CI capping, low-*n* warnings
- **Colorblind-safe palette** — validated via Coblis simulator

## References

- Olson et al. (2021). *Naming unrelated words predicts creativity.* PNAS.
- Pennington et al. (2014). *GloVe: Global Vectors for Word Representation.*
- Cohen (1988). *Statistical Power Analysis for the Behavioral Sciences.*

## Citation

```bibtex
@software{divergent_bench,
  title  = {DAT-Bench: Divergent Thinking Benchmarks for LLMs},
  author = {Nason Zikayo},
  year   = {2025},
  url    = {https://github.com/NasonZ/DAT-Bench}
}
```

## License

MIT — see `LICENSE`.
