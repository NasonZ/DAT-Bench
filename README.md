# DAT Bench 🧠

**Benchmarks for *divergent thinking* in LLMs - starting with DAT, then adding decomposition that stays comparable to standard baselines.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/uv-latest-green.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why this repo

Most benchmarks test facts, logic, or chain-of-thought. **DAT Bench** asks a complementary question:

> **Do models that think more *divergently* produce better *decompositions*  and therefore more thoughtful answers/plans**

We begin with a solid implementation of the **Divergent Association Task (DAT)** and extend to **diversity-primed decomposition** that looks like normal query/task decomposition, but explicitly asks for **orthogonal** ideas to avoid overlap and blind spots.

* **DAT** - measures raw divergent capacity (semantic distance across 10 words).
* **QD-DP** *(planned)* - *Query Decomposition - Diversity-Primed* (standard QD with an orthogonality objective).
* **TD-DP** *(planned)* - *Task Decomposition - Diversity-Primed* (standard plan/approach tree with diversity across top-level approaches).

The point: stay **comparable** to common baselines while testing whether a light push toward diversity actually helps.

---

## What’s here today

* ✅ **DAT scorer** (GloVe by default; embeddings are pluggable).
* ✅ **Visualizations** (ridge plots, statistical heatmaps, word-frequency analysis).
* ✅ **Honest stats** (effect sizes, multiple-comparison correction, small-n warnings).
* ✅ **Multi-provider hooks** (OpenAI, Ollama, OpenRouter; easy to add others).

Next up: QD-DP / TD-DP runners, datasets, and metrics that reuse the same analytics/plots so comparisons stay apples-to-apples.

---

## Key features

* 📊 **Statistical analytics** - Cohen's *d*, Holm/Bonferroni/FDR corrections, bootstrap CIs (in progress).
* 🤖 **Multi-provider support** - OpenAI, Ollama, OpenRouter, and local models.
* 🧪 **Strategy testing** - compare prompting strategies (none, competitive, DAT_instructions, random).
* 🧭 **Comparable decomposition tracks** - standard workflows with an explicit **orthogonality** goal (planned).
* 🧱 **Extensible design** - shared runners, scoring and plotting across tracks.

---

## Quick start

### Prerequisites

* Python **3.10+**
* [UV](https://github.com/astral-sh/uv)
* GloVe embeddings (auto-downloaded on first use)

### Install

```bash
git clone https://github.com/NasonZ/DAT-Bench.git
cd DAT-Bench

uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

uv pip install -e .
```

---

## Usage - DAT benchmark (available)

```bash
# OpenAI
uv run python scripts/run_dat.py \
  --provider openai --model gpt-5-mini \
  --strategy random --samples 10

# Ollama (local)
uv run python scripts/run_dat.py \
  --provider ollama --model llama3.2:3b \
  --strategy competitive --samples 20

# Custom temperature
uv run python scripts/run_dat.py \
  --provider ollama --model qwen3:4b \
  --strategy random --temperature 0.7 --samples 15

# OpenRouter
export OPENROUTER_API_KEY="your-key"
uv run python scripts/run_dat.py \
  --provider openrouter --model meta-llama/llama-3.1-8b-instruct --samples 10
```

### Use as a library

```python
from divergent_bench.dat.scorer import DATScorer
from divergent_bench.llm import create_provider
from divergent_bench.experiments.runner import DATExperiment

provider = create_provider("openai", model="gpt-5.1-mini")
scorer = DATScorer()

experiment = DATExperiment(provider, scorer)
results = experiment.run(strategy="random", num_samples=10)

print(f"Mean DAT score: {results['mean_score']:.2f}")
print(f"Best words: {results['best_sample']['words']}")
```

---

## Decomposition tracks (planned) - *diversity-primed, still standard*

These tracks keep the familiar decomposition workflow; they simply ask the model to make the parts **as different as possible** to improve coverage.

### QD-DP - Query Decomposition (Diversity-Primed)

* **Input:** a complex question (e.g., “How should the UK address the housing crisis?”).
* **Output:** *k* **orthogonal** sub-questions -> brief answers (2–3 sentences each) -> a short synthesis referencing sub-question IDs.
* **Scoring (automatic):**

  * **Orthogonality** - dispersion of sub-questions in embedding space.
  * **Coverage** - greedy match to a small YAML **topic map** per prompt; average matched similarity.
  * **Redundancy penalty** - share of pairs above a similarity threshold (e.g., 0.85).
  * **Synthesis rubric** - 0–5 for Structure, Coverage, Trade-offs, Specificity, Calibration (lightweight judge or rules).

### TD-DP - Task Decomposition (Diversity-Primed)

* **Input:** a complex task (policy, product, infra, ML, etc.).
* **Output:** *m* **distinct** top-level approaches -> pick one -> compact **plan tree** (depth 2–3) with risks & checks.
* **Scoring (LLM-as-judge with rubrics):**

  * **Approach diversity** - dispersion across approach summaries.
  * **Actionability** - rubric evaluating task specificity, dependencies, success criteria, and executability.
  * **Risk assessment** - rubric scoring risk identification, relevance, mitigation strategies, and coverage.
  * **Consistency** - child->parent semantic alignment.

### Baselines & A/B

* **Baseline:** standard decomposition (no diversity prompt).
* **Treatment:** diversity-primed decomposition (orthogonality prompt).
* **Report:** effect sizes for coverage/orthogonality/synthesis quality with multiple-comparison control.

---

## Visualizations

### Ridge plots (DAT now; reused for QD/TD scores)

* Distributions per model/strategy
* Rug plots for small *n* (≤10)
* Small-sample warnings (red: *n*<10, orange: *n*<30)
* Robust axes via 1st–99th percentiles

### Statistical heatmaps

* Performance bars + significance matrix
* Effect sizes or *t*-stats
* Holm/Bonferroni/FDR corrections

### Word frequency analysis (DAT)

* Stacked bars showing which models generated which words
* Multiple normalization modes (raw counts, per-word attribution, total percentage)
* Visual identification of convergence (many models -> same word) vs divergence (unique words per model)

---

## Project structure

```
DAT-Bench/
├── divergent_bench/
│   ├── dat/                      # DAT implementation (available)
│   │   ├── __init__.py
│   │   └── scorer.py
│   ├── llm/                      # LLM provider adapters
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── config.py
│   │   └── providers.py
│   ├── metrics/                  # Divergence metrics
│   │   ├── __init__.py
│   │   ├── dsi.py                # Divergent Semantic Integration
│   │   └── lziv.py               # Lempel-Ziv complexity
│   ├── experiments/
│   │   └── runner.py             # DAT experiment orchestration
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── plots.py
│   │   ├── styles.py
│   │   └── README.md
│   ├── config/
│   │   └── strategies.py         # Prompting strategies config
│   ├── decomposition/            # (planned) QD/TD implementations
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── docs/
│   ├── api/
│   │   └── structured-output.md
│   └── development/
│       ├── ROADMAP.md
│       ├── cli-implementation.md
│       └── technical-notes.md
├── tests/
│   ├── unit/
│   │   ├── test_dat_scorer.py
│   │   ├── test_metrics.py
│   │   └── test_model_detection.py
│   └── integration/
│       ├── test_api_endpoints.py
│       ├── test_end_to_end.py
│       ├── test_ollama_models.py
│       └── test_structured_output.py
├── scripts/
│   ├── run_dat.py
│   ├── demo.py
│   ├── run_qd.py                 # (planned)
│   └── run_td.py                 # (planned)
├── datasets/                     # (planned) YAML topic maps / plan rubrics
│   ├── qd/                       # e.g., housing_crisis_v1.yml
│   └── td/                       # e.g., rag_mvp_v1.yml
├── judges/                       # (planned) small rubric evaluators
├── results/                      # (will be created on first run)
├── visualizations/               # (will be created on first run)
├── pyproject.toml
├── uv.lock
├── CHANGELOG.md
└── LICENSE
```

---

## Scoring at a glance

### DAT (implemented)

* **Score:** Average cosine distance between all pairs of words (first 7 valid words from 10 provided) × 100. Range: 0-200.

### QD-DP (planned)

* **Orthogonality:** Calculates average pairwise distance between all sub-questions
* **Coverage:** greedy matching of sub-questions to topic-map items -> average matched similarity.
* **Redundancy:** Count percentage of question pairs that are too similar (`share of pairs with `cos_sim > τ`).
* **Composite (tunable):** e.g., `0.4*coverage + 0.4*orthogonality − 0.2*redundancy`.

### TD-DP (planned)

* **Approach diversity:** dispersion across `approaches.summary`.
* **Actionability:** LLM/human-judged rubric for executability and completeness.
* **Risk assessment:** LLM/human-judged rubric for risk quality and mitigation.

All tracks report effect sizes and apply multiple-comparison correction for honest leaderboards.

---

## Statistical features

* **Multiple-comparison control:** Holm (default), Bonferroni, Benjamini–Hochberg.
* **Effect sizes:** Cohen’s *d* (small≈0.2, medium≈0.5, large≈0.8, very large>1.0).
* **Small-sample handling:** warnings & rug plots; bootstrap CIs (planned).

---

## Current results (DAT)

*Snapshot (Aug 2025). Higher is better.*

| Model        | Mean DAT | Std Dev | n  |
| ------------ | -------- | ------- | -- |
| llama3.2:3b  | 85.1     | 3.2     | 9  |
| gpt-5-mini   | 80.8     | 4.5     | 3  |
| gpt-5-nano   | 77.1     | 0.9     | 8  |
| gpt-4.1-nano | 75.4     | 4.4     | 3  |
| Qwen3-4B     | 72.7     | 4.1     | 44 |

---

## Roadmap

### Immediate (In Progress)
- Fix CLI argument parsing for batch experiments
- Implement batch runner for systematic comparisons
- Create model leaderboard with statistical significance

### Near-term
- Phoenix/Arize integration for telemetry collection
- Support for Anthropic Claude models
- Export visualizations to interactive HTML
- **QD/TD Divergent Bench implementation:**
  - Query Decomposition with diversity-priming (QD-DP)
  - Task Decomposition with diversity-priming (TD-DP)
  - Orthogonality/coverage/redundancy metrics
  - A/B comparisons vs standard decomposition

### Long-term
- RL environment for divergent thinking training
- **Diverge → Decompose → Converge** track (full creative pipeline)
- Multi-language DAT (beyond English)
- Alternative creativity metrics (e.g., Alternative Uses Task, Remote Associates Test)

---

## Contributing

Contributions welcome! Especially:

* New providers, strategies, and prompt variants
* Better small-sample statistics
* Visualization polish
* **Datasets**: topic maps (QD) and plan rubrics (TD)
* Lightweight judge rubrics/parsers
* Bug fixes and performance improvements

Please ensure:

1. Tests: `pytest tests/`
2. Style: `black . && ruff check .`
3. Types: `mypy divergent_bench/`

---

## References

* Olson et al. (2021). *Naming unrelated words predicts creativity.* PNAS.
* Pennington et al. (2014). *GloVe: Global Vectors for Word Representation.*
* Cohen (1988). *Statistical Power Analysis for the Behavioral Sciences.*

---

## License

MIT - see `LICENSE`.

## Citation

```bibtex
@software{divergent_bench,
  title  = {DAT Bench: Divergent Thinking and Diversity-Primed Decomposition Benchmarks for LLMs},
  author = {Nason Zikayo},
  year   = {2025},
  url    = {https://github.com/NasonZ/DAT-Bench}
}
```
