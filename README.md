# DAT-Bench 🧠

DAT-Bench instruments divergent word choice in LLMs.

It starts with the Divergent Association Task (DAT): generate 10 nouns that are as semantically distant from each other as possible. Responses are scored with GloVe 840B cosine distances and exposed as `verifiers`-compatible reward signals for evaluation and RL training.

The first track is implemented. Decomposition tracks are planned.

## Why this exists

Most model benchmarks reward correctness, instruction following, or reasoning over fixed targets. DAT-Bench asks a narrower complementary question:

> Can a rewardable measure of divergent thinking improve how models explore, decompose, and cover open-ended tasks?

DAT is the first instrument. It gives a cheap, repeatable signal for semantic divergence: not whether the model knows an answer, but whether it can move across distant regions of word space while staying within task constraints.

The planned decomposition tracks test whether that diversity objective carries over from word choice to task structure.

## Current status

Implemented:

- DAT scorer using GloVe 840B cosine distances
- Composite `verifiers` rubric for evaluation and RL training
- Training environment with XML output and format reward
- Eval environment with Pydantic structured output
- Multi-provider experiment runner: OpenAI, Anthropic, Ollama, OpenRouter
- Prompting strategies for DAT trials
- Statistical visualisations: ridge plots, significance matrices, effect sizes
- Hand-curated fixtures for pipeline validation

Planned:

- QD-DP: query decomposition with an explicit diversity objective
- TD-DP: task decomposition with an explicit diversity objective
- Word-chain trajectory benchmark for graph-navigation strategy analysis

## Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv)
- GloVe 840B embeddings
- A word list compatible with the scorer

Set the embedding and word-list paths with environment variables:

```bash
export GLOVE_PATH=/path/to/glove.840B.300d.txt
export WORDS_PATH=/path/to/words.txt
```

The scorer also checks:

```text
data/embeddings/glove.840B.300d.txt
data/words.txt
```

## Install

```bash
git clone https://github.com/NasonZ/DAT-Bench.git
cd DAT-Bench

uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

For development:

```bash
uv pip install -e ".[dev]"
pytest
```

## Run DAT experiments

OpenAI:

```bash
uv run python scripts/run_dat.py \
  --provider openai \
  --model gpt-5-mini \
  --strategy competitive \
  --samples 20
```

Ollama:

```bash
uv run python scripts/run_dat.py \
  --provider ollama \
  --model llama3.2:3b \
  --strategy random \
  --samples 15
```

OpenRouter:

```bash
export OPENROUTER_API_KEY="..."

uv run python scripts/run_dat.py \
  --provider openrouter \
  --model meta-llama/llama-3.1-8b-instruct \
  --samples 10
```

## Use as a verifiers environment

DAT-Bench exposes two environment loaders.

```python
from environments.dat_bench.dat_bench import load_environment, load_eval_environment

# Training: XMLParser + format reward signal
train_env = load_environment(strategy="competitive", num_examples=50)

# Eval: Pydantic structured output, deterministic parsing
eval_env = load_eval_environment(strategy="DAT_instructions", num_examples=30)
```

Training uses XML because format compliance is part of the reward signal. Eval uses structured output so parsing is handled by the provider instead of the model.

The rubric exposes these reward signals:

| Signal | Weight | Purpose |
|---|---:|---|
| `creativity_reward` | 1.0 | Normalised DAT score; primary reward |
| `validity_reward` | 0.2 | Fraction of generated words in the GloVe vocabulary |
| `format_reward` | 0.1 | XML format compliance; training only |
| `raw_dat_score` | 0.0 | Raw DAT score for analysis |
| `valid_word_count` | 0.0 | Count of valid words for analysis |

## Score words directly

```python
from divergent_bench import DATScorer

scorer = DATScorer()
score = scorer.dat([
    "whale", "hammer", "symphony", "cactus", "glacier",
    "umbrella", "passport", "volcano", "whistle", "tapestry",
])

print(f"DAT score: {score:.1f}")
```

## DAT scoring

DAT score is the average cosine distance between all pairs of the first 7 valid unique words, multiplied by 100.

```text
score = mean(pairwise_cosine_distance(first_7_valid_words)) * 100
```

Higher scores indicate greater semantic distance between the generated words. The reward normalises raw DAT scores to `[0, 1]` with a linear map over the empirical range `[40, 100]`.

This is not a general creativity score. It is a specific measure of semantic divergence under a constrained word-generation task.

## Prompting strategies

| Strategy | Temperature | Description |
|---|---:|---|
| `none` | 0.7 | Minimal instructions; baseline |
| `competitive` | 0.7 | Prize framing with tips for maximising distance |
| `DAT_instructions` | 0.7 | Full task context from the original DAT framing |
| `random` | 1.0 | Explicit randomness instruction |

## Decomposition tracks

DAT is the first track because it provides a concrete reward for divergence. The planned decomposition tracks ask whether the same pressure helps models cover more of an open problem.

### QD-DP: Query Decomposition, Diversity-Primed

Given a complex question, generate orthogonal sub-questions, answer each, and synthesise the result.

Candidate metrics:

- orthogonality between sub-questions
- coverage against a topic map
- redundancy penalty
- answer quality after synthesis

### TD-DP: Task Decomposition, Diversity-Primed

Given a complex task, generate distinct top-level approaches, select one, and build a plan tree.

Candidate metrics:

- approach diversity
- actionability
- risk coverage
- plan quality via judge rubric

Both tracks should include standard non-diversity-primed baselines and report effect sizes with multiple-comparison correction.

## Word-chain trajectory benchmark

A word-chain benchmark is planned as a separate trajectory task.

DAT scores a final set of words. Word-chain would score a path through a fully observable word graph: each step changes one word by insertion, deletion, or substitution. This makes model strategy visible, not just final performance.

Candidate signals:

- chain length
- valid transition rate
- bridge pattern frequency
- turning-angle distribution
- word-length oscillation
- local graph density along the path

The design note is in [`docs/research/word-chain-trajectory-benchmark-design.md`](docs/research/word-chain-trajectory-benchmark-design.md).

## Project structure

```text
DAT-Bench/
├── divergent_bench/
│   ├── dat/                          # DAT scorer
│   ├── rubrics/                      # verifiers-compatible reward rubrics
│   ├── config/                       # prompting strategies and model configs
│   ├── data/                         # fixtures for validation
│   ├── experiments/                  # experiment runner
│   ├── llm/                          # provider clients
│   ├── metrics/                      # DSI and Lempel-Ziv metrics
│   ├── visualization/                # plots, styles, loaders
│   ├── decomposition/                # planned QD/TD implementations
│   └── utils/
├── environments/
│   └── dat_bench/                    # verifiers environment definition
├── configs/
│   └── eval/                         # eval configuration
├── docs/
│   ├── research/                     # alignment plans and benchmark designs
│   ├── api/                          # structured-output notes
│   └── development/                  # roadmap and technical notes
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
├── pyproject.toml
└── LICENSE
```

## Statistical features

- Multiple-comparison control: Holm by default, plus Bonferroni and Benjamini-Hochberg
- Effect sizes: Cohen's *d* with standard thresholds
- Small-sample handling: adaptive rendering, CI capping, low-*n* warnings
- Colourblind-safe palette validated with Coblis

## Current limitations

- The DAT scorer requires local GloVe 840B embeddings and a word list.
- Decomposition tracks are not implemented yet.
- Word-chain is a design document, not an implemented environment.
- The packaged console entry point in `pyproject.toml` points to `divergent_bench.cli:app`; that module is not currently present. Use `scripts/run_dat.py` for now.

## Development

Run tests:

```bash
pytest
```

Run a focused test:

```bash
pytest tests/unit/test_dat_scorer.py
```

Run integration tests:

```bash
pytest tests/integration
```

Useful docs:

- [`docs/research/verifiers-alignment-plan.md`](docs/research/verifiers-alignment-plan.md)
- [`docs/research/word-chain-trajectory-benchmark-design.md`](docs/research/word-chain-trajectory-benchmark-design.md)
- [`docs/development/ROADMAP.md`](docs/development/ROADMAP.md)
- [`docs/api/structured-output.md`](docs/api/structured-output.md)

## References

- Olson et al. (2021). *Naming unrelated words predicts creativity.* PNAS.
- Pennington et al. (2014). *GloVe: Global Vectors for Word Representation.*
- Cohen (1988). *Statistical Power Analysis for the Behavioral Sciences.*

## Citation

```bibtex
@software{divergent_bench,
  title  = {DAT-Bench: Divergent Thinking Benchmarks for LLMs},
  author = {Nason Zikayo},
  year   = {2026},
  url    = {https://github.com/NasonZ/DAT-Bench}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
