# DAT Bench 🧠

**A benchmark for measuring divergent thinking and creativity in language models using the Divergent Association Task (DAT).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/uv-latest-green.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

`divergent_bench` implements the **Divergent Association Task (DAT)** to measure creative thinking in language models. The DAT asks models to generate 10 words that are as different from each other as possible, then scores them based on semantic distance.

### Key Features

- 📊 **Production-ready visualizations** with statistical rigor (Cohen's d, multiple comparison correction)
- 🤖 **Multi-provider support**: OpenAI, Ollama, OpenRouter, and local models
- 📈 **Advanced analytics**: Ridge plots, statistical heatmaps, word frequency analysis
- 🔬 **Statistically sound**: Proper handling of small samples, outliers, and multiple comparisons
- 🎯 **Strategy testing**: Compare different prompting strategies (random, thesaurus, etymology, opposites)

---

## Quick Start

### Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) package manager
- GloVe embeddings for scoring (auto-downloaded)

### Installation

```bash
# Clone the repository
git clone https://github.com/NasonZ/DAT-Bench.git
cd DAT-Bench

# Create virtual environment with UV
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Download GloVe embeddings (required for DAT scoring)
# These will be downloaded automatically on first run if not present
```

## Usage

### Running DAT Experiments

```bash
# Run with OpenAI GPT
uv run python scripts/run_dat.py --provider openai --model gpt-5-mini --strategy random --samples 10

# Run with Ollama local model
uv run python scripts/run_dat.py --provider ollama --model llama3.2:3b --strategy thesaurus --samples 20

# Run with custom temperature
uv run python scripts/run_dat.py --provider ollama --model qwen:4b --strategy random --temperature 0.7 --samples 15

# Run with OpenRouter
export OPENROUTER_API_KEY="your-key"
uv run python scripts/run_dat.py --provider openrouter --model meta-llama/llama-3.1-8b-instruct --samples 10
```

### Visualizing Results

```bash
# Generate all visualizations from results directory
uv run python scripts/test_visualization.py

# Visualizations are saved to visualizations/ directory:
# - test_ridge_plot.png: Distribution comparison with rug plots for small samples
# - test_statistical_heatmap.png: Pairwise comparisons with effect sizes
# - test_word_frequency_stacked.png: Word usage patterns across models
```

### Using as a Library

```python
from divergent_bench.dat.scorer import DATScorer
from divergent_bench.llm import create_provider
from divergent_bench.experiments.runner import DATExperiment

# Initialize provider and scorer
provider = create_provider("openai", model="gpt-4-turbo")
scorer = DATScorer()

# Run experiment
experiment = DATExperiment(provider, scorer)
results = experiment.run(strategy="random", num_samples=10)

# Analyze results
print(f"Mean DAT score: {results['mean_score']:.2f}")
print(f"Best words: {results['best_sample']['words']}")
```

---

## Visualization Examples

### Ridge Plots
- Shows score distributions across models
- Rug plots display actual data points for n≤10
- Visual warnings for small samples (red: n<10, orange: n<30)
- Robust x-axis limits using 1st-99th percentiles

### Statistical Heatmaps
- Dual panel: performance bars + significance matrix
- Cohen's d effect sizes or t-statistics
- Multiple comparison correction (Holm, Bonferroni, FDR)
- Clear significance indicators (*, **, ***)

### Word Frequency Analysis
- Stacked bar charts showing model attribution
- Multiple normalization modes (per-word, total, raw counts)
- Visual insights into convergent vs divergent patterns

---

## Project Structure

```
divergent_bench/
├── divergent_bench/           # Main package
│   ├── dat/                  # DAT implementation
│   │   ├── scorer.py         # Semantic distance scoring
│   │   └── prompts.py        # Strategy-specific prompts
│   ├── llm/                  # LLM integrations
│   │   ├── providers.py      # OpenAI, Ollama, OpenRouter
│   │   ├── base.py          # Base provider interface
│   │   └── responses.py     # Response parsing
│   ├── experiments/          # Experiment runners
│   │   └── runner.py        # DAT experiment orchestration
│   └── visualization/        # Analysis and plotting
│       ├── loader.py        # Result loading and preparation
│       ├── plots.py         # Ridge, heatmap, word frequency
│       └── styles.py        # Consistent visual styling
├── scripts/                  # CLI tools
│   ├── run_dat.py           # Main experiment runner
│   └── test_visualization.py # Visualization generator
├── results/                  # Experiment outputs (JSON)
└── visualizations/          # Generated plots (PNG)
```

## Statistical Features

### Multiple Comparison Correction
- **Holm** (default): Controls FWER, more powerful than Bonferroni
- **Bonferroni**: Conservative FWER control
- **Benjamini-Hochberg**: FDR control for many comparisons

### Effect Size Interpretation
- **d ≈ 0.2**: Small effect
- **d ≈ 0.5**: Medium effect
- **d ≈ 0.8**: Large effect
- **d > 1.0**: Very large effect

### Sample Size Handling
- **n < 10**: Unreliable results with red warning + rug plot
- **n < 30**: Limited power with orange warning
- **n ≥ 30**: Generally reliable for parametric tests

---

## Current Results

Latest benchmark results (as of 2025-08):

| Model | Mean DAT Score | Std Dev | n |
|-------|---------------|---------|---|
| llama3.2:3b | 85.1 | 3.2 | 9 |
| gpt-5-mini | 80.8 | 4.5 | 3 |
| gpt-5-nano | 77.1 | 0.9 | 8 |
| gpt-4.1-nano | 75.4 | 4.4 | 3 |
| Qwen3-4B | 72.7 | 4.1 | 44 |

*Higher scores indicate more divergent/creative thinking*

---

## Roadmap

### Immediate (In Progress)
- [ ] Fix CLI argument parsing for batch experiments
- [ ] Implement batch runner for systematic comparisons
- [ ] Add bootstrap confidence intervals for small samples
- [ ] Create model leaderboard with statistical significance

### Near-term
- [ ] Phoenix arize for better telemetry collection
- [ ] Support for Anthropic Claude models
- [ ] Export visualizations to interactive HTML

### Long-term
- [ ] Multi-language DAT (beyond English)
- [ ] Alternative creativity metrics
- [ ] Integration with other creativity benchmarks
- [ ] Real-time streaming evaluation

---

## Contributing

We welcome contributions! Key areas:

- Adding new LLM providers
- Improving statistical methods
- Enhancing visualizations
- Documentation and examples
- Bug fixes and optimizations

Please ensure:
1. Tests pass: `pytest tests/`
2. Code is formatted: `black . && ruff check .`
3. Types check: `mypy divergent_bench/`

---

## References

- Olson et al. (2021). *Naming unrelated words predicts creativity.* PNAS.
- Pennington et al. (2014). *GloVe: Global Vectors for Word Representation.*
- Cohen (1988). *Statistical Power Analysis for the Behavioral Sciences.*

## License

MIT License. See `LICENSE` for details.

## Citation

```bibtex
@software{dat_bench,
  title = {DAT-Bench: Statistical Benchmark for Divergent Thinking in LLMs},
  author = {Nason Zikayo},
  year = {2024},
  url = {https://github.com/NasonZ/DAT-Bench}
}
```