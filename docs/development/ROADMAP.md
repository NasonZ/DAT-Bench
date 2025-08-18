# Divergent Bench - Vision & Roadmap

## Project Vision

`divergent_bench` aims to be the definitive benchmark for measuring creative and divergent thinking in language models, going beyond traditional accuracy metrics to capture the ability to "think differently."

---

## Core Architecture (From Original Plans)

### 1. CLI Module Structure (Critical - Not Yet Implemented)

```python
divergent_bench/
├── cli/
│   ├── __init__.py
│   ├── main.py              # Main CLI entry point with Typer
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── run.py          # Single experiment runs
│   │   ├── batch.py        # Batch experiment orchestration
│   │   ├── analyze.py      # Results analysis and visualization
│   │   └── leaderboard.py  # Generate model rankings
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py       # YAML configuration loading
│   │   └── validator.py    # Configuration validation
│   └── output/
│       ├── __init__.py
│       ├── progress.py     # Rich progress bars
│       └── tables.py       # Formatted result tables
```

### 2. Batch Runner System (Essential for Systematic Comparisons)

**YAML Configuration Support:**
```yaml
# experiments/baseline.yaml
name: "Baseline Model Comparison"
description: "Compare all models with default settings"

models:
  - provider: openai
    model: gpt-4-turbo
  - provider: ollama
    model: llama3.2:3b
  - provider: anthropic
    model: claude-3-sonnet

parameters:
  strategies: [random, thesaurus, etymology, opposites]
  temperatures: [0.5, 0.7, 1.0]
  samples_per_condition: 30

output:
  format: json
  directory: results/baseline/
  visualizations: true
```

**Batch Execution Engine:**
```python
class BatchRunner:
    def __init__(self, config_path: Path):
        self.config = load_config(config_path)
        self.executor = ProcessPoolExecutor(max_workers=4)
        
    async def run(self):
        # Grid search over all parameter combinations
        # Parallel execution with progress tracking
        # Automatic retry on failures
        # Resume from checkpoint on interruption
```

### 3. Analysis Pipeline (For Research Insights)

```python
divergent_bench/
├── analysis/
│   ├── __init__.py
│   ├── aggregator.py       # Combine results across experiments
│   ├── statistics.py       # Advanced statistical analysis
│   ├── patterns.py         # Identify patterns in divergent thinking
│   └── reports.py          # Generate research reports
```

---

## Implementation Phases

### Phase 1: CLI Foundation (Immediate Priority)
**Goal:** Professional CLI interface for running experiments

```bash
# Single experiment
divergent-bench run --provider openai --model gpt-4 --samples 30

# Batch experiments
divergent-bench batch experiments/baseline.yaml

# Generate visualizations
divergent-bench analyze results/ --output-dir reports/

# Create leaderboard
divergent-bench leaderboard --metric dat_score --format markdown
```

**Key Features:**
- Rich terminal output with progress bars
- Resumable experiments (checkpoint support)
- Parallel execution for multiple models
- Automatic result aggregation

### Phase 2: Batch Orchestration
**Goal:** Systematic comparison infrastructure

- **Grid Search**: Automatically test all parameter combinations
- **Smart Scheduling**: Optimize API usage and costs
- **Failure Recovery**: Resume from last successful checkpoint
- **Resource Management**: Respect rate limits, manage quotas

### Phase 3: Advanced Analysis
**Goal:** Extract research insights from results

- **Pattern Recognition**: Identify which strategies work for which models
- **Statistical Significance**: Automated significance testing with corrections
- **Trend Analysis**: Track model improvements over time
- **Cross-Model Insights**: Find universal vs model-specific patterns

### Phase 4: Research Tools
**Goal:** Support academic research on AI creativity

- **Hypothesis Testing**: Built-in statistical tests for research questions
- **Publication Support**: Generate publication-ready figures and tables
- **Dataset Creation**: Export curated datasets for further research
- **Reproducibility**: Full experiment tracking and version control

---

## Technical Implementation Details

### CLI Implementation (Using Typer)

```python
# cli/main.py
import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer(
    name="divergent-bench",
    help="Benchmark for divergent thinking in LLMs",
    rich_markup_mode="rich"
)

@app.command()
def run(
    provider: str = typer.Option(..., help="LLM provider"),
    model: str = typer.Option(..., help="Model name"),
    strategy: str = typer.Option("random", help="Word generation strategy"),
    samples: int = typer.Option(10, help="Number of samples"),
    temperature: float = typer.Option(1.0, help="Temperature"),
    output: Path = typer.Option("results/", help="Output directory")
):
    """Run a single DAT experiment."""
    with Progress() as progress:
        task = progress.add_task("[cyan]Running experiment...", total=samples)
        # Implementation here
```

### Batch Configuration Schema

```python
# config/schema.py
from pydantic import BaseModel, validator
from typing import List, Optional

class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

class ExperimentConfig(BaseModel):
    name: str
    description: Optional[str]
    models: List[ModelConfig]
    parameters: ParameterGrid
    output: OutputConfig
    
    @validator('models')
    def validate_models(cls, v):
        # Ensure all models are available
        return v
```

### Progress Tracking System (Optional)

```python
# output/progress.py
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live

class ExperimentTracker:
    def __init__(self, total_experiments: int):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        
    def update(self, model: str, score: float):
        # Update live dashboard with results
        # Show running statistics
        # Display current best/worst
```

---

## Query Decomposition Extension (Future)

### Beyond Simple Word Generation

```python
divergent_bench/
├── decomposition/
│   ├── __init__.py
│   ├── strategies/
│   │   ├── hierarchical.py  # Break into sub-questions
│   │   ├── perspective.py   # Multiple viewpoints
│   │   └── synthesis.py     # Combine insights
│   ├── scoring/
│   │   ├── diversity.py     # Measure conceptual spread
│   │   ├── coverage.py      # Topic coverage metrics
│   │   └── coherence.py     # Synthesis quality
```

**Example Task:**
```python
query = "What drives the UK housing crisis?"

# Decompose into sub-questions
subquestions = decomposer.decompose(query)
# → Economic factors? Policy decisions? Demographics? Supply chain?

# Generate diverse perspectives
perspectives = decomposer.explore(subquestions)
# → Economist view, resident view, developer view, policy maker view

# Synthesize comprehensive answer
synthesis = decomposer.synthesize(perspectives)
# → Coherent narrative combining all insights
```

---

## Integration Roadmap

### Near-term Integrations
1. **Anthropic Claude**: Native support for Claude models
2. **Google Gemini**: Integration via API
3. **Cohere**: Command models for comparison
4. **Local Models**: Expanded Ollama support

### Platform Integrations
1. **Weights & Biases**: Experiment tracking
2. **HuggingFace Hub**: Model registry and datasets
3. **GitHub Actions**: Automated benchmarking
4. **Paperspace/Modal**: Cloud execution

### Research Integrations
1. **Academic Benchmarks**: Compare with human creativity scores
2. **Psychological Tests**: Align with established creativity measures
3. **Cross-Domain**: Extend to visual, musical creativity

---

## Success Metrics

### Phase 1 Complete When:
- [ ] CLI supports all current functionality
- [ ] Batch experiments run reliably
- [ ] Progress tracking works smoothly
- [ ] Results aggregate automatically

### Phase 2 Complete When:
- [ ] YAML configuration fully supported
- [ ] Grid search works efficiently
- [ ] Checkpoint/resume implemented
- [ ] Resource limits respected

### Phase 3 Complete When:
- [ ] Statistical insights automated
- [ ] Pattern recognition working
- [ ] Research reports generated
- [ ] Publication-ready outputs

---

## Why This Matters

### Research Impact
- First systematic benchmark for LLM creativity
- Enables comparison across model families
- Identifies strategies that enhance divergent thinking
- Provides insights into AI cognitive flexibility

### Practical Applications
- Model selection for creative tasks
- Prompt engineering for innovation
- Understanding model limitations
- Guiding model development

### Academic Contributions
- Reproducible creativity measurement
- Statistical framework for divergent thinking
- Dataset for creativity research
- Baseline for future work

---

## Next Steps (Immediate)

1. **Implement CLI Module**
   - Set up Typer application structure
   - Create command handlers
   - Add Rich progress tracking

2. **Build Batch Runner**
   - YAML configuration loader
   - Parameter grid generator
   - Parallel execution engine

3. **Enhance Analysis**
   - Automated leaderboard generation
   - Statistical significance testing
   - Pattern recognition algorithms

---

## The Grand Vision

`divergent_bench` will become the standard tool for measuring and understanding creative thinking in AI systems, providing researchers and practitioners with:

1. **Quantitative metrics** for creativity assessment
2. **Systematic comparison** across models and strategies
3. **Research insights** into AI divergent thinking
4. **Practical guidance** for creative AI applications

This isn't just about measuring word distances - it's about understanding how AI systems can think differently, generate novel ideas, and contribute to creative problem-solving.