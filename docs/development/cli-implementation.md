# CLI & Batch Runner Implementation Plan

## Overview
Create a production-ready CLI module and batch runner system that integrates properly with the divergent_bench package structure, supports YAML-based configuration, and provides excellent UX.

## Session Goals
1. **Fix broken CLI entry point** in pyproject.toml
2. **Create proper CLI module** with Typer (modern Click alternative)
3. **Implement batch runner** with YAML configuration support
4. **Add progress tracking** and rich output formatting
5. **Ensure backward compatibility** with existing ExperimentRunner

---

## Phase 1: CLI Module Architecture (30 mins)

### 1.1 Create CLI Module Structure
```
divergent_bench/
├── cli/
│   ├── __init__.py         # Main app entry point
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── dat.py          # DAT-specific commands
│   │   ├── batch.py        # Batch experiment commands
│   │   ├── analyze.py      # Analysis commands
│   │   └── config.py       # Configuration management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── output.py       # Rich output formatting
│   │   ├── progress.py     # Progress tracking
│   │   └── validation.py   # Input validation
│   └── app.py              # Main Typer application
```

### 1.2 Main CLI Application (`cli/app.py`)
```python
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
import asyncio
from pathlib import Path

app = typer.Typer(
    name="divergent-bench",
    help="Benchmark for evaluating divergent thinking in LLMs",
    rich_markup_mode="rich",
    add_completion=True
)

console = Console()

# Import subcommands
from .commands import dat, batch, analyze, config

# Register subcommands
app.add_typer(dat.app, name="dat", help="Run DAT experiments")
app.add_typer(batch.app, name="batch", help="Run batch experiments")
app.add_typer(analyze.app, name="analyze", help="Analyze results")
app.add_typer(config.app, name="config", help="Manage configuration")

@app.command()
def version():
    """Show version information."""
    from .. import __version__
    console.print(f"[bold green]divergent-bench[/] version {__version__}")

@app.command()
def list_models():
    """List available models for each provider."""
    # TBD: Implementation details
    pass

@app.command()
def list_strategies():
    """List available DAT strategies with descriptions."""
    # TBD: Implementation details
    pass
```

### 1.3 Fix pyproject.toml Entry Point
```toml
[project.scripts]
divergent-bench = "divergent_bench.cli:main"
# or if using app directly:
divergent-bench = "divergent_bench.cli.app:app"
```

---

## Phase 2: DAT Command Implementation (45 mins)

### 2.1 DAT Commands (`cli/commands/dat.py`)
```python
import typer
from typing import Optional, List
from enum import Enum
from rich.progress import Progress, SpinnerColumn, TextColumn
from ...experiments.runner import ExperimentRunner
from ...config.strategies import DAT_STRATEGIES

app = typer.Typer()

class Provider(str, Enum):
    """Supported LLM providers."""
    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"
    ollama = "ollama"
    deepseek = "deepseek"

@app.command("run")
def run_dat(
    provider: Provider = typer.Option(Provider.openai, help="LLM provider to use"),
    model: Optional[str] = typer.Option(None, help="Specific model (e.g., gpt-4)"),
    strategy: str = typer.Option("none", help="DAT strategy", 
                                 autocompletion=lambda: list(DAT_STRATEGIES.keys())),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    samples: int = typer.Option(10, help="Number of samples to generate"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    structured: bool = typer.Option(True, help="Use structured output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run a single DAT experiment."""
    # TBD: Implementation with progress tracking
    pass

@app.command("compare")
def compare_models(
    providers: List[Provider] = typer.Option(..., help="Providers to compare"),
    strategy: str = typer.Option("none", help="DAT strategy to use"),
    samples: int = typer.Option(5, help="Samples per model"),
    output_format: str = typer.Option("table", help="Output format: table/json/csv"),
):
    """Compare multiple models on the same DAT task."""
    # TBD: Run experiments in parallel, show comparison table
    pass

@app.command("resume")
def resume_experiment(
    results_file: Path = typer.Argument(..., help="Previous results file to resume from"),
    additional_samples: int = typer.Option(10, help="Additional samples to generate"),
):
    """Resume an interrupted experiment."""
    # TBD: Load previous results, continue from where it left off
    pass
```

### 2.2 Progress Tracking Utilities (`cli/utils/progress.py`)
```python
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from contextlib import contextmanager
import asyncio

@contextmanager
def experiment_progress(total_samples: int, description: str = "Running experiment"):
    """Context manager for experiment progress tracking."""
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=Console(),
    ) as progress:
        task = progress.add_task(description, total=total_samples)
        
        def update(n: int = 1):
            progress.update(task, advance=n)
        
        yield update

class LiveExperimentTracker:
    """Live tracking of experiment metrics."""
    def __init__(self):
        self.console = Console()
        self.table = None
        self.results = []
    
    def update(self, iteration: int, score: float, words: List[str]):
        """Update live display with new result."""
        # TBD: Rich table that updates in place
        pass
```

---

## Phase 3: Batch Runner Implementation (1 hour)

### 3.1 Batch Configuration Schema
```yaml
# configs/example_batch.yaml
name: "Multi-Model DAT Benchmark"
description: "Compare multiple models across different strategies"

# Global settings (can be overridden per experiment)
defaults:
  samples: 10
  temperature: 0.7
  output_format: "json"
  save_intermediate: true

# Experiment definitions
experiments:
  - name: "baseline_comparison"
    providers:
      - provider: "openai"
        model: "gpt-4"
      - provider: "anthropic"
        model: "claude-3-5-sonnet"
      - provider: "gemini"
        model: "gemini-1.5-pro"
    strategies: ["none", "etymology", "random"]
    temperatures: [0.5, 0.7, 1.0]
    samples: 20
    
  - name: "ollama_models"
    providers:
      - provider: "ollama"
        models: ["llama3.2:3b", "mistral:7b", "gemma2:9b"]
    strategies: ["none"]
    temperature: 0.7
    samples: 10
    
  - name: "strategy_analysis"
    providers:
      - provider: "openai"
        model: "gpt-4"
    strategies: ["all"]  # Special keyword for all strategies
    temperature: 0.7
    samples: 50

# Analysis configuration
analysis:
  generate_report: true
  include_visualizations: true
  statistical_tests:
    - "mann_whitney"
    - "kruskal_wallis"
  
# Output configuration  
output:
  directory: "results/batch_{timestamp}"
  format: "json"
  include_raw_responses: false
  generate_summary: true
```

### 3.2 Batch Runner Class (`experiments/batch_runner.py`)
```python
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import asyncio
from datetime import datetime
from dataclasses import dataclass
import json

@dataclass
class BatchConfig:
    """Configuration for batch experiments."""
    name: str
    description: str
    defaults: Dict[str, Any]
    experiments: List[Dict[str, Any]]
    analysis: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None

class BatchRunner:
    """Orchestrate multiple experiments from YAML configuration."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        self.start_time = None
        
    def _load_config(self) -> BatchConfig:
        """Load and validate YAML configuration."""
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        # TBD: Validation with Pydantic
        return BatchConfig(**data)
    
    async def run(self, resume_from: Optional[Path] = None):
        """Execute all experiments in the batch."""
        self.start_time = datetime.now()
        
        if resume_from:
            self._load_checkpoint(resume_from)
        
        for exp_config in self.config.experiments:
            await self._run_experiment(exp_config)
            
        if self.config.analysis:
            await self._run_analysis()
            
        self._save_final_results()
    
    async def _run_experiment(self, exp_config: Dict):
        """Run a single experiment configuration."""
        # TBD: Implementation details
        # - Expand provider/model combinations
        # - Handle "all" strategies keyword
        # - Run with progress tracking
        # - Save intermediate results
        pass
    
    def _save_checkpoint(self, experiment_name: str):
        """Save intermediate results for resumption."""
        checkpoint_path = Path(f".checkpoints/{self.config.name}_{experiment_name}.json")
        # TBD: Save current state
        pass
```

### 3.3 Batch Commands (`cli/commands/batch.py`)
```python
import typer
from pathlib import Path
from rich.console import Console
from typing import Optional

app = typer.Typer()
console = Console()

@app.command("run")
def run_batch(
    config: Path = typer.Argument(..., help="Path to YAML configuration file"),
    resume: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
    dry_run: bool = typer.Option(False, help="Validate config without running"),
    parallel: int = typer.Option(1, help="Number of parallel experiments"),
):
    """Run batch experiments from YAML configuration."""
    if dry_run:
        # TBD: Validate and show execution plan
        pass
    else:
        # TBD: Run with BatchRunner
        pass

@app.command("validate")
def validate_config(
    config: Path = typer.Argument(..., help="Configuration file to validate"),
):
    """Validate a batch configuration file."""
    # TBD: Load config, check for errors, show summary
    pass

@app.command("list")
def list_batches(
    results_dir: Path = typer.Option(Path("results"), help="Results directory"),
):
    """List all batch experiment results."""
    # TBD: Scan directory, show table of batches
    pass

@app.command("monitor")
def monitor_batch(
    checkpoint_dir: Path = typer.Option(Path(".checkpoints"), help="Checkpoint directory"),
):
    """Monitor running batch experiments."""
    # TBD: Watch checkpoint files, show live progress
    pass
```

---

## Phase 4: Integration & Polish (45 mins)

### 4.1 Enhanced Output Formatting (`cli/utils/output.py`)
```python
from rich.table import Table
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
import pandas as pd
from typing import List, Dict, Any

class ResultsFormatter:
    """Format experiment results for display."""
    
    def __init__(self):
        self.console = Console()
    
    def format_dat_results(self, results: List[Dict]) -> Table:
        """Create rich table for DAT results."""
        table = Table(title="DAT Experiment Results")
        table.add_column("Iteration", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Words (first 5)", style="green")
        table.add_column("Strategy", style="yellow")
        
        for r in results:
            words_preview = ", ".join(r['words'][:5]) + "..."
            table.add_row(
                str(r['iteration']),
                f"{r['score']:.2f}",
                words_preview,
                r.get('strategy', 'none')
            )
        
        return table
    
    def format_comparison(self, comparison_data: Dict[str, List[float]]) -> Table:
        """Create comparison table for multiple models."""
        # TBD: Statistical comparison table
        pass
    
    def export_results(self, results: Any, format: str, path: Path):
        """Export results in various formats."""
        if format == "json":
            # TBD: JSON export
            pass
        elif format == "csv":
            # TBD: CSV export with pandas
            pass
        elif format == "markdown":
            # TBD: Markdown report
            pass
```

### 4.2 Configuration Management (`cli/commands/config.py`)
```python
import typer
from pathlib import Path
from rich.console import Console
import os

app = typer.Typer()
console = Console()

@app.command("show")
def show_config():
    """Show current configuration from environment."""
    # TBD: Display all relevant env vars and settings
    pass

@app.command("init")
def init_config(
    output: Path = typer.Option(Path(".env"), help="Output path for .env file"),
    interactive: bool = typer.Option(True, help="Interactive setup"),
):
    """Initialize configuration file."""
    # TBD: Interactive setup wizard for API keys, paths, etc.
    pass

@app.command("validate")
def validate_env():
    """Validate environment configuration."""
    # TBD: Check API keys, GloVe paths, etc.
    pass
```

---

## Phase 5: Testing & Documentation (30 mins)

### 5.1 CLI Tests
```python
# tests/test_cli.py
from typer.testing import CliRunner
from divergent_bench.cli.app import app

runner = CliRunner()

def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "divergent-bench" in result.stdout

def test_dat_run_command():
    result = runner.invoke(app, ["dat", "run", "--help"])
    assert result.exit_code == 0
    
# TBD: More comprehensive tests
```

### 5.2 Update README with CLI Usage
```markdown
## CLI Usage

### Quick Start
```bash
# Install with CLI extras
pip install -e ".[cli]"

# Initialize configuration
divergent-bench config init

# Run single DAT experiment
divergent-bench dat run --provider openai --strategy none --samples 10

# Compare multiple models
divergent-bench dat compare --providers openai anthropic --samples 5

# Run batch experiments
divergent-bench batch run configs/benchmark.yaml

# Monitor running batch
divergent-bench batch monitor

# Analyze results
divergent-bench analyze results/
```

### Batch Configuration
Create a YAML file defining your experiments:
[Include example YAML]

### Advanced Usage
[TBD: Document advanced features]
```

---

## Implementation Order

1. **Fix pyproject.toml** (5 mins)
2. **Create CLI module structure** (10 mins)
3. **Implement basic app.py** (15 mins)
4. **Port run_dat.py functionality** (20 mins)
5. **Add progress tracking** (15 mins)
6. **Implement BatchConfig with Pydantic** (20 mins)
7. **Create BatchRunner core logic** (30 mins)
8. **Add batch commands** (15 mins)
9. **Implement output formatting** (15 mins)
10. **Add tests and documentation** (15 mins)

## Key Decisions to Make

1. **Typer vs Click**: Typer is more modern with better type hints - Recommend Typer
2. **Async everywhere**: Keep async for all experiment runs
3. **Progress tracking**: Use Rich for beautiful terminal output
4. **Configuration format**: YAML for human readability
5. **Checkpoint format**: JSON for easy parsing
6. **Parallel execution**: Use asyncio.gather with semaphore for rate limiting

## Dependencies to Add

```toml
[project.dependencies]
# Add to existing
typer = ">=0.9.0"
rich = ">=13.0.0"  # Already present
pyyaml = ">=6.0"

[project.optional-dependencies]
cli = [
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "pandas>=2.0.0",  # For data export
]
```

## Success Criteria

1. ✅ `divergent-bench` command works from anywhere
2. ✅ All existing functionality accessible via CLI
3. ✅ Batch experiments run from YAML config
4. ✅ Progress tracking with Rich
5. ✅ Results saved in multiple formats
6. ✅ Experiments can be resumed if interrupted
7. ✅ Clear documentation and help text

## TBD Items for Discussion

1. **Parallel execution strategy**: How many concurrent API calls?
2. **Rate limiting**: Per-provider limits?
3. **Result aggregation**: What statistics to include?
4. **Visualization**: Include plots in CLI or separate command?
5. **Cloud storage**: Support for S3/GCS result uploads?
6. **Notification**: Slack/email when batch completes?
7. **Cost tracking**: Estimate and track API costs?