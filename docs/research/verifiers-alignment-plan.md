# Verifiers Deep Dive & DAT-Bench Alignment Plan

**Date:** 2026-03-13
**Purpose:** Consolidated understanding of verifiers architecture + concrete plan to align DAT-Bench for evaluation, analysis, and fine-tuning.

---

## Part 1: Verifiers Architecture (What I Learned)

### Core Mental Model

Verifiers is built around one central idea: **an Environment contains everything needed to run and score a model on a task**. The framework handles the orchestration (rollouts, concurrency, checkpointing, saving) — you just define the task, the scoring, and optionally the interaction protocol.

```
Environment = Dataset + Interaction Protocol + Rubric
```

### Type System Foundation (`verifiers/types.py`)

The type system is provider-agnostic Pydantic models throughout:

**Messages:**
- `SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolMessage` — all extend `CustomBaseModel` (Pydantic with dict-like access via `__getitem__`)
- `AssistantMessage` has: `content`, `reasoning_content`, `thinking_blocks`, `tool_calls`
- `Messages = list[Message]`

**Response chain:**
- `Response` wraps `ResponseMessage` (extends `AssistantMessage` with `finish_reason`, `is_truncated`, `tokens`)
- `ResponseTokens` carries prompt/completion IDs, masks, logprobs, routed experts

**Data flow types:**
- `RolloutInput` = `{prompt, example_id, task, ?answer, ?info}` — what goes into a rollout
- `State` = mutable dict tracking everything during a rollout. Has forwarding: `state["answer"]` looks in `state["input"]["answer"]`
- `TrajectoryStep` = `{prompt, completion, response, tokens, reward, advantage, is_truncated, trajectory_id, extras}`
- `RolloutOutput` = serialized result. Required: `example_id, task, prompt, completion, reward, timing, is_completed, is_truncated, metrics`
- `GenerateOutputs` = `{outputs: list[RolloutOutput], metadata: GenerateMetadata}`

**Key insight:** `State` is a dict subclass with INPUT_FIELDS forwarding. When you do `state["prompt"]`, it checks `state["input"]["prompt"]` first. This means reward functions can just ask for `completion, answer` by name and the rubric handles the plumbing.

### Environment Hierarchy

```
Environment (ABC)
└── MultiTurnEnv (rollout loop, stop conditions, trajectory tracking)
    ├── SingleTurnEnv (max_turns=1, env_response raises NotImplementedError)
    └── ToolEnv (tool calling + execution)
        └── StatefulToolEnv (per-rollout state injected into tools)
```

**SingleTurnEnv is just `MultiTurnEnv(max_turns=1)`** — the whole framework runs through the same rollout loop regardless.

### The Rollout Loop (MultiTurnEnv.rollout)

```python
state = await self.init_state(input, client, model, sampling_args)
state = await self.setup_state(state)              # subclass hook
while not await self.is_completed(state):          # checks all @vf.stop methods
    prompt_messages = await self.get_prompt_messages(state)
    response = await self.get_model_response(state, prompt_messages)
    await self.add_model_response(state, prompt_messages, response)  # → TrajectoryStep
await self.render_completion(state)                # assemble state["completion"]
```

**Stop conditions** are methods decorated with `@vf.stop`, checked in priority order after each model turn. Built-in: `has_error` (priority=100), `prompt_too_long`, `max_turns_reached`, `has_final_env_response`.

**Cleanup** via `@vf.cleanup` (per-rollout), **teardown** via `@vf.teardown` (process exit).

### `init_state` Details

Creates State dict with:
- `input`: deepcopy of RolloutInput
- `prompt`: normalized Messages
- `client`, `model`, `sampling_args`
- `is_completed=False`, `is_truncated=False`
- `tool_defs`: resolved from info or env
- `trajectory=[]`, `completion=None`
- `reward=None`, `metrics=None`, `error=None`
- `timing`: RolloutTiming (generation_ms, scoring_ms, total_ms)
- `trajectory_id`: UUID hex
- Usage tracker

### Dataset Handling

Datasets are HuggingFace `Dataset` objects. The environment ensures columns:
- `example_id` (int, auto-generated if missing)
- `prompt` (Messages, built from `question` + `system_prompt` if not present)
- `task` (string, defaults to `env_id`)
- `answer` (optional, for scoring)
- `info` (optional, structured metadata)

Datasets can be raw `Dataset` (eagerly loaded) or `DatasetBuilder` callables (lazily loaded on first access).

### Rubric System (`rubrics/rubric.py`)

Rubrics hold reward functions + weights. Score = weighted sum.

**Individual reward functions:**
```python
async def my_reward(completion, answer, **kwargs) -> float:
```
Can request any of: `completion, prompt, answer, state, task, info` + any class_objects (like `parser`, `judge`).

**Group reward functions** (detected by plural params: `completions, answers, states`):
```python
async def diversity(completions) -> list[float]:
```

**Scoring flow:**
1. `score_group(states)` — iterates funcs in order
2. Individual funcs: `asyncio.gather` across states (parallel per-state)
3. Group funcs: called once with all states
4. Final: `state["reward"] = weighted_sum`, `state["metrics"] = {func_name: score}`
5. Also computes `state["advantage"] = reward - mean_reward` for RL

**Metrics** = reward funcs with `weight=0` (tracked but don't affect reward).

### Client Abstraction (`clients/client.py`)

Abstract `Client[ClientT, MessagesT, ResponseT, ToolT]`:
- `setup_client(config) → native client`
- `to_native_prompt(Messages) → (native_format, extra_kwargs)`
- `to_native_tool(Tool) → native_format`
- `get_native_response(prompt, model, sampling_args, tools) → native_response`
- `raise_from_native_response(response)` — raises `vf.Error` subclasses
- `from_native_response(response) → vf.Response`
- `get_response()` — orchestrates the above chain

Implementations: `OpenAIChatCompletionsClient`, `OpenAIChatCompletionsTokenClient`, `AnthropicMessagesClient`.

### Evaluation Pipeline

CLI: `prime eval run <env_id_or_toml> -m <model> -n <num_examples> -r <rollouts_per_example>`

Flow:
1. Import env module, call `load_environment(**env_args)`
2. Get eval dataset (or train dataset fallback)
3. Create RolloutInputs from dataset rows
4. `environment.generate(inputs, client, model, ...)` runs rollouts with concurrency
5. Scores via rubric (grouped by example_id by default)
6. Saves `results.jsonl` + `metadata.json`
7. Supports resume from checkpoint

### Training Integration

Verifiers environments produce `RolloutOutput` with all data needed for GRPO:
- `prompt` + `completion` = the training pair
- `reward` = the RL signal
- `trajectory` with `TrajectoryStep` objects containing `tokens` (prompt_ids, completion_ids, logprobs)
- `advantage` = reward relative to group mean

Training happens via:
1. **Hosted Training** (Prime Intellect) — TOML config pointing at env
2. **prime-rl** — open-source async RL trainer
3. **Other trainers** (Tinker, SkyRL, rLLM) — via adapter patterns

### How Real Environments Look

**gsm8k (simplest):**
```python
def load_environment(system_prompt, num_train_examples, num_eval_examples):
    dataset = load_example_dataset("gsm8k", split="train")
    eval_dataset = load_example_dataset("gsm8k", split="test")
    rubric = vf.MathRubric()
    return vf.SingleTurnEnv(dataset=dataset, eval_dataset=eval_dataset,
                            system_prompt=system_prompt, parser=rubric.parser, rubric=rubric)
```

**reverse_text (custom reward):**
```python
def load_environment():
    parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")
    def lcs_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)
    rubric = vf.Rubric(funcs=[lcs_reward_func])
    return vf.SingleTurnEnv(dataset=dataset, system_prompt=..., parser=parser, rubric=rubric)
```

**wiki_search (multi-turn tools + judge):**
```python
def load_environment(max_turns=10, judge_model="gpt-4.1-mini"):
    tools = [search_pages, view_sections, read_section]  # async functions
    judge_rubric = JudgeRubric(judge_client=..., judge_model=judge_model)
    judge_rubric.add_reward_func(judge_reward_func)
    return vf.ToolEnv(dataset=dataset, tools=tools, rubric=judge_rubric, max_turns=max_turns)
```

---

## Part 2: DAT-Bench Current State

### What It Has
- **LLM providers** via `OpenAICompatibleClient` (OpenAI, Anthropic, DeepSeek, Gemini, Ollama, OpenRouter)
- **Structured output** via Pydantic `DATWords` model (10 words exactly)
- **GloVe-based scorer** (`DATScorer`) — pairwise cosine distance → DAT score 0-200
- **Additional metrics**: DSI, Lempel-Ziv complexity
- **Async experiment runner** with incremental JSON saving
- **Statistical visualization**: ridge plots, heatmaps (Cohen's d + multiple comparison correction), word frequency, triangular distance matrices
- **Strategy system**: 4 DAT prompting strategies + temperature configs

### What It Lacks for Verifiers Alignment
1. No `Environment` subclass — uses standalone `ExperimentRunner`
2. No `Rubric` — scoring is inline in the runner
3. No `RolloutInput/Output/State` — uses flat JSON dicts
4. No dataset abstraction — prompts are generated on-the-fly from strategy config
5. No trajectory tracking — single-turn but no structured output
6. No advantage computation or group-level scoring
7. No TOML eval config integration
8. No training data export

---

## Part 3: Alignment Plan

### 3.1 Create DAT Environment (`environments/dat_bench/dat_bench.py`)

This is the primary integration point. DAT is fundamentally single-turn (one prompt → one set of 10 words), so `SingleTurnEnv` is the right base.

```python
import verifiers as vf
from datasets import Dataset
from divergent_bench.dat.scorer import DATScorer
from divergent_bench.config.strategies import DAT_STRATEGIES, DEFAULT_TEMPERATURES


def load_environment(
    strategy: str = "my_prompt",
    num_examples: int = 50,
    system_prompt: str | None = None,
) -> vf.Environment:
    """Load DAT benchmark as a verifiers environment.

    Args:
        strategy: DAT prompting strategy (none, random, t_prompt, my_prompt)
        num_examples: Number of evaluation examples (each is one DAT trial)
        system_prompt: Optional system prompt override
    """
    # DAT has no "dataset" in the traditional sense — each trial is the same
    # prompt. We create a synthetic dataset of N identical trials.
    prompt_text = DAT_STRATEGIES[strategy]
    dataset = Dataset.from_dict({
        "question": [prompt_text] * num_examples,
        "answer": [""] * num_examples,  # No ground truth for creativity
        "info": ['{"strategy": "' + strategy + '"}'] * num_examples,
    })

    # Rubric wrapping the GloVe scorer
    rubric = DATRubric(strategy=strategy)

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        sampling_args={"temperature": DEFAULT_TEMPERATURES.get(strategy, 0.7)},
    )
```

**Key design choice:** Each row in the dataset is one DAT trial (same prompt, independent sample). This naturally maps to verifiers' `rollouts_per_example` for variance estimation and `num_examples` for total trials.

Actually — on reflection, a better mapping: use `num_examples=1` with `rollouts_per_example=N`. That way all N trials for the same strategy are grouped, enabling group-level scoring (diversity bonus, variance tracking). But this doesn't work cleanly because the prompt is literally identical each time. The right call: **multiple dataset rows, each an independent trial**, scored individually. Group scoring isn't meaningful when the "answer" is open-ended creativity.

### 3.2 Create DAT Rubric (`divergent_bench/rubrics/dat_rubric.py`)

```python
import verifiers as vf
from divergent_bench.dat.scorer import DATScorer


class DATRubric(vf.Rubric):
    def __init__(self, strategy: str = "none", **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self._scorer: DATScorer | None = None  # Lazy load (GloVe is 5GB)

        # Primary reward: normalized DAT score
        self.add_reward_func(self.dat_score, weight=1.0)

        # Metrics (tracked, not weighted)
        self.add_metric(self.num_valid_words)
        self.add_metric(self.raw_dat_score)

    @property
    def scorer(self) -> DATScorer:
        if self._scorer is None:
            self._scorer = DATScorer()
        return self._scorer

    async def dat_score(self, completion, state, **kwargs) -> float:
        """Normalized DAT score as reward (0.0 to 1.0)."""
        words = self._extract_words(completion)
        state["dat_words"] = words
        raw_score = self.scorer.dat(words, minimum=7)
        if raw_score is None:
            return 0.0
        state["raw_dat_score"] = raw_score
        # Normalize: DAT scores typically range 40-95
        # Map to 0-1 range for RL compatibility
        return max(0.0, min(1.0, (raw_score - 40) / 60))

    async def num_valid_words(self, state, **kwargs) -> float:
        words = state.get("dat_words", [])
        valid = [w for w in words if self.scorer.validate(w) is not None]
        return float(len(valid))

    async def raw_dat_score(self, state, **kwargs) -> float:
        return float(state.get("raw_dat_score", 0.0))

    def _extract_words(self, completion) -> list[str]:
        """Extract word list from model completion."""
        if not completion:
            return []
        content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else ""
        # Parse various formats (numbered lists, comma-separated, etc.)
        # Reuse logic from ExperimentRunner._parse_word_list
        import re
        words = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering, bullets, markdown
            cleaned = re.sub(r'^[\d]+[.):\s]+', '', line)
            cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
            cleaned = cleaned.strip().strip('"').strip("'")
            # Take first word if line has descriptions
            if " - " in cleaned:
                cleaned = cleaned.split(" - ")[0].strip()
            if cleaned and len(cleaned) < 30:
                words.append(cleaned.lower())
        return words[:10]
```

### 3.3 What Changes in DAT-Bench

**Nothing breaks.** The standalone runner, visualization, and analysis continue to work. We're *adding* a verifiers-compatible layer on top:

```
divergent_bench/
├── dat/scorer.py                # Unchanged
├── llm/providers.py             # Unchanged
├── experiments/runner.py         # Unchanged (standalone mode)
├── visualization/               # Unchanged
├── rubrics/                     # NEW
│   └── dat_rubric.py            # vf.Rubric wrapping DATScorer
├── config/strategies.py         # Unchanged
└── __init__.py                  # Export DATRubric

environments/
└── dat_bench/
    ├── dat_bench.py             # load_environment() → vf.SingleTurnEnv
    └── pyproject.toml           # Package metadata
```

### 3.4 pyproject.toml for the Environment

```toml
[project]
name = "dat-bench"
description = "Divergent Association Task benchmark for measuring creativity in LLMs"
tags = ["single-turn", "creativity", "eval"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.8",
    "divergent-bench",  # The existing package
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["dat_bench.py", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 30
rollouts_per_example = 1
```

### 3.5 Running Evaluation via Verifiers

```bash
# Install
prime env install dat-bench

# Run evaluation
prime eval run dat-bench -m gpt-4.1-mini -n 30 -r 1 -s
prime eval run dat-bench -m claude-sonnet -n 30 -r 1 -s -a '{"strategy": "t_prompt"}'

# Multi-model eval via TOML
prime eval run configs/eval/dat-benchmark.toml
```

```toml
# configs/eval/dat-benchmark.toml
num_examples = 30
rollouts_per_example = 1

[[eval]]
env_id = "dat-bench"
model = "gpt-4.1-mini"
env_args = {strategy = "my_prompt"}

[[eval]]
env_id = "dat-bench"
model = "claude-sonnet"
env_args = {strategy = "my_prompt"}
```

### 3.6 Training Data Flow

Once evaluation produces `results.jsonl`, the data is already in GRPO-compatible format:

```json
{
  "example_id": 0,
  "prompt": [{"role": "user", "content": "Generate 10 words..."}],
  "completion": [{"role": "assistant", "content": "1. whale\n2. hammer\n..."}],
  "reward": 0.64,
  "metrics": {"dat_score": 0.64, "num_valid_words": 10.0, "raw_dat_score": 78.4}
}
```

For fine-tuning:
- **Reward normalization**: The `dat_score` reward function already normalizes to 0-1
- **Advantage**: Computed automatically by `rubric.score_group()` as `reward - group_mean`
- **Token-level data**: Use `OpenAIChatCompletionsTokenClient` (via `--api-client-type openai_chat_completions_token`) to get logprobs for proper GRPO

### 3.7 Preserving Standalone Analysis

The key insight: **verifiers handles orchestration and scoring; DAT-Bench's visualization and statistical analysis remain standalone**.

Workflow:
1. Run eval via verifiers → `results.jsonl`
2. Convert `results.jsonl` to DAT-Bench's JSON format (trivial: extract `completion`, parse words, add metadata)
3. Run existing visualization pipeline on converted results
4. OR: extend `loader.py` to also load `results.jsonl` directly

```python
# In visualization/loader.py, add:
def load_verifiers_results(path: str) -> pd.DataFrame:
    """Load results from verifiers' results.jsonl format."""
    import jsonlines
    rows = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            completion = obj.get("completion", [])
            content = completion[-1]["content"] if completion else ""
            rows.append({
                "model": obj.get("task", "unknown"),
                "score": obj["metrics"].get("raw_dat_score", 0),
                "words": _parse_words(content),
                "strategy": json.loads(obj.get("info", "{}")).get("strategy", "none"),
                "temperature": 0.7,  # from metadata
            })
    return pd.DataFrame(rows)
```

---

## Part 4: Implementation Sequence

### Step 1: Create DATRubric (1-2 hours)
- Wrap `DATScorer` in `vf.Rubric` subclass
- Normalized reward (0-1) + raw score metric + valid word count metric
- Word extraction from completion messages
- Test with mock completions

### Step 2: Create Environment (1 hour)
- `load_environment()` function returning `vf.SingleTurnEnv`
- Synthetic dataset from strategy prompt
- Wire in rubric, sampling args
- `pyproject.toml`

### Step 3: Verify Evaluation Works (30 min)
- `prime env install dat-bench`
- `prime eval run dat-bench -m gpt-4.1-mini -n 5 -r 1 -s`
- Check `results.jsonl` has correct structure

### Step 4: Results Loader Bridge (1 hour)
- Extend `visualization/loader.py` to read `results.jsonl`
- Verify existing plots work with verifiers-produced data
- End-to-end: eval → load → ridge plot

### Step 5: Multi-Model Eval TOML (30 min)
- Create `configs/eval/dat-benchmark.toml`
- Run across available models
- Generate comparison visualizations

### Step 6: Training Data Validation (1 hour)
- Verify reward normalization is appropriate for GRPO
- Check token-level data with token client
- Document training config requirements

---

## Part 5: What This Unlocks

1. **Unified evaluation**: Run DAT alongside math/code/reasoning benchmarks in the same pipeline
2. **Model comparison**: Use verifiers' built-in pass@k, aggregate metrics, resume/checkpoint
3. **Fine-tuning on creativity**: GRPO with DAT score as reward signal
4. **Word chain benchmark**: Same pattern — `WordChainRubric(vf.Rubric)` + `load_environment()` + existing analysis
5. **Combined training**: `EnvGroup` with DAT + word chain + other creativity benchmarks for multi-task RL

---

## Part 6: Key Gotchas

1. **GloVe loading is slow** (~30s for 840B). Use lazy loading in the rubric, not at import time. Consider if the verifiers evaluation pipeline will re-instantiate the environment per rollout (it doesn't — one `load_environment()` call, rubric persists).

2. **DAT has no ground truth answer**. The `answer` field will be empty string. Reward comes purely from the GloVe-based scorer, not from comparison to a known answer.

3. **Reward scale matters for RL**. DAT raw scores range ~40-95. Must normalize to a range where GRPO's advantage computation is meaningful. The proposed (score - 40) / 60 mapping gives 0-1.

4. **Structured output helps but isn't required**. The verifiers client doesn't support `output_type` for structured output — it sends plain chat messages. The rubric's `_extract_words()` must robustly parse free-form text. This is actually fine — the existing DAT-Bench runner already has text parsing fallback.

5. **Temperature matters for creativity benchmarks**. The default `sampling_args` in the environment should set temperature (0.7 for most strategies, 1.0 for "random"). This interacts with verifiers' `--temperature` CLI flag — document the precedence.

6. **No per-provider routing**. Verifiers uses its own client abstraction (`OpenAIChatCompletionsClient`, `AnthropicMessagesClient`). DAT-Bench's `OpenAICompatibleClient` is not used when running through verifiers. This is fine — verifiers handles the API calls.
