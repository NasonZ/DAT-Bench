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
- **Statistical visualization**: ranked dot plots, significance matrices, word-model heatmaps, triangular distance matrices
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

## Part 2.5: Reference Environments — Patterns Worth Borrowing

Before designing DAT's integration, these existing verifiers environments informed the approach. Each solves a problem DAT also faces.

### `vf_reverse_text` — the closest structural analog

**Reference:** `environments/vf_reverse_text/vf_reverse_text.py`

The simplest SingleTurnEnv with a continuous (non-binary) reward. DAT's structure mirrors this almost exactly:

```python
parser = vf.XMLParser(["think", "answer"], answer_field="answer")
def lcs_reward_func(completion, answer, **kwargs) -> float:
    response = parser.parse_answer(completion) or ""
    return lcs_ratio(response, answer)  # continuous 0-1
rubric = vf.Rubric(funcs=[lcs_reward_func, parser.get_format_reward_func()], weights=[1.0, 0.2])
return vf.SingleTurnEnv(dataset=dataset, system_prompt=..., parser=parser, rubric=rubric)
```

**What to borrow:**
- The `parser.get_format_reward_func()` pattern — a free reward signal that nudges the model toward clean output format. DAT should use this too.
- Continuous reward (LCS ratio) rather than binary pass/fail — same as DAT's normalized score.
- The weight split: primary reward at 1.0, format compliance at 0.2.

### `vf_wordle` — composite reward with multiple signals

**Reference:** `environments/vf_wordle/vf_wordle.py`

Wordle uses four reward functions, each capturing a different quality dimension:

```python
rubric = vf.Rubric(parser=parser)
rubric.add_reward_func(check_answer_reward_func)         # binary: got the word?
rubric.add_reward_func(partial_credit_reward_func)       # green/yellow letter count
rubric.add_reward_func(count_turns_reward_func)          # efficiency bonus
rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)
```

**What to borrow:**
- **Partial credit matters for RL.** A single DAT score conflates "model output nonsense" (0 valid words) with "model was uncreative" (10 valid words, low diversity). Wordle separates these — `check_answer` vs `partial_credit` — giving the trainer a richer gradient. DAT should similarly separate word validity from semantic distance.
- **Efficiency as a reward dimension.** DAT's analog: penalizing models that pad output with explanations, descriptions, or caveats instead of giving clean word lists.

### `vf_continuation_quality` — no ground truth, external quality scorer

**Reference:** `environments/vf_continuation_quality/vf_continuation_quality.py`

This is the pattern for "there is no correct answer — quality is assessed by an external system":

```python
def grade_reward(prompt, completion, answer, state, **kwargs) -> float:
    judge_response = rubric.judge(prompt, completion, answer, state, **kwargs)
    judge_grade = grade_parser.parse_answer(judge_response) or "F"
    return {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}.get(judge_grade, 0.0)
rubric.add_reward_func(grade_reward, weight=1.0)
```

**What to borrow:**
- **Validates the "no ground truth" architecture.** DAT's `answer` field is empty string — reward comes purely from the GloVe scorer, not from comparison to a known answer. `continuation_quality` does the same thing with a judge model instead of GloVe.
- **The grade-to-float mapping pattern.** DAT's `(raw_score - 40) / 60` normalization serves the same purpose as the A→1.0/B→0.75 mapping — compressing a qualitative scale into RL-compatible [0, 1].

### `vf_toxicity_explanation` — composite evaluation with rich `info`

**Reference:** `environments/vf_toxicity_explanation/vf_toxicity_explanation.py`

Shows how to use the `info` dict for per-example metadata:

```python
return {
    "question": f"Analyze the following text...",
    "answer": "toxic" if is_toxic else "non-toxic",
    "info": {
        "is_toxic": is_toxic,
        "categories": toxicity_details,
        "text": example["text"],
        "label": "toxic" if is_toxic else "non-toxic",
    },
}
```

**What to borrow:**
- **Per-row `info` dicts instead of identical rows.** The original DAT plan creates N identical dataset rows. But `info` can carry strategy variant, temperature, trial index — making each row distinguishable in results without changing the prompt. This is important for downstream analysis (grouping by strategy in the visualization pipeline).

### `vf_sentence_repeater` — custom MultiTurnEnv subclass

**Reference:** `environments/vf_sentence_repeater/vf_sentence_repeater.py`

A custom `MultiTurnEnv` with overridden `is_completed()` and `env_response()`:

```python
class SentenceRepeaterEnv(vf.MultiTurnEnv):
    def is_completed(self, messages, state, **kwargs) -> bool:
        return state["turn"] >= len(state["info"]["questions"])

    def env_response(self, messages, state, **kwargs):
        return [{"role": "user", "content": state["info"]["questions"][state["turn"]]}], state
```

**What to borrow (for word chain, not DAT):**
- This is the exact pattern for the word chain trajectory benchmark. Each turn: model extends the chain → environment validates the link and returns feedback → model continues. `is_completed()` checks chain validity or max length. The `info` dict carries the valid word graph for verification.

---

## Part 3: Alignment Plan

### 3.1 Create DAT Rubric (`divergent_bench/rubrics/dat_rubric.py`)

The rubric is the most important piece — it defines the reward signal that drives both evaluation quality and RL training effectiveness.

**Design rationale — composite rewards, not a single score:**

The original plan had one reward function (`dat_score`). But examining `vf_wordle`, a single score conflates distinct failure modes. A model that outputs "the, a, is, of, in, to, it, he, we, my" (10 valid common words, low DAT score ~40) is qualitatively different from one that outputs "here are ten creative words: [paragraph of explanation]" (0 parseable words). The RL trainer needs different gradient signals for these cases.

Following Wordle's pattern of `check_answer` + `partial_credit` + `format_compliance`:

```python
import verifiers as vf
from divergent_bench.dat.scorer import DATScorer


class DATRubric(vf.Rubric):
    """Rubric for the Divergent Association Task.

    Composite reward signal with three components:
    1. Creativity score (primary) — normalized GloVe-based DAT score
    2. Word validity (secondary) — proportion of words in GloVe vocabulary
    3. Format compliance (parser) — did the model follow the output format?

    The separation matters for RL: a model that outputs valid but common words
    gets credit for compliance and validity but low creativity reward. A model
    that outputs gibberish gets nothing. This gives the trainer a richer
    gradient than a single collapsed score.

    Reference: follows the multi-reward pattern from vf_wordle
    (check_answer + partial_credit + format_compliance).
    """

    def __init__(self, strategy: str = "none", parser=None, **kwargs):
        parser = parser or vf.XMLParser(
            fields=["words"], answer_field="words"
        )
        super().__init__(parser=parser, **kwargs)
        self.strategy = strategy
        self._scorer: DATScorer | None = None  # Lazy load (GloVe is 5GB)

        # Primary reward: normalized DAT score (weight=1.0)
        self.add_reward_func(self.creativity_reward, weight=1.0)

        # Secondary reward: word validity proportion (weight=0.2)
        # Rationale: nudges the model toward GloVe-valid vocabulary without
        # dominating the creativity signal. A model producing 10 valid common
        # words gets 0.2 * 1.0 = 0.2 from this alone.
        self.add_reward_func(self.validity_reward, weight=0.2)

        # Format compliance from parser (weight=0.1)
        # Reference: same pattern as vf_reverse_text and vf_wordle
        self.add_reward_func(parser.get_format_reward_func(), weight=0.1)

        # Metrics (weight=0, tracked but don't affect reward)
        self.add_reward_func(self.raw_dat_score_metric, weight=0.0)
        self.add_reward_func(self.valid_word_count_metric, weight=0.0)

    @property
    def scorer(self) -> DATScorer:
        if self._scorer is None:
            self._scorer = DATScorer()
        return self._scorer

    async def creativity_reward(self, completion, state, **kwargs) -> float:
        """Normalized DAT score as primary reward (0.0 to 1.0).

        Normalization: (raw_score - 40) / 60. DAT scores typically range
        40-95 across models. This maps to [0, ~0.92] for the observed range,
        with theoretical maximum 1.0 at raw score 100.

        Reference: same normalization-to-unit-interval pattern as
        vf_continuation_quality's grade mapping (A→1.0, B→0.75, ...).
        """
        words = self._extract_words(completion, state)
        raw_score = self.scorer.dat(words, minimum=7)
        if raw_score is None:
            return 0.0
        state["raw_dat_score"] = raw_score
        return max(0.0, min(1.0, (raw_score - 40) / 60))

    async def validity_reward(self, state, **kwargs) -> float:
        """Proportion of extracted words that exist in GloVe vocabulary.

        Returns 0.0-1.0. Gives the model credit for producing real words
        even when the creativity score is low. Separates "uncreative" from
        "didn't follow the task at all".

        Reference: analogous to vf_wordle's partial_credit_reward_func
        which gives credit for yellow/green letters even on wrong guesses.
        """
        words = state.get("dat_words", [])
        if not words:
            return 0.0
        valid = [w for w in words if self.scorer.validate(w) is not None]
        return len(valid) / len(words)

    async def raw_dat_score_metric(self, state, **kwargs) -> float:
        """Raw DAT score (0-200 scale) as a tracked metric."""
        return float(state.get("raw_dat_score", 0.0))

    async def valid_word_count_metric(self, state, **kwargs) -> float:
        """Count of GloVe-valid words as a tracked metric."""
        words = state.get("dat_words", [])
        valid = [w for w in words if self.scorer.validate(w) is not None]
        return float(len(valid))

    def _extract_words(self, completion, state) -> list[str]:
        """Extract word list from model completion.

        Tries parser-based extraction first (structured <words> tags),
        falls back to regex parsing of numbered lists, comma-separated
        values, etc. This dual approach means the model gets format_reward
        for using tags, but still gets scored if it doesn't.

        Stores result in state["dat_words"] for downstream reward functions.
        """
        if not completion:
            state["dat_words"] = []
            return []

        # Try parser first (structured output via XMLParser)
        parsed = self.parser.parse_answer(completion)
        if parsed:
            # Parser returned content from <words> tags
            words = [w.strip().lower() for w in parsed.split(",") if w.strip()]
            if len(words) >= 5:
                state["dat_words"] = words[:10]
                return words[:10]

        # Fallback: regex parsing of free-form text
        content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else ""
        import re
        words = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r'^[\d]+[.):\s]+', '', line)
            cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
            cleaned = cleaned.strip().strip('"').strip("'")
            if " - " in cleaned:
                cleaned = cleaned.split(" - ")[0].strip()
            if cleaned and len(cleaned) < 30 and " " not in cleaned:
                words.append(cleaned.lower())
        state["dat_words"] = words[:10]
        return words[:10]
```

### 3.2 Create DAT Environment (`environments/dat_bench/dat_bench.py`)

DAT is single-turn (one prompt → one set of 10 words), so `SingleTurnEnv` is the right base — same as `vf_reverse_text`.

```python
import verifiers as vf
from datasets import Dataset
from divergent_bench.dat.scorer import DATScorer
from divergent_bench.config.strategies import DAT_STRATEGIES, DEFAULT_TEMPERATURES
from divergent_bench.rubrics.dat_rubric import DATRubric


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
    prompt_text = DAT_STRATEGIES[strategy]
    temperature = DEFAULT_TEMPERATURES.get(strategy, 0.7)

    # Each row is one independent DAT trial. The prompt is identical across
    # rows — variance comes from sampling (temperature > 0). The info dict
    # carries per-row metadata so results are distinguishable downstream.
    #
    # Design choice: N rows × 1 rollout, not 1 row × N rollouts.
    # Group scoring (advantage = reward - group_mean) requires rollouts of
    # the *same example* to be meaningfully comparable. For DAT, every trial
    # is independent open-ended generation — there's no "this trial was
    # harder" dimension to normalize against. Independent rows are cleaner.
    #
    # Reference: vf_toxicity_explanation uses per-row info dicts the same way.
    dataset = Dataset.from_dict({
        "question": [prompt_text] * num_examples,
        "answer": [""] * num_examples,  # No ground truth for creativity
        "info": [
            {"strategy": strategy, "temperature": temperature, "trial": i}
            for i in range(num_examples)
        ],
    })

    # Parser: XMLParser so the model can output <words>w1, w2, ...</words>
    # and get format_reward credit. Rubric falls back to regex if model
    # doesn't use tags.
    parser = vf.XMLParser(fields=["words"], answer_field="words")

    rubric = DATRubric(strategy=strategy, parser=parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        sampling_args={"temperature": temperature},
    )
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
  "completion": [{"role": "assistant", "content": "<words>whale, hammer, ...</words>"}],
  "reward": 0.74,
  "metrics": {
    "creativity_reward": 0.64,
    "validity_reward": 1.0,
    "format_reward": 1.0,
    "raw_dat_score_metric": 78.4,
    "valid_word_count_metric": 10.0
  }
}
```

The composite reward `0.74 = 0.64×1.0 + 1.0×0.2 + 1.0×0.1` (creativity + validity + format). Compare to a model that outputs unformatted gibberish: `0.0 = 0.0×1.0 + 0.0×0.2 + 0.0×0.1`. Compare to a model that outputs 10 common valid words in correct format but low creativity (raw score 52): `0.42 = 0.2×1.0 + 1.0×0.2 + 1.0×0.1`. The trainer can distinguish all three cases.

For fine-tuning:
- **Reward normalization**: Composite score naturally falls in [0, 1.3] range. GRPO's advantage computation (reward - group_mean) works well here.
- **Advantage**: Computed automatically by `rubric.score_group()` as `reward - group_mean`
- **Token-level data**: Use `OpenAIChatCompletionsTokenClient` (via `--api-client-type openai_chat_completions_token`) to get logprobs for proper GRPO

### 3.7 Preserving Standalone Analysis

The key insight: **verifiers handles orchestration and scoring; DAT-Bench's visualization and statistical analysis remain standalone**.

Workflow:
1. Run eval via verifiers → `results.jsonl`
2. Load directly into visualization pipeline via `load_verifiers_results()`
3. Existing plots work unchanged — they only need `model`, `score`, `words` columns

```python
# In visualization/loader.py (already partially implemented):
def load_verifiers_results(path: str) -> pd.DataFrame:
    """Load results from verifiers' results.jsonl format."""
    import jsonlines
    rows = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            completion = obj.get("completion", [])
            content = completion[-1]["content"] if completion else ""
            info = obj.get("info", {})
            rows.append({
                "model": obj.get("model", "unknown"),
                "score": obj["metrics"].get("raw_dat_score_metric", 0),
                "words": _parse_words(content),
                "strategy": info.get("strategy", "none"),
                "temperature": info.get("temperature", 0.7),
                "creativity_reward": obj["metrics"].get("creativity_reward", 0),
                "validity_reward": obj["metrics"].get("validity_reward", 0),
            })
    return pd.DataFrame(rows)
```

### 3.8 Word Chain Benchmark — MultiTurnEnv Pattern

The word chain trajectory benchmark (see `word-chain-trajectory-benchmark-design.md`) follows the `vf_sentence_repeater` pattern — a custom `MultiTurnEnv` subclass where the environment validates each step:

```python
class WordChainEnv(vf.MultiTurnEnv):
    """Each turn: model extends chain → env validates → model continues.

    Reference: follows vf_sentence_repeater pattern
    (environments/vf_sentence_repeater/vf_sentence_repeater.py)
    """
    def is_completed(self, messages, state, **kwargs) -> bool:
        # Chain is done when model outputs DONE or link is invalid
        return state.get("chain_broken", False) or state["turn"] >= self.max_turns

    def env_response(self, messages, state, **kwargs):
        # Validate the latest word link, return feedback
        last_word = self._extract_last_word(messages)
        prev_word = state.get("chain", [""])[-1]
        valid, edit_type = validate_link(prev_word, last_word)
        if not valid:
            state["chain_broken"] = True
            return [{"role": "user", "content": f"Invalid link: '{last_word}' is not one edit from '{prev_word}'. Chain ended."}], state
        state["chain"].append(last_word)
        return [{"role": "user", "content": f"Valid ({edit_type}). Chain length: {len(state['chain'])}. Continue."}], state
```

This is distinct from DAT (which is single-turn) but uses the same rubric infrastructure. An `EnvGroup` could combine both for multi-task creativity RL training.

---

## Part 4: Implementation Sequence

### Step 1: Create DATRubric
- Composite reward: creativity (1.0) + validity (0.2) + format (0.1)
- XMLParser with `<words>` field + regex fallback
- Lazy GloVe loading
- Test with mock completions covering: valid words, invalid words, malformed output, correct XML format, free-text format

### Step 2: Create Environment
- `load_environment()` function returning `vf.SingleTurnEnv`
- Per-row `info` dicts with strategy/temperature/trial metadata
- Wire in parser, rubric, sampling args
- `pyproject.toml`

### Step 3: Verify Evaluation Works
- `prime env install dat-bench`
- `prime eval run dat-bench -m gpt-4.1-mini -n 5 -r 1 -s`
- Check `results.jsonl` has all expected metrics: `creativity_reward`, `validity_reward`, `format_reward`, `raw_dat_score_metric`, `valid_word_count_metric`

### Step 4: Results Loader Bridge
- Extend `visualization/loader.py` to read `results.jsonl`
- Verify existing plots work with verifiers-produced data
- End-to-end: eval → load → ranked_dot_plot

### Step 5: Multi-Model Eval TOML
- Create `configs/eval/dat-benchmark.toml`
- Run across available models
- Generate comparison visualizations

### Step 6: Training Data Validation
- Verify composite reward scale is appropriate for GRPO
- Inspect advantage distribution across a batch — should be centered near 0 with meaningful spread
- Check token-level data with token client
- Document training config requirements

---

## Part 5: What This Unlocks

1. **Unified evaluation**: Run DAT alongside math/code/reasoning benchmarks in the same pipeline. Same CLI, same TOML configs, same results format.
2. **Model comparison**: Use verifiers' built-in pass@k, aggregate metrics, resume/checkpoint — no custom orchestration needed.
3. **Fine-tuning on creativity**: GRPO with composite reward signal (creativity + validity + format). The multi-reward approach gives richer gradients than a single DAT score.
4. **Word chain benchmark**: Same rubric infrastructure, `MultiTurnEnv` subclass following `vf_sentence_repeater` pattern.
5. **Combined training**: `EnvGroup` with DAT + word chain + other creativity benchmarks for multi-task RL.
6. **Cross-benchmark analysis**: Correlation between creativity scores and reasoning/math performance becomes trivially queryable when both run through verifiers.

---

## Part 6: Key Gotchas

1. **GloVe loading is slow** (~30s for 840B). Use lazy loading in the rubric (`@property`), not at import time. Verifiers calls `load_environment()` once; the rubric persists across all rollouts.

2. **DAT has no ground truth answer**. The `answer` field will be empty string. Reward comes purely from the GloVe-based scorer, not from comparison to a known answer. This is validated by `vf_continuation_quality` which does the same with a judge model.

3. **Reward scale matters for RL**. The composite reward ranges [0, ~1.3] in practice. GRPO's advantage computation (reward - group_mean) works well when there's meaningful variance in this range. Monitor: if all models cluster near the same composite score, the advantage signal is weak and training won't converge. Consider adjusting weights or adding a diversity group reward.

4. **Parsing strategy: XMLParser vs structured output vs regex.** Three options, each with trade-offs:
   - **XMLParser** (recommended for RL): `XMLParser(fields=["words"])` gives a free format compliance reward via `parser.get_format_reward_func()`. The model learns to use `<words>` tags. Regex fallback handles models that don't cooperate. Reference: every real verifiers environment uses a parser.
   - **Structured output** (recommended for pure eval): Pass `response_format=DATWords` via `sampling_args`. Guarantees clean JSON with exactly 10 words. No parsing failures, but removes the format reward gradient — the model always produces valid output by construction.
   - **Pure regex** (original plan): Ad-hoc `_extract_words()`. Works but gives no format signal to the trainer and is fragile across model output styles.

   For training, XMLParser is preferred — the format reward is a cheap, reliable learning signal. For evaluation-only runs, structured output via Pydantic is cleaner and eliminates parsing noise.

5. **Temperature matters for creativity benchmarks**. The environment sets temperature via `sampling_args`. Verifiers CLI also has `--temperature` which takes precedence. Document this: DAT's strategy-specific temperatures (0.7 default, 1.0 for "random") should be respected.

6. **Structured output IS available via `sampling_args`.** Verifiers spreads `sampling_args` directly into `client.chat.completions.create(**sampling_args)` (see `environment.py:get_model_response`), and also supports `extra_body` for provider-specific extensions. This means `response_format` (including OpenAI's Pydantic structured output) can be passed through. DAT could use this instead of XMLParser — e.g. `sampling_args={"response_format": DATWords}` where `DATWords` is the existing Pydantic model. Trade-off: structured output guarantees clean parsing but removes the format compliance reward signal (model always produces valid JSON). For RL training, the XMLParser + format reward approach may give richer gradients; for pure evaluation, structured output is cleaner. Worth testing both.

7. **Per-row `info` dicts, not identical rows**. Even though every DAT trial has the same prompt, each row should carry a unique `info` dict with `{strategy, temperature, trial}`. This makes results distinguishable in `results.jsonl` and enables downstream grouping in the visualization pipeline. Reference: `vf_toxicity_explanation` uses this pattern for per-example metadata.
