# Word Chain Trajectory Benchmark — Design Document

**Authors:** NasonZ / Team
**Date:** 2026-03-12
**Status:** Draft
**Inspiration:** ["The Bridge Pattern: How Opus 4.5 Exploits Word Graph Topology"](https://x.com/scaling01/status/2031855619360088558) by @scaling01 (LisanBench)

---

## 1. Executive Summary

We propose adding a **word chain trajectory benchmark** to `divergent_bench`. The benchmark asks language models to produce the longest possible chain of valid English words where each consecutive pair differs by exactly one edit operation (insertion, deletion, or substitution). Unlike traditional benchmarks that only measure final scores, this benchmark treats **every chain as a walk through a fully observable graph**, enabling rich trajectory-level analysis of model strategy, creativity, and word knowledge.

The key insight from @scaling01's LisanBench work is that different models adopt qualitatively different strategies — some exploit graph topology through "bridge patterns" (insert→substitute→delete sequences), while others rely purely on substitution. These strategies are detectable through geometric and spectral analysis of the trajectory.

### Goals

1. **Benchmark word knowledge and strategic reasoning** across frontier LLMs
2. **Produce trajectory-level analysis** — not just scores, but *how* models solve the problem
3. **Detect emergent strategies** like bridge patterns, with rigorous statistical metrics
4. **Generate publication-quality interactive visualizations** (3D trajectory plots, turning angle distributions, autocorrelation, bridge rate comparisons)
5. **Reuse existing `divergent_bench` infrastructure** (LLM providers, result storage, statistical analysis, styling)

---

## 2. Background & References

### 2.1 Word Ladders

The word ladder puzzle was invented by Lewis Carroll in 1877. The computational version asks: given a dictionary, find a path between two words where each step changes exactly one letter (substitution only). Our variant generalizes this:

- **Allowed operations:** insertion, deletion, substitution (Levenshtein distance = 1)
- **Objective:** maximize chain length from a starting word (open-ended, no target word)
- **Constraint:** no repeated words in the chain

This generalization creates a much denser graph and enables the bridge pattern strategy.

**References:**
- Knuth, D.E. (1993). *The Stanford GraphBase: A Platform for Combinatorial Computing.* ACM Press. (Original computational word ladder treatment)
- Levenshtein, V.I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals." *Soviet Physics Doklady*, 10(8), 707–710.

### 2.2 LisanBench

@scaling01's LisanBench (March 2026) demonstrated that word chain benchmarks reveal qualitative strategy differences between models:

- **Bridge pattern**: Instead of `name→game` (1 substitution, 1 point), Opus 4.5 goes `name→names→games→game` (3 moves, 3 points). This exploits the insert→substitute→delete (I→S→D) cycle through plural forms.
- **Square wave trajectory**: The bridge pattern creates a characteristic oscillation between word lengths 4 and 5, visible as a square wave in word-length-over-time plots and a period-4 signal in turning angle autocorrelation.
- **Model fingerprinting**: The bridge pattern rate acts as a model signature — GLM-5 showing the same pattern as Opus 4.5 added to evidence of training data contamination.

**Key metrics introduced by LisanBench:**
1. Chain length (primary score)
2. Bridge pattern rate (I→S→D / D→S→I subsequence frequency)
3. Turning angle distribution (geometric analysis of trajectory)
4. Turning angle autocorrelation (periodic strategy detection)
5. Word length oscillation over chain steps
6. Graph density along trajectory path

### 2.3 Divergent Association Task (DAT)

Our existing `divergent_bench` implements the DAT (Olson et al., 2021), which measures creativity by asking models to generate semantically distant words scored via GloVe embeddings. The word chain benchmark complements DAT by testing a different dimension: **strategic navigation through structured word space** rather than open-ended semantic divergence.

**Reference:**
- Olson, J.A., Nahas, J., Chmoulevitch, D., Cropper, S.J., & Webb, M.E. (2021). "Naming unrelated words predicts creativity." *Proceedings of the National Academy of Sciences*, 118(25), e2022340118.

---

## 3. Architecture

### 3.1 Where It Lives

```
divergent_bench/
├── divergent_bench/
│   ├── dat/                          # Existing DAT module (unchanged)
│   ├── wordchain/                    # NEW: Word chain module
│   │   ├── __init__.py
│   │   ├── graph.py                  # Word graph construction & embedding
│   │   ├── scorer.py                 # Chain validation & scoring
│   │   ├── prompts.py                # Prompting strategies
│   │   └── metrics.py                # Trajectory metrics (turning angles, bridges, etc.)
│   ├── llm/                          # Existing (reused as-is)
│   ├── experiments/
│   │   ├── runner.py                 # Existing DAT runner
│   │   └── chain_runner.py           # NEW: Word chain experiment runner
│   ├── visualization/
│   │   ├── plots.py                  # Existing plots (reused)
│   │   ├── styles.py                 # Existing styles (extended)
│   │   ├── loader.py                 # Existing loader (extended)
│   │   ├── trajectory_plots.py       # NEW: 3D trajectory, turning angles, etc.
│   │   └── trajectory_interactive.py # NEW: Plotly interactive HTML exports
│   └── config/
│       ├── strategies.py             # Existing DAT strategies
│       └── chain_config.py           # NEW: Starting words, difficulty tiers
├── scripts/
│   ├── run_dat.py                    # Existing
│   ├── run_wordchain.py              # NEW: CLI for word chain experiments
│   ├── build_graph.py                # NEW: One-time graph construction script
│   └── analyze_chains.py             # NEW: Analysis & visualization CLI
├── data/
│   ├── embeddings/                   # Existing GloVe data
│   ├── wordlist/                     # NEW: Dictionary files
│   │   └── words_en.txt
│   └── graph/                        # NEW: Precomputed graph artifacts
│       ├── word_graph.gpickle        # NetworkX graph (serialized)
│       └── layout_3d.npy             # Precomputed 3D node positions
├── results/                          # Existing DAT results
└── results_wordchain/                # NEW: Word chain results
```

### 3.2 Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Dictionary  │───→│  Word Graph  │───→│  3D Layout      │
│  (words.txt) │    │  (NetworkX)  │    │  (spring/KK)    │
└─────────────┘    └──────────────┘    └─────────────────┘
                          │                      │
                          ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  LLM API    │───→│  Raw Chains  │───→│  Validated +    │
│  (providers) │    │  (JSON)      │    │  Scored Chains  │
└─────────────┘    └──────────────┘    └─────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
                   │  Trajectory  │    │  Statistical  │    │  Interactive │
                   │  Metrics     │    │  Plots (mpl)  │    │  3D (Plotly) │
                   └─────────────┘    └──────────────┘    └──────────────┘
```

---

## 4. Module Specifications

### 4.1 `wordchain/graph.py` — Word Graph Construction

**Purpose:** Build and manage the edit-distance-1 word graph.

#### Algorithm: Bucket/Wildcard Approach

Brute-force pairwise comparison of N words is O(N² × L). Instead, use the **wildcard bucket** method (O(N × L)):

```python
# For each word, generate all wildcard patterns:
# "cat" → ["_at", "c_t", "ca_",       # substitution buckets
#           "_cat", "c_at", "ca_t", "cat_",  # insertion buckets (word with _ inserted)
#           "at", "ct", "ca"]           # deletion buckets (remove each char)
#
# Words sharing a bucket are edit-distance-1 neighbors.
```

**For substitution neighbors:** Replace each character position with a wildcard `_`. All words mapping to the same pattern (e.g., `_at` → `{cat, bat, hat, mat, sat, fat, rat, ...}`) are substitution neighbors.

**For insertion neighbors:** A word of length L is an insertion neighbor of a word of length L-1 if removing one character from the longer word yields the shorter word. Use deletion buckets: for each word, generate all strings with one character removed. If that string is itself a valid word, they are insertion/deletion neighbors.

**For deletion neighbors:** Symmetric with insertion — handled by the same deletion bucket approach.

```python
class WordGraph:
    """Edit-distance-1 word graph built from a dictionary."""

    def __init__(self, wordlist_path: str | Path):
        """Build graph from dictionary file."""

    def build(self) -> nx.Graph:
        """Construct the graph using wildcard bucket method.

        Returns graph where:
          - Nodes: valid English words with attributes {length: int}
          - Edges: edit-distance-1 pairs with attributes {op: 'sub'|'ins'|'del'}
        """

    def compute_layout(self, algorithm: str = "spring", dim: int = 3,
                       seed: int = 42, iterations: int = 100) -> dict[str, np.ndarray]:
        """Compute 3D positions for all nodes.

        Args:
            algorithm: "spring" (Fruchterman-Reingold), "kamada_kawai", or "spectral"
            dim: Number of dimensions (2 or 3)
            seed: Random seed for reproducibility
            iterations: Number of layout iterations (spring only)

        Returns:
            Dict mapping word → np.ndarray of shape (dim,)
        """

    def get_neighbors(self, word: str) -> list[str]:
        """Return all edit-distance-1 neighbors of a word."""

    def get_subgraph(self, words: list[str], radius: int = 1) -> nx.Graph:
        """Extract subgraph containing given words + neighbors within radius."""

    def node_density(self, word: str, radius: int = 1) -> float:
        """Local graph density around a word (degree or clustering coefficient)."""

    def save(self, path: str | Path): ...

    @classmethod
    def load(cls, path: str | Path) -> "WordGraph": ...

    # Properties
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def average_degree(self) -> float: ...
```

#### Dictionary Selection

**Recommended: NLTK words corpus** filtered to lowercase alphabetic words, length 2–15.

| Source | Words | Notes |
|--------|-------|-------|
| NLTK `words` | ~236,000 | Broad, includes obscure words |
| `/usr/share/dict/words` | ~100,000 | System-dependent, variable quality |
| SCOWL (size 60) | ~100,000 | Configurable, well-curated |
| SCOWL (size 80) | ~170,000 | More comprehensive |
| Filtered NLTK (common only) | ~50,000–80,000 | Best for LLM-fair benchmarking |

**Recommendation:** Use NLTK words filtered to remove extremely obscure entries. The graph should contain words that a well-trained LLM would plausibly know. Consider cross-referencing with word frequency lists (e.g., Google Ngrams top 100K) to keep only reasonably common words.

#### Expected Graph Statistics

For a ~60,000 word filtered dictionary:
- **Nodes:** ~60,000
- **Edges:** ~200,000–500,000 (depending on inclusion of insert/delete edges)
- **Average degree:** ~7–15
- **Largest connected component:** >95% of nodes
- **Dense core:** 3–4 letter words (degree 20+)
- **Sparse periphery:** 10+ letter words (degree 1–3)

#### Layout Algorithm Choice

| Algorithm | Complexity | Quality | Notes |
|-----------|------------|---------|-------|
| `spring_layout` (Fruchterman-Reingold) | O(N² × iter) | Good | Best for showing community structure; 60K nodes needs k parameter tuning and ~50-100 iterations |
| `kamada_kawai_layout` | O(N³) | Best | Infeasible for >5K nodes unless using subgraph |
| `spectral_layout` | O(N × k²) | Fair | Fast, but poor local structure |

**Recommendation — Hybrid approach:** Use `spectral_layout` for initial positioning (fast, O(N×M), captures global community structure), then refine with `spring_layout` initialized from spectral positions for a small number of iterations. This is a well-known best practice for large sparse graphs:

```python
pos_init = nx.spectral_layout(G, dim=3)
pos_final = nx.spring_layout(G, dim=3, pos=pos_init, k=1/sqrt(N), iterations=50, seed=42)
```

For the full graph (~60K nodes) this will take minutes. **Precompute once and cache as `.npy`.** For interactive exploration of individual chains, extract a subgraph (chain words + 1-hop neighbors) and apply `kamada_kawai_layout` for better local quality.

**References:**
- Fruchterman, T.M.J. & Reingold, E.M. (1991). "Graph drawing by force-directed placement." *Software: Practice and Experience*, 21(11), 1129–1164.
- Koren, Y. (2005). "Drawing Graphs by Eigenvectors: Theory and Practice." *Computers and Mathematics with Applications*, 49(11-12), 1867–1888.

---

### 4.2 `wordchain/scorer.py` — Chain Validation & Scoring

**Purpose:** Validate word chains and compute primary scores.

```python
@dataclass
class ChainValidation:
    """Result of validating a word chain."""
    chain: list[str]                    # Original chain
    valid_chain: list[str]              # Chain truncated at first invalid step
    is_fully_valid: bool                # All steps valid?
    invalid_step: int | None            # Index of first invalid step (if any)
    invalid_reason: str | None          # Why it failed
    num_valid_steps: int                # Length of valid chain - 1
    move_types: list[str]              # 'sub', 'ins', 'del' for each valid step
    words_not_in_dict: list[str]        # Words not found in dictionary
    repeated_words: list[str]           # Words that appeared more than once


class ChainScorer:
    """Validates and scores word chains against the word graph."""

    def __init__(self, graph: WordGraph):
        """Initialize with a prebuilt word graph."""

    def validate_chain(self, chain: list[str]) -> ChainValidation:
        """Validate each step in the chain.

        Checks:
        1. Each word exists in the dictionary
        2. Each consecutive pair has edit distance exactly 1
        3. No word is repeated
        4. Classifies each move as insertion/deletion/substitution

        The chain is truncated at the first invalid step.
        """

    def score(self, chain: list[str]) -> int:
        """Primary score = number of valid steps (valid chain length - 1)."""

    def classify_move(self, word_a: str, word_b: str) -> str:
        """Classify the edit operation between two words.

        Returns:
            'ins' if len(word_b) == len(word_a) + 1
            'del' if len(word_b) == len(word_a) - 1
            'sub' if len(word_b) == len(word_a)

        Note: This classification by length is sufficient because
        edit distance is guaranteed to be 1 (validated separately).
        """

    @staticmethod
    def edit_distance_one(a: str, b: str) -> bool:
        """Check if two words have Levenshtein distance exactly 1."""
```

#### Move Classification

This is central to bridge detection. Given edit distance is exactly 1:

| len(b) - len(a) | Operation | Example |
|------------------|-----------|---------|
| +1 | Insertion (I) | `name` → `names` |
| 0 | Substitution (S) | `names` → `games` |
| -1 | Deletion (D) | `games` → `game` |

No ambiguity exists when edit distance is guaranteed to be 1.

---

### 4.3 `wordchain/metrics.py` — Trajectory Metrics

**Purpose:** Compute all trajectory-level metrics for analysis.

```python
class TrajectoryMetrics:
    """Compute geometric and sequential metrics from word chain trajectories."""

    def __init__(self, graph: WordGraph):
        """Initialize with graph (needed for node positions and density)."""

    # === Geometric metrics (require 3D layout) ===

    def turning_angles(self, chain: list[str]) -> np.ndarray:
        """Compute turning angle at each interior point of the chain.

        For points p[i-1], p[i], p[i+1], the turning angle is the angle
        between displacement vectors v1 = p[i] - p[i-1] and v2 = p[i+1] - p[i].

            angle = arccos( dot(v1, v2) / (|v1| × |v2|) )

        For numerical stability near 0° and 180°, use atan2:
            cross = cross_product(v1, v2)
            angle = atan2(|cross|, dot(v1, v2))

        Returns array of angles in degrees, length = len(chain) - 2.

        Reference: Standard computational geometry. The turning angle
        measures directional change; 0° = straight ahead, 90° = right angle,
        180° = full reversal.
        """

    def turning_angle_autocorrelation(self, chain: list[str],
                                       max_lag: int = 20) -> np.ndarray:
        """Autocorrelation of the turning angle series.

        A period-4 signal (positive at lag 4N, negative at lag 4N+2)
        indicates the bridge pattern's square-wave structure.

        Uses normalized autocorrelation:
            R(k) = Σ (x[i] - μ)(x[i+k] - μ) / Σ (x[i] - μ)²

        Reference: Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015).
        Time Series Analysis: Forecasting and Control. 5th ed. Wiley.
        """

    def displacement_vectors(self, chain: list[str]) -> np.ndarray:
        """3D displacement vectors between consecutive chain positions.

        Returns array of shape (len(chain)-1, 3).
        """

    def path_length_3d(self, chain: list[str]) -> float:
        """Total Euclidean path length through 3D space."""

    def bounding_box_volume(self, chain: list[str]) -> float:
        """Volume of the axis-aligned bounding box of chain positions.

        Measures how much 3D space the trajectory covers.
        """

    def density_along_path(self, chain: list[str]) -> np.ndarray:
        """Local graph density at each node visited.

        Returns array of node degrees (or clustering coefficients).
        """

    # === Sequential metrics (graph-only, no layout needed) ===

    def bridge_sequences(self, move_types: list[str]) -> list[tuple[int, int]]:
        """Find all I→S→D subsequences (bridge patterns).

        Also supports extended bridge detection via regex on the move
        string: r'I+S*D+' matches one or more insertions, zero or more
        substitutions, one or more deletions. The strict 'ISD' pattern
        is a special case.

        Returns list of (start_index, end_index) tuples.
        """

    def anti_bridge_sequences(self, move_types: list[str]) -> list[tuple[int, int]]:
        """Find all D→S→I subsequences (anti-bridge patterns).

        Extended: r'D+S*I+' for generalized anti-bridges.
        """

    def bridge_rate(self, move_types: list[str]) -> float:
        """Fraction of moves that participate in bridge patterns (I→S→D).

        bridge_rate = (3 × num_bridges) / len(move_types)
        """

    def anti_bridge_rate(self, move_types: list[str]) -> float:
        """Fraction of moves that participate in anti-bridge patterns (D→S→I)."""

    def move_type_distribution(self, move_types: list[str]) -> dict[str, float]:
        """Fraction of insertions, substitutions, and deletions."""

    def word_lengths(self, chain: list[str]) -> np.ndarray:
        """Array of word lengths at each step."""

    def word_length_oscillation(self, chain: list[str]) -> float:
        """Measure of word length oscillation (std dev of word lengths).

        High oscillation indicates bridge-style strategy.
        Low oscillation indicates substitution-only strategy.
        """

    def unique_word_length_range(self, chain: list[str]) -> tuple[int, int]:
        """Min and max word lengths visited."""

    # === Aggregate metrics (across multiple chains) ===

    def model_summary(self, chains: list[list[str]],
                      move_types_list: list[list[str]]) -> dict:
        """Aggregate metrics for a model across all starting words.

        Returns dict with:
          - mean_chain_length, std_chain_length
          - mean_bridge_rate, std_bridge_rate
          - mean_anti_bridge_rate
          - move_type_distribution (aggregate)
          - mean_turning_angle, turning_angle_mode
          - autocorrelation_period4_strength
          - mean_word_length_oscillation
          - mean_bounding_box_volume
        """
```

#### Turning Angle Math (Detail)

Given three consecutive 3D positions `p₁, p₂, p₃`:

```
v₁ = p₂ - p₁    (displacement entering p₂)
v₂ = p₃ - p₂    (displacement leaving p₂)

cos(θ) = (v₁ · v₂) / (|v₁| × |v₂|)
θ = arccos(clamp(cos(θ), -1, 1))    # clamp for numerical stability
```

- θ ≈ 0°: continuing in the same direction
- θ ≈ 90°: right-angle turn (characteristic of bridge pattern)
- θ ≈ 180°: full reversal (backtracking)

#### Autocorrelation for Period Detection

For a signal `x` of length `N` with mean `μ`:

```
R(k) = (1/(N-k)) × Σᵢ₌₀ᴺ⁻ᵏ⁻¹ (x[i] - μ)(x[i+k] - μ)  /  var(x)
```

For the bridge pattern's square wave (period 4):
- R(0) = 1.0 (always)
- R(2) ≈ -0.8 (half-period: anti-correlated)
- R(4) ≈ +0.75 (full period: correlated)
- R(6) ≈ -0.6
- Decaying oscillation with period 4

**Implementation:** Use `numpy.correlate` in "full" mode, or `statsmodels.tsa.stattools.acf`.

---

### 4.4 `wordchain/prompts.py` — Prompting Strategies

```python
CHAIN_SYSTEM_PROMPT = """You are playing a word chain game. You will be given a starting
word. Your goal is to produce the longest possible chain of valid English words where
each consecutive pair differs by exactly one letter operation:

- **Substitution**: Replace one letter (e.g., "cat" → "bat")
- **Insertion**: Add one letter (e.g., "cat" → "cart")
- **Deletion**: Remove one letter (e.g., "cart" → "cat")

Rules:
1. Every word must be a valid, common English word
2. Each consecutive pair must differ by exactly one operation
3. No word may appear more than once
4. Produce as long a chain as possible

Output your chain as a numbered list, one word per line. Nothing else."""

CHAIN_USER_PROMPT = "Starting word: {word}"
```

**Strategy variants to test:**
1. **baseline**: Just the rules, no strategy hints
2. **think_ahead**: Add "Think carefully about which paths lead to longer chains"
3. **plural_hint**: Add "Consider that adding or removing 's' counts as one operation"
4. **explicit_bridge**: Explicitly describe the bridge strategy and ask the model to use it (measures whether models can follow the strategy vs. discovering it organically)

---

### 4.5 `config/chain_config.py` — Starting Words & Difficulty

```python
@dataclass
class StartingWord:
    word: str
    difficulty: str          # "easy", "medium", "hard"
    graph_degree: int        # Number of edit-distance-1 neighbors
    notes: str = ""

STARTING_WORDS: list[StartingWord] = [...]
```

#### Difficulty Tiers

Difficulty is determined by **graph degree** (number of neighbors) and **word length**:

| Tier | Degree | Length | Examples | Why |
|------|--------|--------|----------|-----|
| Easy | 15+ | 3–4 | hat, cat, bat, not, top | Dense neighborhood, many escape routes |
| Medium | 5–14 | 4–6 | lamp, rock, name, stone | Moderate neighbors, some dead ends |
| Hard | 1–4 | 7+ | strange, countries, elephant | Sparse neighborhood, must find rare connections |

**Selection process:**
1. Build the graph
2. Sort words by degree
3. Sample uniformly across difficulty tiers (~17 easy, ~17 medium, ~16 hard)
4. Manual review to ensure words are unambiguous and well-known
5. Total: **50 starting words** (matching LisanBench)

---

### 4.6 `experiments/chain_runner.py` — Experiment Runner

```python
class ChainExperimentRunner:
    """Orchestrates word chain experiments across models."""

    def __init__(self, provider: str, model: str, graph: WordGraph,
                 scorer: ChainScorer, **llm_kwargs):
        """Initialize with LLM provider and prebuilt graph/scorer."""

    async def run_single(self, starting_word: str, strategy: str = "baseline",
                         temperature: float = 0.0, max_tokens: int = 8192,
                         num_samples: int = 1) -> list[ChainResult]:
        """Run the chain task for one starting word, possibly multiple samples.

        Args:
            starting_word: The word to start from
            strategy: Prompting strategy name
            temperature: Sampling temperature (0.0 for greedy by default)
            max_tokens: Max generation tokens (chains can be long)
            num_samples: Number of independent samples per starting word

        Returns:
            List of ChainResult objects
        """

    async def run_benchmark(self, starting_words: list[StartingWord] | None = None,
                            strategy: str = "baseline",
                            temperature: float = 0.0,
                            num_samples: int = 3,
                            save_dir: str = "results_wordchain") -> BenchmarkResult:
        """Run full benchmark across all starting words.

        For each starting word, generates num_samples chains.
        Saves results incrementally to JSON.
        Rate-limited to avoid API throttling.
        """

    def parse_chain(self, raw_response: str) -> list[str]:
        """Parse LLM response into a list of words.

        Handles formats:
        - Numbered lists: "1. cat\n2. bat\n..."
        - Comma-separated: "cat, bat, hat, ..."
        - Arrow-separated: "cat → bat → hat → ..."
        - One word per line
        - Markdown formatting
        """

    def _save_result(self, result: ChainResult, save_dir: str): ...
```

#### Result Schema

```python
@dataclass
class ChainResult:
    """Result of a single word chain generation."""
    # Identification
    model: str
    provider: str
    starting_word: str
    starting_word_difficulty: str
    strategy: str
    temperature: float
    sample_index: int
    timestamp: str

    # Raw output
    raw_response: str

    # Parsed & validated chain
    parsed_chain: list[str]           # As parsed from response
    valid_chain: list[str]            # Truncated at first error
    chain_length: int                 # len(valid_chain) - 1
    is_fully_valid: bool
    invalid_step: int | None
    invalid_reason: str | None
    words_not_in_dict: list[str]
    repeated_words: list[str]

    # Move analysis
    move_types: list[str]             # ['sub', 'ins', 'del', ...]
    move_type_counts: dict[str, int]  # {'sub': 45, 'ins': 12, 'del': 11}

    # Bridge metrics
    bridge_count: int
    anti_bridge_count: int
    bridge_rate: float
    anti_bridge_rate: float

    # Word length metrics
    word_lengths: list[int]
    mean_word_length: float
    word_length_std: float
    word_length_range: tuple[int, int]

    # Trajectory metrics (populated post-hoc with graph layout)
    turning_angles: list[float] | None
    mean_turning_angle: float | None
    autocorrelation: list[float] | None
    bounding_box_volume: float | None
    density_along_path: list[float] | None

    # API metadata
    generation_time: float
    usage: dict | None                # Token counts


@dataclass
class BenchmarkResult:
    """Aggregate result across all starting words for one model."""
    model: str
    provider: str
    strategy: str
    temperature: float
    num_starting_words: int
    num_samples_per_word: int
    results: list[ChainResult]

    # Aggregates
    mean_chain_length: float
    std_chain_length: float
    median_chain_length: float
    total_bridge_rate: float
    total_anti_bridge_rate: float
    move_type_distribution: dict[str, float]
```

#### JSON Storage Format

```json
{
  "model": "claude-opus-4-5-20250901",
  "provider": "anthropic",
  "starting_word": "not",
  "starting_word_difficulty": "easy",
  "strategy": "baseline",
  "temperature": 0.0,
  "sample_index": 0,
  "timestamp": "2026-03-12T14:30:00",
  "raw_response": "1. not\n2. dot\n3. dog\n...",
  "parsed_chain": ["not", "dot", "dog", "..."],
  "valid_chain": ["not", "dot", "dog", "..."],
  "chain_length": 127,
  "is_fully_valid": true,
  "move_types": ["sub", "sub", "ins", "sub", "del", "..."],
  "move_type_counts": {"sub": 98, "ins": 15, "del": 14},
  "bridge_count": 11,
  "bridge_rate": 0.26,
  "word_lengths": [3, 3, 3, 4, 4, 3, "..."],
  "mean_turning_angle": 87.3,
  "generation_time": 12.4,
  "usage": {"input_tokens": 250, "output_tokens": 1500}
}
```

---

## 5. Visualization Specifications

### 5.1 Static Plots (matplotlib/seaborn) — `visualization/trajectory_plots.py`

These extend the existing `plots.py` patterns and reuse `styles.py`.

#### 5.1.1 Chain Length Comparison (Bar Chart)

```python
def chain_length_bar(df: pd.DataFrame,
                     group_by: str = "model",
                     show_error: bool = True) -> Figure:
    """Horizontal bar chart of mean chain length per model.

    Similar to the left panel of statistical_heatmap().
    Error bars show SEM. Sorted by mean score.
    """
```

#### 5.1.2 Bridge Pattern Rate (Grouped Bar Chart)

```python
def bridge_rate_bar(df: pd.DataFrame,
                    show_anti_bridge: bool = True) -> Figure:
    """Grouped bar chart: Bridge (I→S→D) and Anti-Bridge (D→S→I) rates per model.

    Replicates the chart from @scaling01's analysis.
    Teal bars = bridge rate, coral bars = anti-bridge rate.
    X-axis: models sorted by bridge rate descending.
    Y-axis: rate as percentage of total moves.
    """
```

#### 5.1.3 Turning Angle Distribution (Grouped Bar Chart / Histogram)

```python
def turning_angle_distribution(df: pd.DataFrame,
                                models: list[str] | None = None,
                                bins: int = 18) -> Figure:
    """Distribution of turning angles across models.

    X-axis: turning angle (0°–180°) in bins.
    Y-axis: percentage of moves.
    Grouped bars, one color per model.

    Models with bridge patterns will show a spike at ~90°.
    """
```

#### 5.1.4 Word Length Over Chain Steps (Line Chart)

```python
def word_length_over_steps(chains: dict[str, list[str]],
                           starting_word: str,
                           models: list[str] | None = None) -> Figure:
    """Word length at each step of the longest chain per model.

    X-axis: chain step index.
    Y-axis: word length.
    One line per model.

    Bridge-pattern models show square-wave oscillation.
    Substitution-heavy models show flat lines with late jumps.
    """
```

#### 5.1.5 Autocorrelation of Turning Angles (Line Chart)

```python
def turning_angle_autocorrelation(chains: dict[str, list[str]],
                                   graph: WordGraph,
                                   models: list[str] | None = None,
                                   max_lag: int = 20) -> Figure:
    """Autocorrelation plot of turning angle series per model.

    X-axis: lag.
    Y-axis: correlation coefficient.
    One line per model.

    Period-4 signal (peaks at lag 4, 8, 12; troughs at 2, 6, 10)
    indicates bridge pattern.

    Includes reference subplot showing ideal square wave autocorrelation.
    """
```

#### 5.1.6 Move Type Distribution (Stacked Bar)

```python
def move_type_distribution(df: pd.DataFrame) -> Figure:
    """Stacked bar chart showing insertion/substitution/deletion proportions per model.

    100% stacked horizontal bars.
    Colors: substitution=blue, insertion=green, deletion=red.
    """
```

#### 5.1.7 Statistical Heatmap (Reuse Existing)

The existing `statistical_heatmap()` from `plots.py` can be reused directly — just pass chain length scores instead of DAT scores. No modification needed.

#### 5.1.8 Difficulty Scatter (New)

```python
def difficulty_scatter(df: pd.DataFrame) -> Figure:
    """Scatter plot: starting word difficulty (graph degree) vs chain length.

    One point per (model, starting_word) pair.
    Color by model. Shows how models handle easy vs hard starting words.
    Regression lines per model.
    """
```

### 5.2 Interactive Plots (Plotly) — `visualization/trajectory_interactive.py`

These produce interactive HTML files for exploration.

#### 5.2.1 3D Trajectory Plot

```python
def trajectory_3d(graph: WordGraph,
                  chains: dict[str, list[str]],
                  starting_word: str,
                  show_background_graph: bool = True,
                  color_by_density: bool = False,
                  background_opacity: float = 0.1) -> go.Figure:
    """Interactive 3D visualization of word chain trajectories.

    Components:
    1. Background: grey nodes and edges of the word graph
       (subgraph around visited words, radius=2)
    2. Foreground: colored trajectory lines per model
    3. Node labels: word text on hover
    4. Density overlay: edge color mapped to local graph density

    Args:
        graph: WordGraph with precomputed 3D layout
        chains: Dict mapping model_name → word chain
        starting_word: The starting word (for title and subgraph extraction)
        show_background_graph: Whether to render background graph structure
        color_by_density: Color edges by local graph density
        background_opacity: Opacity of background elements

    Returns:
        Plotly Figure object (can be exported to HTML)

    Implementation notes:
    - Use go.Scatter3d with mode='lines+markers' for trajectories
    - Use go.Scatter3d with mode='lines' and opacity=0.1 for background edges
    - Use go.Scatter3d with mode='markers+text' for labeled nodes
    - Set hovertemplate to show word, degree, step number
    - Camera default: isometric view, with orbit controls
    """
```

#### 5.2.2 3D Trajectory with Animation

```python
def trajectory_3d_animated(graph: WordGraph,
                           chain: list[str],
                           model_name: str,
                           fps: int = 5) -> go.Figure:
    """Animated 3D trajectory showing the chain being built step by step.

    Uses Plotly animation frames. Each frame adds one more edge to the path.
    Play/pause controls. Slider for manual scrubbing.
    """
```

#### 5.2.3 Interactive Dashboard

```python
def chain_dashboard(results_dir: str,
                    graph: WordGraph) -> None:
    """Generate a multi-page HTML dashboard with all visualizations.

    Pages:
    1. Leaderboard: sortable table of model scores
    2. Trajectory explorer: 3D plots with model/word selectors
    3. Strategy analysis: bridge rates, move distributions
    4. Statistical comparison: heatmaps, significance tests

    Uses Plotly subplots + dropdown menus for interactivity.
    Exports as self-contained HTML.
    """
```

### 5.3 Style Integration

Extend `styles.py`:

```python
# Add to MODEL_SPECIFIC_COLORS:
CHAIN_MODEL_COLORS = {
    "claude-opus-4.5": "#E8853D",    # Orange (matching @scaling01's choice)
    "gpt-5.4": "#4ECDC4",            # Teal
    "gemini-3.1-pro": "#FF6B6B",     # Coral
    "deepseek-v3.2": "#45B7D1",      # Light blue
    # ... extend as needed
}

# Plotly-specific dark theme (matching @scaling01's aesthetic):
PLOTLY_DARK_THEME = {
    "paper_bgcolor": "#1a1a2e",
    "plot_bgcolor": "#16213e",
    "font_color": "#e0e0e0",
    "gridcolor": "#2a2a4a",
}
```

---

## 6. CLI Interface

### 6.1 `scripts/build_graph.py`

```bash
# Build and cache the word graph (one-time setup, ~2-5 minutes)
uv run python scripts/build_graph.py \
    --wordlist data/wordlist/words_en.txt \
    --output data/graph/ \
    --layout spring \
    --layout-iterations 100 \
    --seed 42

# Output:
#   data/graph/word_graph.gpickle    (~50MB)
#   data/graph/layout_3d.npy         (~5MB)
#   data/graph/graph_stats.json      (node count, edge count, degree distribution)
```

### 6.2 `scripts/run_wordchain.py`

```bash
# Run single model on all starting words
uv run python scripts/run_wordchain.py \
    --provider openai \
    --model gpt-5.4 \
    --strategy baseline \
    --temperature 0.0 \
    --samples 3 \
    --output results_wordchain/

# Run specific starting word (useful for debugging)
uv run python scripts/run_wordchain.py \
    --provider anthropic \
    --model claude-opus-4-5-20250901 \
    --word "not" \
    --samples 1

# Batch run: all strategies × multiple temperatures
uv run python scripts/run_wordchain.py \
    --provider openai \
    --model gpt-5.4 \
    --batch \
    --samples 3
```

### 6.3 `scripts/analyze_chains.py`

```bash
# Generate all static visualizations
uv run python scripts/analyze_chains.py \
    --results results_wordchain/ \
    --graph data/graph/ \
    --output visualizations_wordchain/ \
    --all

# Generate specific plots
uv run python scripts/analyze_chains.py \
    --results results_wordchain/ \
    --plot bridge_rate \
    --plot turning_angles \
    --plot word_length

# Generate interactive 3D HTML
uv run python scripts/analyze_chains.py \
    --results results_wordchain/ \
    --graph data/graph/ \
    --interactive \
    --word "not" \
    --output visualizations_wordchain/trajectory_not.html

# Full interactive dashboard
uv run python scripts/analyze_chains.py \
    --results results_wordchain/ \
    --graph data/graph/ \
    --dashboard \
    --output visualizations_wordchain/dashboard.html

# Print leaderboard to terminal
uv run python scripts/analyze_chains.py \
    --results results_wordchain/ \
    --leaderboard
```

---

## 7. Dependencies (Additions to pyproject.toml)

```toml
[project]
dependencies = [
    # ... existing deps ...
    "networkx>=3.0",          # Graph construction and algorithms
    "plotly>=5.18.0",         # Interactive 3D visualizations
    "kaleido>=0.2.1",         # Plotly static image export
    "nltk>=3.8.0",            # Word list (words corpus)
    "statsmodels>=0.14.0",    # Autocorrelation functions (acf)
]
```

**Note:** `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn` are already dependencies.

---

## 8. Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Set up `wordchain/` module structure
- [ ] Implement `graph.py`: word graph construction with bucket method
- [ ] Implement `scorer.py`: chain validation and move classification
- [ ] Select dictionary, build graph, verify statistics
- [ ] Write unit tests for graph construction and chain validation
- [ ] Select 50 starting words across difficulty tiers

### Phase 2: Experiment Runner (Week 2)
- [ ] Implement `prompts.py`: baseline and variant strategies
- [ ] Implement `chain_runner.py`: LLM interaction and chain parsing
- [ ] Implement `chain_config.py`: starting words and difficulty metadata
- [ ] Run initial experiments with 2-3 models to validate pipeline
- [ ] Write unit tests for parsing and experiment flow

### Phase 3: Metrics & Analysis (Week 3)
- [ ] Implement `metrics.py`: all trajectory metrics
- [ ] Bridge pattern detection (I→S→D sequences)
- [ ] Turning angle computation and autocorrelation
- [ ] Word length oscillation metrics
- [ ] Aggregate model summaries
- [ ] Write unit tests for metric calculations

### Phase 4: Static Visualizations (Week 4)
- [ ] `trajectory_plots.py`: chain length bars, bridge rate bars
- [ ] Turning angle distribution, word length over steps
- [ ] Autocorrelation plots, move type distribution
- [ ] Difficulty scatter plot
- [ ] Integrate with existing `styles.py`
- [ ] Reuse `statistical_heatmap()` for model comparison

### Phase 5: Interactive Visualizations (Week 5)
- [ ] `trajectory_interactive.py`: 3D trajectory plots
- [ ] Animated trajectory builder
- [ ] Interactive dashboard with model/word selectors
- [ ] HTML export
- [ ] Dark theme styling matching reference visualizations

### Phase 6: Full Benchmark Run & Polish (Week 6)
- [ ] Run full benchmark across all target models
- [ ] Generate complete visualization suite
- [ ] Write analysis report
- [ ] Update README with word chain benchmark documentation
- [ ] Performance optimization (graph caching, parallel API calls)

---

## 9. Testing Strategy

```
tests/
├── test_graph.py              # Graph construction, neighbor lookup, layout
├── test_scorer.py             # Chain validation, move classification, edge cases
├── test_metrics.py            # Turning angles, bridges, autocorrelation
├── test_chain_runner.py       # Parsing, API mocking, result serialization
└── test_trajectory_plots.py   # Plot generation smoke tests
```

### Key Test Cases

**Graph:**
- Known neighbors: `cat` should have `bat, hat, mat, sat, fat, rat, ...` as substitution neighbors and `cats, cart, cast, coat, chat, ...` as insertion neighbors
- Symmetry: if A is neighbor of B, B is neighbor of A
- Edge operation labels are correct
- Graph is connected (largest component >90%)

**Scorer:**
- Valid chain scores correctly
- Chain with invalid word truncates at correct step
- Chain with repeated word truncates at correct step
- Move classification: `name→names` = insertion, `names→games` = substitution, `games→game` = deletion
- Empty chain, single word, all same word

**Metrics:**
- Known bridge sequence `[I, S, D]` detected
- Turning angle of 90° for orthogonal vectors
- Autocorrelation of perfect square wave matches expected pattern
- Word length oscillation of `[4,5,4,5,4,5]` has high oscillation score

**Runner:**
- Parse numbered list format
- Parse arrow-separated format
- Parse comma-separated format
- Handle malformed responses gracefully
- Result JSON serialization round-trips correctly

---

## 10. Model Target List

Priority models for initial benchmark:

| Model | Provider | Why |
|-------|----------|-----|
| Claude Opus 4.6 | Anthropic | Current frontier, successor to bridge-pattern Opus 4.5 |
| Claude Sonnet 4.6 | Anthropic | Cost-efficient, check if bridge pattern persists |
| GPT-5.4 | OpenAI | Frontier comparison, known non-bridge strategy |
| GPT-5-mini | OpenAI | Smaller reasoning model |
| Gemini 3.1 Pro | Google (OpenRouter) | Third-party frontier |
| DeepSeek V3.2 Speciale | DeepSeek (OpenRouter) | Strong open-weight model |
| Grok-4 | xAI (OpenRouter) | Reported bridge pattern by @scaling01 |
| Llama 3.2 3B | Ollama | Small local baseline |
| Qwen3-4B | Ollama | Small local baseline |

---

## 11. Reuse Map from Existing `divergent_bench`

| Module | Reusability | Action |
|--------|-------------|--------|
| `llm/providers.py` | **100% reusable** | Use as-is. Define `WordChain(BaseModel)` for structured output. |
| `llm/config.py` | **100% reusable** | Use as-is for provider configuration. |
| `visualization/styles.py` | **100% reusable** | Extend `MODEL_SPECIFIC_COLORS` with new models. |
| `visualization/loader.py` | **Mostly reusable** | Parameterize `required_cols` (currently hardcodes `words`). Add chain-specific columns. |
| `visualization/plots.py` | **Partially reusable** | `ridge_plot`, `statistical_heatmap`, `score_distribution` work for any numeric score. `word_frequency_stacked` adaptable. `triangular_matrix` is DAT-specific. |
| `experiments/runner.py` | **Pattern reusable** | Copy async skeleton, incremental save, batch structure. Replace `DATWords`, `_parse_word_list`, scorer, strategies with chain equivalents. |
| `config/strategies.py` | **Pattern reusable** | New file `chain_config.py` following same pattern. |

## 12. Known LLM Failure Modes

Based on @scaling01's work and general word-ladder benchmarking:

| Failure Mode | Description | Mitigation |
|---|---|---|
| **Hallucinated words** | Model produces words not in any dictionary | Validate against word list; truncate at first invalid word |
| **Edit distance > 1** | Model changes 2+ letters in one step | Explicit edit distance check per step |
| **Substitution-only thinking** | Model forgets insertion/deletion are allowed | Few-shot examples showing all three operations |
| **Early termination** | Model stops at 15-30 words when longer chains exist | Explicit "produce the longest possible chain" instruction; consider continuation prompting |
| **Repetition** | Model revisits a word in long chains | Validate no-repeat constraint |
| **Format errors** | Model adds commentary, numbering inconsistencies | Robust parser handling multiple formats |

## 13. Open Questions

1. **Dictionary choice:** Should we use the exact same dictionary as LisanBench for comparability, or curate our own? If @scaling01 publishes their word list, we should use it.

2. **Temperature:** Should we benchmark at temperature 0 (greedy, reproducible) or temperature > 0 (stochastic, multiple samples)? LisanBench appears to use multiple trials per starting word, taking the longest.

3. **Max tokens:** Long chains (200+ words) require significant output tokens. Budget ~8K–16K tokens per generation. Some models may hit output limits.

4. **Scoring policy:** Do we score the raw parsed chain, or the validated chain (truncated at first error)? Validated chain is more rigorous but penalizes models that make one mistake mid-chain.

5. **Graph layout stability:** Force-directed layouts are non-deterministic. We must fix the random seed and precompute once. All trajectory analysis depends on consistent positions.

6. **Contamination:** Models may have seen LisanBench results or word ladder strategies in training data. Consider including novel starting words not in LisanBench.

---

## 12. References

1. Olson, J.A., Nahas, J., Chmoulevitch, D., Cropper, S.J., & Webb, M.E. (2021). "Naming unrelated words predicts creativity." *PNAS*, 118(25), e2022340118.

2. @scaling01. (2026, March 11). "The Bridge Pattern: How Opus 4.5 Exploits Word Graph Topology." X/Twitter. https://x.com/scaling01/status/2031855619360088558

3. Knuth, D.E. (1993). *The Stanford GraphBase: A Platform for Combinatorial Computing.* ACM Press.

4. Levenshtein, V.I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals." *Soviet Physics Doklady*, 10(8), 707–710.

5. Fruchterman, T.M.J. & Reingold, E.M. (1991). "Graph drawing by force-directed placement." *Software: Practice and Experience*, 21(11), 1129–1164.

6. Kamada, T. & Kawai, S. (1989). "An algorithm for drawing general undirected graphs." *Information Processing Letters*, 31(1), 7–15.

7. Pennington, J., Socher, R., & Manning, C.D. (2014). "GloVe: Global Vectors for Word Representation." *EMNLP 2014*.

8. Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). *Time Series Analysis: Forecasting and Control.* 5th ed. Wiley.

9. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences.* 2nd ed. Lawrence Erlbaum.

10. NetworkX Developers. (2024). "NetworkX: Network Analysis in Python." https://networkx.org/

11. Plotly Technologies Inc. (2024). "Plotly Python Graphing Library." https://plotly.com/python/
