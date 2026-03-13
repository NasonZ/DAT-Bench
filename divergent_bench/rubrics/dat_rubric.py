"""
DAT Rubric — verifiers-compatible reward function wrapping DATScorer.

Provides normalized DAT score as a reward signal (0-1) plus raw metrics
for analysis. Lazy-loads GloVe embeddings on first use.
"""

import re
import logging

import verifiers as vf

from ..dat.scorer import DATScorer

logger = logging.getLogger(__name__)

# Normalization bounds for mapping raw DAT scores (0-200) to [0, 1].
# Empirically, model scores cluster in 40-100; this maps that to ~0.0-1.0.
_NORM_LOW = 40.0
_NORM_HIGH = 100.0


class DATRubric(vf.Rubric):
    """Rubric that scores model completions via the Divergent Association Task.

    Reward functions:
      - dat_score (weight=1.0): normalized DAT score in [0, 1]
    Metrics (weight=0, tracked only):
      - num_valid_words: how many of the 10 words are in the GloVe vocabulary
      - raw_dat_score: unnormalized score (0-200 scale)
    """

    def __init__(self, strategy: str = "none", minimum_words: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self.minimum_words = minimum_words
        self._scorer: DATScorer | None = None

        # Primary reward
        self.add_reward_func(self._dat_score, weight=1.0)
        # Tracked metrics (don't affect reward)
        self.add_metric(self._num_valid_words)
        self.add_metric(self._raw_dat_score)

    @property
    def scorer(self) -> DATScorer:
        """Lazy-load scorer — GloVe 840B takes ~30s to load."""
        if self._scorer is None:
            logger.info("Loading DATScorer (GloVe 840B)...")
            self._scorer = DATScorer()
        return self._scorer

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    async def _dat_score(self, completion, state, **kwargs) -> float:
        """Normalized DAT score as reward (0.0 to 1.0)."""
        words = _extract_words(completion)
        state["dat_words"] = words

        raw = self.scorer.dat(words, minimum=self.minimum_words)
        if raw is None:
            state["raw_dat_score"] = 0.0
            return 0.0

        state["raw_dat_score"] = float(raw)
        # Linear map: [_NORM_LOW, _NORM_HIGH] → [0, 1], clamped
        normalized = (raw - _NORM_LOW) / (_NORM_HIGH - _NORM_LOW)
        return max(0.0, min(1.0, float(normalized)))

    async def _num_valid_words(self, state, **kwargs) -> float:
        """Count of words that exist in the GloVe vocabulary."""
        words = state.get("dat_words", [])
        return float(sum(1 for w in words if self.scorer.validate(w) is not None))

    async def _raw_dat_score(self, state, **kwargs) -> float:
        """Unnormalized DAT score (0-200 scale)."""
        return float(state.get("raw_dat_score", 0.0))


# ------------------------------------------------------------------
# Word extraction (reuses logic from ExperimentRunner._parse_word_list)
# ------------------------------------------------------------------

def _extract_words(completion) -> list[str]:
    """Extract up to 10 words from a verifiers completion (list[Message]).

    Handles numbered lists, bullets, comma-separated, markdown bold, and
    word-with-description formats.
    """
    if not completion:
        return []

    # Get text content from the last assistant message
    last = completion[-1]
    if isinstance(last, dict):
        content = last.get("content", "")
    elif hasattr(last, "content"):
        content = last.content or ""
    else:
        content = str(last)

    # Handle list[ContentPart] content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                parts.append(part.text)
        content = "\n".join(parts)

    if not isinstance(content, str) or not content.strip():
        return []

    # Strip markdown bold/italic
    cleaned = re.sub(r"\*+([^*]+)\*+", r"\1", content)
    # Strip numbered list markers
    cleaned = re.sub(r"^\d+[.):\s]+", "", cleaned, flags=re.MULTILINE)
    # Strip bullets
    cleaned = re.sub(r"^[-*\u2022]\s*", "", cleaned, flags=re.MULTILINE)

    words: list[str] = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Extract word before description separators
        if " - " in line:
            word = line.split(" - ")[0].strip()
        elif ":" in line:
            word = line.split(":")[0].strip()
        else:
            word = line

        # Clean punctuation, lowercase
        word = re.sub(r"[^\w\s-]", "", word).strip().lower()

        # Filter meta-text lines
        skip_tokens = {"provide", "diverse", "words", "nouns", "here", "list", "following"}
        if word and len(word) > 1 and not any(s in word for s in skip_tokens):
            words.append(word)

    # Fallback: comma-separated
    if len(words) < 5 and "," in content:
        comma_words = [
            re.sub(r"[^\w\s-]", "", w).strip().lower()
            for w in content.split(",")
        ]
        comma_words = [w for w in comma_words if w and len(w) > 1 and w.isalpha()]
        if len(comma_words) >= len(words):
            words = comma_words

    return words[:10]
