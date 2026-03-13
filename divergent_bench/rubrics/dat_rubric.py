"""
DAT Rubric — verifiers-compatible composite reward for the Divergent Association Task.

Reward signals:
  creativity_reward (weight=1.0) — normalized DAT score, the primary training signal
  validity_reward   (weight=0.2) — fraction of 10 words that are valid GloVe entries
  format_reward     (weight=0.1) — XMLParser format compliance (free from verifiers)

Tracked metrics (weight=0.0, don't affect reward):
  raw_dat_score     — unnormalized DAT score (0–200 scale) for analysis
  valid_word_count  — integer count of valid words

Parsing strategy (extract_words):
  1. XML  — XMLParser extracts <words>...</words> (training path)
  2. JSON — parses structured output responses (eval path)
  3. Regex — numbered lists / comma-separated / bullets (fallback)

Structured output workaround (verifiers ≤0.1.2):
  When response_format is set to a Pydantic model, the OpenAI/Anthropic SDKs
  return the parsed object at message.parsed and the JSON serialization at
  message.content. Verifiers currently only reads message.content
  (see verifiers/utils/response_utils.py:get_chat_response_fields), so the
  rubric receives a JSON string, not a Pydantic object. The JSON extraction
  step (_extract_words_from_json) handles this. When verifiers adds native
  structured output support (surfacing message.parsed), this workaround can
  be removed and extract_words simplified.
"""

import json
import re
import logging
from typing import Optional

import verifiers as vf

from ..dat.scorer import DATScorer

logger = logging.getLogger(__name__)

# Linear normalization bounds: raw DAT 40–100 → reward 0.0–1.0
_NORM_LOW = 40.0
_NORM_HIGH = 100.0


def _extract_words_from_text(text: str) -> list[str]:
    """Extract up to 10 words from free-form text.

    Handles numbered lists, bullets, comma-separated, markdown bold,
    and word-with-description formats. Mirrors ExperimentRunner._parse_word_list
    so scoring is consistent between the old pipeline and verifiers.
    """
    if not text or not text.strip():
        return []

    cleaned = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    cleaned = re.sub(r"^\d+[.):\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[-*\u2022]\s*", "", cleaned, flags=re.MULTILINE)

    skip_tokens = {"provide", "diverse", "words", "nouns", "here", "list", "following"}
    words: list[str] = []

    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue

        if " - " in line:
            word = line.split(" - ")[0].strip()
        elif ":" in line:
            word = line.split(":")[0].strip()
        else:
            word = line

        word = re.sub(r"[^\w\s-]", "", word).strip().lower()
        if word and len(word) > 1 and not any(s in word for s in skip_tokens):
            words.append(word)

    # Fallback: comma-separated
    if len(words) < 5 and "," in text:
        comma_words = [
            re.sub(r"[^\w\s-]", "", w).strip().lower() for w in text.split(",")
        ]
        comma_words = [w for w in comma_words if w and len(w) > 1 and w.isalpha()]
        if len(comma_words) >= len(words):
            words = comma_words

    return words[:10]


def _extract_words_from_json(text: str) -> list[str] | None:
    """Extract words from a JSON string (structured output response).

    When using response_format with a Pydantic model, verifiers surfaces
    message.content which is the JSON serialization. The parsed Pydantic
    object lives at message.parsed but verifiers doesn't pass it through.
    """
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "words" in data:
            words = data["words"]
            if isinstance(words, list):
                return [w.strip().lower() for w in words if isinstance(w, str)]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _get_completion_text(completion) -> str:
    """Extract text content from the last assistant message in a completion."""
    if not completion:
        return ""

    last = completion[-1] if isinstance(completion, list) else completion
    if isinstance(last, dict):
        content = last.get("content", "")
    elif hasattr(last, "content"):
        content = last.content or ""
    else:
        content = str(last)

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                parts.append(part.text)
        content = "\n".join(parts)

    return content if isinstance(content, str) else ""


class DATRubric(vf.Rubric):
    """Composite rubric for the Divergent Association Task.

    Uses XMLParser to extract words from <words>...</words> tags, with regex
    fallback for robustness. Reward functions are non-async (verifiers
    call_reward_func doesn't await).

    Args:
        parser: XMLParser instance (created automatically if None).
        strategy: DAT prompting strategy name (stored in state for logging).
        minimum_words: Minimum valid GloVe words required for scoring.
    """

    def __init__(
        self,
        parser: Optional[vf.XMLParser] = None,
        strategy: str = "none",
        minimum_words: int = 7,
        **kwargs,
    ):
        parser = parser or vf.XMLParser(fields=["words"], answer_field="words")
        super().__init__(parser=parser, **kwargs)

        self.strategy = strategy
        self.minimum_words = minimum_words
        self._scorer: Optional[DATScorer] = None

        # Primary reward signals
        self.add_reward_func(self._creativity_reward, weight=1.0)
        self.add_reward_func(self._validity_reward, weight=0.2)
        self.add_reward_func(parser.get_format_reward_func(), weight=0.1)

        # Tracked metrics (weight=0.0 → don't affect training reward)
        self.add_reward_func(self._raw_dat_score_metric, weight=0.0)
        self.add_reward_func(self._valid_word_count_metric, weight=0.0)

    @property
    def scorer(self) -> DATScorer:
        """Lazy-load scorer — GloVe 840B takes ~30s on first call."""
        if self._scorer is None:
            logger.info("Loading DATScorer (GloVe 840B)...")
            self._scorer = DATScorer()
        return self._scorer

    # ------------------------------------------------------------------
    # Word extraction: XML first, regex fallback
    # ------------------------------------------------------------------

    def extract_words(self, completion) -> list[str]:
        """Extract words from completion. Tries in order:

        1. XMLParser — extracts <words>...</words> content
        2. JSON — handles structured output (Pydantic DATWords responses)
        3. Regex — numbered lists, bullets, comma-separated, etc.
        """
        # 1. XML extraction
        answer = self.parser.parse_answer(completion)
        if answer:
            words = _extract_words_from_text(answer)
            if len(words) >= 5:
                return words

        # 2. JSON extraction (structured output responses)
        text = _get_completion_text(completion)
        json_words = _extract_words_from_json(text)
        if json_words and len(json_words) >= 5:
            return json_words[:10]

        # 3. Regex fallback
        return _extract_words_from_text(text)

    # ------------------------------------------------------------------
    # Reward functions (sync — verifiers doesn't await these)
    # ------------------------------------------------------------------

    def _creativity_reward(self, completion, state, **kwargs) -> float:
        """Normalized DAT score as primary reward (0.0–1.0)."""
        words = self.extract_words(completion)
        state["dat_words"] = words

        raw = self.scorer.dat(words, minimum=self.minimum_words)
        if raw is None:
            state["raw_dat_score"] = 0.0
            return 0.0

        state["raw_dat_score"] = float(raw)
        normalized = (raw - _NORM_LOW) / (_NORM_HIGH - _NORM_LOW)
        return max(0.0, min(1.0, float(normalized)))

    def _validity_reward(self, state, **kwargs) -> float:
        """Fraction of extracted words that are valid GloVe entries (0.0–1.0)."""
        words = state.get("dat_words", [])
        if not words:
            return 0.0
        valid_count = sum(1 for w in words if self.scorer.validate(w) is not None)
        return valid_count / len(words)

    # ------------------------------------------------------------------
    # Tracked metrics (weight=0.0)
    # ------------------------------------------------------------------

    def _raw_dat_score_metric(self, state, **kwargs) -> float:
        """Unnormalized DAT score (0–200 scale)."""
        return float(state.get("raw_dat_score", 0.0))

    def _valid_word_count_metric(self, state, **kwargs) -> float:
        """Count of words in the GloVe vocabulary."""
        words = state.get("dat_words", [])
        return float(sum(1 for w in words if self.scorer.validate(w) is not None))
