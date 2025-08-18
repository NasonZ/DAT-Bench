"""
DAT Scorer - Core implementation for Divergent Association Task.
Based on DAT_GPT/scripts/dat.py with minimal modifications for configurability.
Original copyright 2021 Jay Olson.
"""

import re
import itertools
import os
import logging
from pathlib import Path
from typing import List, Optional
import numpy
import scipy.spatial.distance

logger = logging.getLogger(__name__)


class DATScorer:
    """Compute score for Divergent Association Task."""
    
    def __init__(self, model_path=None, dictionary_path=None, pattern="^[a-z][a-z-]*[a-z]$"):
        """
        Initialize DAT scorer.
        
        Args:
            model_path: Path to GloVe embeddings. If None, looks in standard locations.
            dictionary_path: Path to word dictionary. If None, looks in standard locations.
            pattern: Regex pattern for valid words.
        """
        # Find model path
        if model_path is None:
            model_path = os.getenv("GLOVE_PATH")
            if model_path is None or not Path(model_path).exists():
                # Try standard locations
                for candidate in [
                    "/root/projects/divergent_thinking/divergent-association-task/glove.840B.300d.txt",
                    "data/embeddings/glove.840B.300d.txt",
                    "glove.840B.300d.txt"
                ]:
                    if Path(candidate).exists():
                        model_path = candidate
                        break
        
        # Find dictionary path
        if dictionary_path is None:
            dictionary_path = os.getenv("WORDS_PATH")
            if dictionary_path is None or not Path(dictionary_path).exists():
                # Try standard locations
                for candidate in [
                    "/root/projects/divergent_thinking/divergent-association-task/words.txt",
                    "data/words.txt",
                    "words.txt"
                ]:
                    if Path(candidate).exists():
                        dictionary_path = candidate
                        break
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Loading dictionary from {dictionary_path}")
        
        # Keep unique words matching pattern from dictionary
        words = set()
        with open(dictionary_path, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))
        
        # Load vectors for words in dictionary
        self.vectors = {}
        with open(model_path, "r", encoding="utf8") as f:
            for line in f:
                tokens = line.split(" ")
                word = tokens[0]
                if word in words:
                    vector = numpy.asarray(tokens[1:], "float32")
                    self.vectors[word] = vector
        
        logger.info(f"Loaded {len(self.vectors)} word vectors")
    
    def validate(self, word):
        """Clean up word and find best candidate to use."""
        # Strip unwanted characters
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        if len(clean) <= 1:
            return None  # Word too short
        
        # Generate candidates for possible compound words
        # "valid" -> ["valid"]
        # "cul de sac" -> ["cul-de-sac", "culdesac"]
        # "top-hat" -> ["top-hat", "tophat"]
        candidates = []
        if " " in clean:
            candidates.append(re.sub(r" +", "-", clean))
            candidates.append(re.sub(r" +", "", clean))
        else:
            candidates.append(clean)
            if "-" in clean:
                candidates.append(re.sub(r"-+", "", clean))
        
        for cand in candidates:
            if cand in self.vectors:
                return cand  # Return first word that is in model
        return None  # Could not find valid word
    
    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words."""
        return scipy.spatial.distance.cosine(
            self.vectors.get(word1), 
            self.vectors.get(word2)
        )
    
    def dat(self, words, minimum=7):
        """
        Compute DAT score.
        
        Args:
            words: List of words to score.
            minimum: Minimum number of valid words required.
            
        Returns:
            DAT score (0-200) or None if insufficient valid words.
        """
        # Keep only valid unique words
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)
        
        logger.info(f'Number of valid words: {len(uniques)}')
        
        # Keep subset of words
        if len(uniques) >= minimum:
            subset = uniques[:minimum]
        else:
            return None  # Not enough valid words
        
        # Compute distances between each pair of words
        distances = []
        for word1, word2 in itertools.combinations(subset, 2):
            dist = self.distance(word1, word2)
            distances.append(dist)
        
        self.distances = distances  # Store for analysis
        
        # Compute the DAT score (average semantic distance multiplied by 100)
        return (sum(distances) / len(distances)) * 100

