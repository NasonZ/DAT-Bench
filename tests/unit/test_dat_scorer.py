"""Test suite for DAT scorer - defining expected behavior."""

import pytest
import numpy as np
from pathlib import Path
import os
import sys

from divergent_bench.dat.scorer import DATScorer


class TestDATScorer:
    """Test the DAT scoring functionality."""
    
    @pytest.fixture
    def mock_glove_path(self, tmp_path):
        """Create a mock GloVe file for testing."""
        glove_file = tmp_path / "test_glove.txt"
        # Create minimal GloVe embeddings for testing
        # Need at least 10 dimensions for the code to accept them
        embeddings = [
            "cat 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
            "dog 0.1 0.2 0.4 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
            "bird 0.1 0.3 0.4 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
            "quantum 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0",
            "pizza 0.5 0.6 0.7 0.8 0.9 0.1 0.2 0.3 0.4 0.5",
            "democracy 0.3 0.7 0.9 0.2 0.4 0.6 0.8 0.1 0.3 0.5",
            "volcano 0.8 0.2 0.1 0.9 0.7 0.5 0.3 0.1 0.9 0.7",
            "jazz 0.4 0.5 0.6 0.7 0.8 0.9 0.1 0.2 0.3 0.4",
            "bacteria 0.7 0.8 0.9 0.1 0.2 0.3 0.4 0.5 0.6 0.7",
            "happiness 0.2 0.8 0.5 0.3 0.9 0.6 0.4 0.1 0.7 0.5",
            "test-word 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
        ]
        glove_file.write_text("\n".join(embeddings))
        return str(glove_file)
    
    @pytest.fixture
    def mock_words_path(self, tmp_path):
        """Create a mock words dictionary for testing."""
        words_file = tmp_path / "test_words.txt"
        words = ["cat", "dog", "bird", "quantum", "pizza", "democracy", 
                 "volcano", "jazz", "bacteria", "happiness", "test-word"]
        words_file.write_text("\n".join(words))
        return str(words_file)
    
    def test_initialization_with_paths(self, mock_glove_path, mock_words_path):
        """Test that DATScorer can be initialized with custom paths."""
        scorer = DATScorer(
            model_path=mock_glove_path,
            dictionary_path=mock_words_path
        )
        assert scorer is not None
        assert len(scorer.vectors) > 0
        assert "cat" in scorer.vectors
        assert "quantum" in scorer.vectors
    
    def test_initialization_with_env_vars(self, mock_glove_path, mock_words_path, monkeypatch):
        """Test initialization using environment variables."""
        monkeypatch.setenv("GLOVE_PATH", mock_glove_path)
        monkeypatch.setenv("WORDS_PATH", mock_words_path)
        
        scorer = DATScorer()
        assert scorer is not None
        assert len(scorer.vectors) > 0
    
    def test_validate_word_basic(self, mock_glove_path, mock_words_path):
        """Test word validation for basic cases."""
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        # Valid words
        assert scorer.validate("cat") == "cat"
        assert scorer.validate("CAT") == "cat"  # Case insensitive
        assert scorer.validate("  cat  ") == "cat"  # Strips whitespace
        
        # Invalid words
        assert scorer.validate("x") is None  # Too short
        assert scorer.validate("") is None  # Empty
        assert scorer.validate("notindict") is None  # Not in dictionary
    
    def test_validate_compound_words(self, mock_glove_path, mock_words_path):
        """Test validation of compound words."""
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        # Test hyphenated word (assuming "test-word" is in vectors)
        assert scorer.validate("test word") in ["test-word", "testword"]
        assert scorer.validate("test-word") == "test-word"
    
    def test_distance_calculation(self, mock_glove_path, mock_words_path):
        """Test cosine distance calculation between words."""
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        # Distance between same word should be 0
        dist_same = scorer.distance("cat", "cat")
        assert dist_same == pytest.approx(0, abs=0.001)
        
        # Distance between different words should be > 0
        dist_diff = scorer.distance("cat", "dog")
        assert dist_diff > 0
        assert dist_diff <= 2  # Cosine distance range is 0-2
        
        # Distance between very different concepts should be larger
        dist_far = scorer.distance("cat", "quantum")
        dist_near = scorer.distance("cat", "dog")
        assert dist_far > dist_near
    
    def test_dat_score_calculation(self, mock_glove_path, mock_words_path):
        """Test DAT score calculation."""
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        # Test with similar words (low creativity)
        similar_words = ["cat", "dog", "bird"]
        score_low = scorer.dat(similar_words, minimum=3)
        assert score_low is not None
        assert 0 <= score_low <= 200  # Score is percentage of max distance
        
        # Test with diverse words (high creativity)
        diverse_words = ["cat", "quantum", "democracy", "volcano", "jazz", "bacteria", "happiness"]
        score_high = scorer.dat(diverse_words)
        assert score_high is not None
        assert score_high > score_low  # Diverse words should score higher
    
    def test_dat_minimum_words(self, mock_glove_path, mock_words_path):
        """Test that DAT requires minimum number of valid words."""
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        # Too few words
        few_words = ["cat", "dog"]
        assert scorer.dat(few_words, minimum=7) is None
        
        # Enough words
        enough_words = ["cat", "dog", "bird", "quantum", "pizza", "democracy", "volcano"]
        assert scorer.dat(enough_words, minimum=7) is not None
    
    def test_dat_removes_duplicates(self, mock_glove_path, mock_words_path):
        """Test that DAT removes duplicate words."""
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        # Words with duplicates - need at least 7 unique after deduplication
        words_with_dupes = ["cat", "dog", "cat", "bird", "dog", "quantum", "pizza", "democracy", "volcano", "jazz"]
        score = scorer.dat(words_with_dupes)
        
        # Should still calculate score if unique words >= minimum
        assert score is not None
    
    def test_dat_with_invalid_words(self, mock_glove_path, mock_words_path):
        """Test DAT handling of invalid words mixed with valid ones."""
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        # Mix of valid and invalid words
        mixed_words = ["cat", "invalidword", "dog", "xyz", "bird", "quantum", 
                       "pizza", "democracy", "volcano", "notreal"]
        score = scorer.dat(mixed_words)
        
        # Should still work if enough valid words remain
        assert score is not None
    
    def test_logging_output(self, mock_glove_path, mock_words_path, caplog):
        """Test that appropriate logging is produced."""
        import logging
        
        scorer = DATScorer(mock_glove_path, mock_words_path)
        
        with caplog.at_level(logging.INFO):
            words = ["cat", "dog", "bird"]
            scorer.dat(words, minimum=3)
            
        # Check that we logged the number of valid words
        assert "valid words" in caplog.text.lower()


class TestDATScorerIntegration:
    """Integration tests with real GloVe model (if available)."""
    
    @pytest.mark.skipif(
        not Path("/root/projects/divergent_thinking/divergent-association-task/glove.840B.300d.txt").exists(),
        reason="GloVe model not available"
    )
    def test_with_real_glove(self):
        """Test with actual GloVe embeddings if available."""
        glove_path = "/root/projects/divergent_thinking/divergent-association-task/glove.840B.300d.txt"
        words_path = "/root/projects/divergent_thinking/divergent-association-task/words.txt"
        
        scorer = DATScorer(glove_path, words_path)
        
        # Test known word sets
        low_creativity = ["arm", "leg", "hand", "foot", "head", "shoulder", "knee"]
        high_creativity = ["quantum", "pizza", "democracy", "cactus", "symphony", 
                          "bacteria", "happiness"]
        
        score_low = scorer.dat(low_creativity)
        score_high = scorer.dat(high_creativity)
        
        assert score_low < 60  # Low creativity should score < 60
        assert score_high > 80  # High creativity should score > 80
        assert score_high > score_low