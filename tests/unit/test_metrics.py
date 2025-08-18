"""
Unit tests for divergent thinking metrics (DSI and LZiv).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from divergent_bench.metrics import DSIScorer, LZivScorer


class TestLZivScorer:
    """Test Lempel-Ziv complexity scorer."""
    
    def test_initialization(self):
        """Test LZiv scorer initialization."""
        scorer = LZivScorer(normalize=True)
        assert scorer.normalize is True
        
        scorer = LZivScorer(normalize=False)
        assert scorer.normalize is False
    
    def test_calculate_string(self):
        """Test LZ complexity calculation for string."""
        scorer = LZivScorer(normalize=True)
        
        # Repetitive text should have lower complexity
        repetitive = "the cat the cat the cat the cat"
        score_repetitive = scorer.calculate(repetitive)
        
        # Diverse text should have higher complexity
        diverse = "the quick brown fox jumps over lazy dog"
        score_diverse = scorer.calculate(diverse)
        
        # Basic check that scores are computed
        assert isinstance(score_repetitive, float)
        assert isinstance(score_diverse, float)
        assert score_repetitive >= 0
        assert score_diverse >= 0
    
    def test_calculate_word_list(self):
        """Test LZ complexity calculation for word list."""
        scorer = LZivScorer(normalize=True)
        
        words = ["cat", "dog", "elephant", "mouse"]
        score = scorer.calculate(words)
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_simple_lziv_implementation(self):
        """Test the simple LZ implementation."""
        scorer = LZivScorer(normalize=True)
        scorer.use_antropy = False  # Force simple implementation
        
        text = "abcabcabc"
        score = scorer.calculate(text)
        
        assert isinstance(score, float)
        assert score >= 0
        
        # Empty text
        assert scorer.calculate("") == 0
    
    def test_batch_calculate(self):
        """Test batch calculation."""
        scorer = LZivScorer(normalize=True)
        
        texts = [
            "the cat sat on mat",
            "quick brown fox jumps",
            ["list", "of", "words"]
        ]
        
        scores = scorer.batch_calculate(texts)
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(s >= 0 for s in scores)


class TestDSIScorer:
    """Test Divergent Semantic Integration scorer."""
    
    @pytest.fixture
    def mock_glove_file(self, tmp_path):
        """Create a mock GloVe file."""
        glove_path = tmp_path / "test_glove.txt"
        
        # Create simple embeddings
        with open(glove_path, 'w') as f:
            words = ["cat", "dog", "tree", "ocean", "algorithm", "democracy"]
            for i, word in enumerate(words):
                # Create orthogonal vectors for maximum divergence
                vector = np.zeros(300)
                vector[i * 50] = 1.0  # Spread out in embedding space
                vector_str = ' '.join(map(str, vector))
                f.write(f"{word} {vector_str}\n")
        
        return glove_path
    
    def test_initialization_glove(self, mock_glove_file):
        """Test DSI scorer initialization with GloVe."""
        scorer = DSIScorer(embedding_model="glove", model_path=str(mock_glove_file))
        
        assert scorer.embedding_model == "glove"
        assert len(scorer.embeddings) == 6  # Number of words in mock file
    
    def test_initialization_invalid_model(self):
        """Test initialization with invalid model."""
        with pytest.raises(ValueError, match="Unknown embedding model"):
            DSIScorer(embedding_model="invalid")
    
    def test_get_glove_embeddings(self, mock_glove_file):
        """Test getting GloVe embeddings."""
        scorer = DSIScorer(embedding_model="glove", model_path=str(mock_glove_file))
        
        # Test with string
        embeddings = scorer.get_embeddings("cat dog tree")
        assert embeddings.shape == (3, 300)
        
        # Test with list
        embeddings = scorer.get_embeddings(["cat", "dog"])
        assert embeddings.shape == (2, 300)
        
        # Test with OOV words
        embeddings = scorer.get_embeddings(["cat", "unknown"])
        assert embeddings.shape == (2, 300)
    
    def test_calculate_dsi(self, mock_glove_file):
        """Test DSI calculation."""
        scorer = DSIScorer(embedding_model="glove", model_path=str(mock_glove_file))
        
        # Test with divergent words (should have high DSI)
        divergent_words = ["cat", "ocean", "algorithm"]
        dsi_divergent = scorer.calculate_dsi(divergent_words)
        
        assert isinstance(dsi_divergent, float)
        assert 0 <= dsi_divergent <= 2  # Cosine distance range
        
        # Test with single word (should return 0)
        dsi_single = scorer.calculate_dsi(["cat"])
        assert dsi_single == 0.0
        
        # Test with details
        dsi, details = scorer.calculate_dsi(divergent_words, return_details=True)
        assert isinstance(details, dict)
        assert "num_embeddings" in details
        assert "mean_distance" in details
        assert "std_distance" in details
        assert details["num_embeddings"] == 3
    
    def test_calculate_successive_dsi(self, mock_glove_file):
        """Test successive DSI calculation."""
        scorer = DSIScorer(embedding_model="glove", model_path=str(mock_glove_file))
        
        words = ["cat", "dog", "tree", "ocean"]
        dsi = scorer.calculate_successive_dsi(words)
        
        assert isinstance(dsi, float)
        assert 0 <= dsi <= 2
        
        # Single word should return 0
        assert scorer.calculate_successive_dsi(["cat"]) == 0.0
    
    def test_batch_calculate(self, mock_glove_file):
        """Test batch DSI calculation."""
        scorer = DSIScorer(embedding_model="glove", model_path=str(mock_glove_file))
        
        texts = [
            ["cat", "dog", "tree"],
            "ocean algorithm democracy",
            ["cat", "cat", "cat"]  # Repetitive
        ]
        
        scores = scorer.batch_calculate(texts)
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        
        # Repetitive text should have lower DSI
        assert scores[2] < scores[0]
    
    @patch('transformers.BertModel')
    @patch('transformers.BertTokenizer')
    def test_bert_embeddings(self, mock_tokenizer, mock_model):
        """Test BERT embedding functionality."""
        # Mock BERT components
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the forward pass
        mock_output = Mock()
        mock_output.last_hidden_state = Mock()
        mock_output.last_hidden_state.__getitem__.return_value.numpy.return_value = np.random.randn(10, 768)
        mock_model_instance.return_value = mock_output
        
        try:
            scorer = DSIScorer(embedding_model="bert")
            assert scorer.embedding_model == "bert"
        except ImportError:
            pytest.skip("transformers not installed")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_sentence_transformer_embeddings(self, mock_st):
        """Test sentence-transformers functionality."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(3, 384)
        mock_st.return_value = mock_model
        
        try:
            scorer = DSIScorer(embedding_model="sentence-transformers")
            assert scorer.embedding_model == "sentence-transformers"
            
            embeddings = scorer.get_embeddings(["sentence 1", "sentence 2", "sentence 3"])
            assert embeddings.shape == (3, 384)
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestMetricsIntegration:
    """Test integration of metrics with DAT scores."""
    
    @pytest.fixture
    def mock_glove_file(self, tmp_path):
        """Create a mock GloVe file with DAT words."""
        glove_path = tmp_path / "test_glove.txt"
        
        # Create embeddings for typical DAT words
        dat_words = [
            "cat", "democracy", "volcano", "algorithm", "violin",
            "desert", "empathy", "satellite", "fungus", "justice",
            "ocean", "philosophy", "electron", "symphony", "mountain"
        ]
        
        with open(glove_path, 'w') as f:
            for i, word in enumerate(dat_words):
                # Create diverse embeddings
                vector = np.random.randn(300)
                vector = vector / np.linalg.norm(vector)  # Normalize
                vector_str = ' '.join(map(str, vector))
                f.write(f"{word} {vector_str}\n")
        
        return glove_path
    
    def test_metrics_for_dat_words(self, mock_glove_file):
        """Test metrics calculation for typical DAT word lists."""
        # Initialize scorers
        dsi_scorer = DSIScorer(embedding_model="glove", model_path=str(mock_glove_file))
        lziv_scorer = LZivScorer(normalize=True)
        
        # Divergent word list (high creativity)
        divergent_words = ["cat", "democracy", "volcano", "algorithm", "violin"]
        
        # Similar word list (low creativity)
        similar_words = ["cat", "cat", "cat", "cat", "cat"]
        
        # Calculate DSI
        dsi_divergent = dsi_scorer.calculate_dsi(divergent_words)
        dsi_similar = dsi_scorer.calculate_dsi(similar_words)
        
        # Calculate LZiv
        lziv_divergent = lziv_scorer.calculate(divergent_words)
        lziv_similar = lziv_scorer.calculate(similar_words)
        
        # Divergent words should have higher DSI
        assert dsi_divergent > dsi_similar or dsi_divergent > 0
        
        # Both should be valid scores
        assert 0 <= dsi_divergent <= 2
        assert 0 <= dsi_similar <= 2
        assert lziv_divergent >= 0
        assert lziv_similar >= 0