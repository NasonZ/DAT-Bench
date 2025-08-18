"""
Divergent Semantic Integration (DSI) scorer.
Adapted from DAT_GPT/scripts/analyze_stories_dsi-lziv.py

References:
    Johnson et al., 2022 Extracting Creativity from Narratives using 
    Distributional Semantic Modeling https://osf.io/ath2s/
"""

import logging
from typing import List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DSIScorer:
    """
    Calculate Divergent Semantic Integration (DSI) scores.
    
    DSI measures semantic divergence in text by computing the average
    cosine distance between word/sentence embeddings.
    """
    
    def __init__(self, embedding_model: str = "glove", model_path: Optional[str] = None):
        """
        Initialize DSI scorer.
        
        Args:
            embedding_model: Type of embeddings ("glove", "bert", or "sentence-transformers")
            model_path: Path to embedding model (for GloVe)
        """
        self.embedding_model = embedding_model
        self.model = None
        self.tokenizer = None
        
        if embedding_model == "glove":
            self._load_glove(model_path)
        elif embedding_model == "bert":
            self._load_bert()
        elif embedding_model == "sentence-transformers":
            self._load_sentence_transformers()
        else:
            raise ValueError(f"Unknown embedding model: {embedding_model}")
    
    def _load_glove(self, model_path: Optional[str] = None):
        """Load GloVe embeddings."""
        if model_path is None:
            # Try to find GloVe from environment or standard locations
            import os
            model_path = os.getenv("GLOVE_PATH")
            if model_path is None or not Path(model_path).exists():
                for candidate in [
                    "/root/projects/divergent_thinking/divergent-association-task/glove.840B.300d.txt",
                    "data/embeddings/glove.840B.300d.txt",
                ]:
                    if Path(candidate).exists():
                        model_path = candidate
                        break
        
        if model_path is None or not Path(model_path).exists():
            logger.warning(f"GloVe model not found at {model_path}, DSI will use fallback")
            self.embeddings = {}
            return
        
        logger.info(f"Loading GloVe embeddings from {model_path}")
        self.embeddings = {}
        
        with open(model_path, 'r', encoding='utf8') as f:
            for line in f:
                tokens = line.split(' ')
                word = tokens[0]
                vector = np.asarray(tokens[1:], dtype='float32')
                self.embeddings[word] = vector
        
        logger.info(f"Loaded {len(self.embeddings)} word embeddings")
    
    def _load_bert(self):
        """Load BERT model for embeddings."""
        try:
            from transformers import BertModel, BertTokenizer
            import torch
            
            logger.info("Loading BERT model...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.model.eval()
            self.torch = torch
            self.cos = torch.nn.CosineSimilarity(dim=0)
        except ImportError:
            logger.error("transformers and torch required for BERT embeddings")
            raise
    
    def _load_sentence_transformers(self):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading sentence-transformers model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.error("sentence-transformers required for this embedding model")
            raise
    
    def get_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get embeddings for text.
        
        Args:
            text: Input text (string or list of words/sentences)
            
        Returns:
            Array of embeddings
        """
        if self.embedding_model == "glove":
            return self._get_glove_embeddings(text)
        elif self.embedding_model == "bert":
            return self._get_bert_embeddings(text)
        elif self.embedding_model == "sentence-transformers":
            return self._get_sentence_embeddings(text)
    
    def _get_glove_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get GloVe embeddings for words."""
        if isinstance(text, str):
            words = text.lower().split()
        else:
            words = [w.lower() for w in text]
        
        embeddings = []
        for word in words:
            if word in self.embeddings:
                embeddings.append(self.embeddings[word])
            else:
                # Use random vector for OOV words
                embeddings.append(np.random.randn(300) * 0.1)
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def _get_bert_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get BERT embeddings for text."""
        if isinstance(text, list):
            text = ' '.join(text)
        
        # Tokenize and get BERT embeddings
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state[0].numpy()
        
        return embeddings
    
    def _get_sentence_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get sentence embeddings."""
        if isinstance(text, str):
            sentences = [text]
        else:
            sentences = text
        
        embeddings = self.model.encode(sentences)
        return embeddings
    
    def calculate_dsi(self, text: Union[str, List[str]], return_details: bool = False) -> Union[float, Tuple[float, dict]]:
        """
        Calculate DSI score for text.
        
        DSI is the mean cosine distance between all pairs of embeddings.
        
        Args:
            text: Input text (string or list of words/sentences)
            return_details: Whether to return additional details
            
        Returns:
            DSI score (0-1, higher = more divergent)
            If return_details=True, returns (score, details_dict)
        """
        embeddings = self.get_embeddings(text)
        
        if len(embeddings) < 2:
            logger.warning("Not enough embeddings for DSI calculation")
            if return_details:
                return 0.0, {"num_embeddings": len(embeddings)}
            return 0.0
        
        # Calculate pairwise cosine distances
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings, metric='cosine')
        
        # Mean distance is the DSI score
        dsi_score = float(np.mean(distances))
        
        if return_details:
            details = {
                "num_embeddings": len(embeddings),
                "mean_distance": dsi_score,
                "std_distance": float(np.std(distances)),
                "min_distance": float(np.min(distances)),
                "max_distance": float(np.max(distances))
            }
            return dsi_score, details
        
        return dsi_score
    
    def calculate_successive_dsi(self, text: Union[str, List[str]]) -> float:
        """
        Calculate DSI using successive pairs only (like in original paper).
        
        This calculates the mean cosine distance between successive embeddings
        rather than all pairs.
        
        Args:
            text: Input text
            
        Returns:
            Successive DSI score
        """
        embeddings = self.get_embeddings(text)
        
        if len(embeddings) < 2:
            return 0.0
        
        # Calculate distances between successive pairs
        distances = []
        for i in range(len(embeddings) - 1):
            # Cosine distance between successive embeddings
            from scipy.spatial.distance import cosine
            dist = cosine(embeddings[i], embeddings[i + 1])
            distances.append(dist)
        
        return float(np.mean(distances))
    
    def batch_calculate(self, texts: List[Union[str, List[str]]]) -> List[float]:
        """
        Calculate DSI for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of DSI scores
        """
        return [self.calculate_dsi(text) for text in texts]