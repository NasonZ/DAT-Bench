"""
Lempel-Ziv complexity scorer for measuring text complexity.
Adapted from DAT_GPT/scripts/analyze_stories_dsi-lziv.py
"""

import logging
from typing import Union, List
import numpy as np

logger = logging.getLogger(__name__)


class LZivScorer:
    """Calculate Lempel-Ziv complexity for text."""
    
    def __init__(self, normalize: bool = True):
        """
        Initialize LZiv scorer.
        
        Args:
            normalize: Whether to normalize the complexity score.
        """
        self.normalize = normalize
        
        # Try to import antropy, fall back to simple implementation if not available
        try:
            from antropy import lziv_complexity
            self.lziv_complexity = lziv_complexity
            self.use_antropy = True
        except ImportError:
            logger.warning("antropy not installed, using simple LZ complexity implementation")
            self.use_antropy = False
    
    def calculate(self, text: Union[str, List[str]]) -> float:
        """
        Calculate the Lempel-Ziv complexity of text.
        
        Args:
            text: Input text (string or list of words).
            
        Returns:
            LZ complexity score (normalized if requested).
        """
        if isinstance(text, list):
            text = ' '.join(text)
        
        if self.use_antropy:
            return self.lziv_complexity(text, normalize=self.normalize)
        else:
            return self._simple_lziv(text)
    
    def _simple_lziv(self, text: str) -> float:
        """
        Simple implementation of Lempel-Ziv complexity.
        
        Based on the algorithm from:
        Kaspar, F. & Schuster, H. (1987). Easily calculable measure for the 
        complexity of spatiotemporal patterns. Physical Review A, 36(2), 842.
        
        Args:
            text: Input text string.
            
        Returns:
            Normalized LZ complexity.
        """
        # Convert text to binary sequence
        binary = ''.join(format(ord(c), '08b') for c in text)
        n = len(binary)
        
        if n == 0:
            return 0
        
        # Calculate LZ complexity
        i = 0
        k = 1
        l = 1
        k_max = 1
        c = 1
        
        while True:
            if binary[i + k - 1] == binary[l + k - 1]:
                k += 1
                if l + k > n:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                
                i += 1
                
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > n:
                        break
                    else:
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    k = 1
        
        if self.normalize:
            # Normalize by the upper bound: n / log2(n)
            if n > 1:
                return c / (n / np.log2(n))
            else:
                return 0
        else:
            return c
    
    def batch_calculate(self, texts: List[Union[str, List[str]]]) -> List[float]:
        """
        Calculate LZ complexity for multiple texts.
        
        Args:
            texts: List of texts (strings or word lists).
            
        Returns:
            List of LZ complexity scores.
        """
        return [self.calculate(text) for text in texts]