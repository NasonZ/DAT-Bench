"""
Experiment Runner - Bridges LLM module with DAT scorer.
Handles running experiments with different models and strategies.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ..llm import create_llm_client
from ..config.strategies import DAT_STRATEGIES, CREATIVE_PROMPTS, DEFAULT_TEMPERATURES
from ..dat.scorer import DATScorer


class DATWords(BaseModel):
    """Structured output for DAT word generation."""
    words: List[str] = Field(
        description="List of exactly 10 single English nouns that are as different from each other as possible",
        min_length=10,
        max_length=10
    )

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run divergent thinking experiments with various LLMs."""
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize experiment runner.
        
        Args:
            provider: LLM provider (openai, anthropic, gemini, etc.)
            model: Specific model to use (optional)
        """
        self.provider = provider
        self.client = create_llm_client(provider=provider, model=model)
        self.dat_scorer = None  # Lazy load when needed
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def _get_dat_scorer(self) -> DATScorer:
        """Lazy load DAT scorer."""
        if self.dat_scorer is None:
            self.dat_scorer = DATScorer()
        return self.dat_scorer
    
    async def run_dat_experiment(
        self,
        strategy: str = "none",
        temperature: Optional[float] = None,
        num_samples: int = 10,
        save_incrementally: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run DAT experiment with specified strategy.
        
        Args:
            strategy: DAT strategy to use (none, thesaurus, etymology, etc.)
            temperature: Generation temperature (uses default if None)
            num_samples: Number of samples to generate
            save_incrementally: Save results after each sample
            
        Returns:
            List of results with words, scores, and metadata
        """
        if strategy not in DAT_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(DAT_STRATEGIES.keys())}")
        
        if temperature is None:
            temperature = DEFAULT_TEMPERATURES.get(strategy, 0.7)
        
        prompt = DAT_STRATEGIES[strategy]
        results = []
        
        logger.info(f"Starting DAT experiment: {strategy=}, {temperature=}, {num_samples=}")
        
        for i in range(num_samples):
            try:
                import time
                start_time = time.time()
                words = None
                raw_response = None
                usage_info = None
                generation_method = None
                reasoning_summary = None
                
                # Try structured output first (if supported)
                try:
                    if self.client.supports_structured_output:
                        result = await self.client.generate(
                            messages=[{"role": "user", "content": prompt}],
                            output_type=DATWords,
                            temperature=temperature,
                            max_tokens=4096,  # Increased for GPT-5 models that use reasoning tokens
                            reasoning_effort="medium"  # For reasoning models
                            # reasoning_summary="auto"  # Disabled - requires org verification
                        )
                        if isinstance(result, DATWords):
                            words = [w.lower() for w in result.words]
                            raw_response = f"Structured output: {result.words}"
                            generation_method = "structured"
                            # Get usage from the client's last response
                            if hasattr(self.client, 'get_last_usage'):
                                usage_info = self.client.get_last_usage()
                            # Get reasoning summary if available
                            if hasattr(self.client, 'get_last_reasoning_summary'):
                                reasoning_summary = self.client.get_last_reasoning_summary()
                except Exception as e:
                    logger.debug(f"Structured output failed, falling back to text parsing: {e}")
                
                # Fallback to regular generation and parsing
                if words is None:
                    response = await self.client.generate(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=4096,  # Increased for GPT-5 models that use reasoning tokens
                        reasoning_effort="medium"  # For reasoning models
                        # reasoning_summary="auto"  # Disabled - requires org verification
                    )
                    raw_response = response.content
                    # Parse words from response
                    words = self._parse_word_list(response.content)
                    generation_method = "text_parsing"
                    # Extract usage info if available
                    if hasattr(response, 'usage') and response.usage:
                        usage_info = response.usage
                    # Extract reasoning summary if available
                    reasoning_summary = None
                    if hasattr(response, 'metadata') and response.metadata:
                        reasoning_summary = response.metadata.get('reasoning_summary')
                
                generation_time = time.time() - start_time
                
                # Calculate DAT score
                scorer = self._get_dat_scorer()
                score = scorer.dat(words) if len(words) >= 7 else None
                
                # Build comprehensive result object
                result = {
                    "iteration": i,
                    "timestamp": datetime.now().isoformat(),
                    "provider": self.provider,
                    "model": self.client.model if hasattr(self.client, 'model') else None,
                    "strategy": strategy,
                    "temperature": temperature,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "words": words,
                    "score": float(score) if score is not None else None,  # Convert numpy float to Python float
                    "num_valid_words": len([w for w in words if scorer.validate(w)]),
                    "generation_method": generation_method,
                    "generation_time": round(generation_time, 2),
                    "usage": usage_info  # Will include reasoning_tokens for GPT-5/o4 models
                }
                
                # Add reasoning summary if available
                if 'reasoning_summary' in locals() and reasoning_summary:
                    result["reasoning_summary"] = reasoning_summary
                
                results.append(result)
                if score:
                    logger.info(f"Sample {i+1}/{num_samples}: Score={score:.2f}")
                else:
                    logger.info(f"Sample {i+1}/{num_samples}: Score=N/A")
                
                # Save incrementally
                if save_incrementally:
                    self._save_results(results, strategy, temperature)
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in iteration {i}: {e}")
                continue
        
        return results
    
    def _parse_word_list(self, response: str) -> List[str]:
        """
        Parse word list from LLM response.
        
        Handles various formats:
        - Numbered lists (1. word, 2. word)
        - Comma-separated
        - Newline-separated
        - Bulleted lists
        - Markdown formatting (**word**)
        - Words with descriptions (word - description)
        """
        import re
        
        # Remove markdown bold/italic formatting
        cleaned = re.sub(r'\*+([^*]+)\*+', r'\1', response)
        
        # Remove list markers (numbers, bullets, dashes)
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^[-*•]\s*', '', cleaned, flags=re.MULTILINE)
        
        # Extract words before descriptions (handle "word - description" or "word:")
        lines = cleaned.split('\n')
        extracted_words = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract word before dash or colon
            if ' - ' in line:
                word = line.split(' - ')[0].strip()
            elif ':' in line:
                word = line.split(':')[0].strip()
            else:
                word = line.strip()
            
            # Clean up the word
            word = re.sub(r'[^\w\s-]', '', word)  # Remove punctuation except hyphens
            word = word.strip().lower()
            
            # Filter out common non-word lines
            if word and len(word) > 1 and not any(skip in word for skip in [
                "provide", "diverse", "words", "nouns", "here", "list", "following"
            ]):
                extracted_words.append(word)
        
        # If we didn't get enough words, try comma separation
        if len(extracted_words) < 5:
            if ',' in response:
                comma_words = [
                    re.sub(r'[^\w\s-]', '', w).strip().lower()
                    for w in response.split(',')
                ]
                comma_words = [
                    w for w in comma_words 
                    if w and len(w) > 1 and w.isalpha()
                ]
                if len(comma_words) >= len(extracted_words):
                    extracted_words = comma_words
        
        # Take first 10 valid words
        return extracted_words[:10]
    
    def _save_results(self, results: List[Dict], strategy: str, temperature: float):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.provider}_{strategy}_temp{temperature}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.debug(f"Saved results to {filepath}")
    
    async def run_batch_experiments(
        self,
        strategies: List[str],
        temperatures: List[float],
        samples_per_condition: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Run multiple experiments with different conditions.
        
        Args:
            strategies: List of strategies to test
            temperatures: List of temperatures to test
            samples_per_condition: Number of samples per strategy/temperature combination
            
        Returns:
            Dictionary mapping condition keys to results
        """
        all_results = {}
        
        for strategy in strategies:
            for temperature in temperatures:
                key = f"{strategy}_temp{temperature}"
                logger.info(f"Running batch: {key}")
                
                results = await self.run_dat_experiment(
                    strategy=strategy,
                    temperature=temperature,
                    num_samples=samples_per_condition
                )
                
                all_results[key] = results
                
                # Brief pause between conditions
                await asyncio.sleep(2)
        
        return all_results
    
    def visualize_results(self, results_path: Optional[Path] = None) -> None:
        """
        Generate visualizations from results.
        
        Args:
            results_path: Path to results (defaults to self.results_dir)
        """
        try:
            from ..visualization import load_results, prepare_data, ridge_plot, apply_plot_style
            import matplotlib.pyplot as plt
            
            if results_path is None:
                results_path = self.results_dir
            
            # Load and prepare data
            df = load_results(results_path)
            df = prepare_data(df)
            
            if df.empty:
                logger.warning("No data to visualize")
                return
            
            # Apply styling
            apply_plot_style()
            
            # Create visualizations
            viz_dir = Path("visualizations")
            viz_dir.mkdir(exist_ok=True)
            
            # Ridge plot
            fig = ridge_plot(df, title=f"DAT Scores - {self.provider}")
            if fig:
                output_path = viz_dir / f"{self.provider}_ridge_plot.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved visualization to {output_path}")
            
        except ImportError:
            logger.warning("Visualization module not available")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Returns statistics about the results.
        """
        import numpy as np
        
        scores = [r['score'] for r in results if r['score'] is not None]
        
        if not scores:
            return {"error": "No valid scores"}
        
        return {
            "num_samples": len(results),
            "num_valid": len(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "median_score": np.median(scores),
        }