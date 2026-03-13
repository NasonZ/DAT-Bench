"""
Simple data loader for divergent_bench results.
Flexible approach - minimal transformations, maximum compatibility.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_results(path: Union[str, Path], pattern: str = "*.json") -> pd.DataFrame:
    """
    Load JSON results from divergent_bench experiments.
    
    Args:
        path: Directory containing JSON results or single JSON file
        pattern: Glob pattern for JSON files (if path is directory)
    
    Returns:
        DataFrame with all results
    """
    path = Path(path)
    all_results = []
    
    # Handle directory vs single file
    if path.is_dir():
        json_files = sorted(path.glob(pattern))
        logger.info(f"Found {len(json_files)} result files")
    else:
        json_files = [path]
    
    # Load each file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Handle both single result and array of results
                if isinstance(data, dict):
                    data = [data]
                
                # Add source file for tracking
                for item in data:
                    item['source_file'] = json_file.name
                
                all_results.extend(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping malformed JSON file {json_file}: {e}")
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
    
    if not all_results:
        logger.warning("No results loaded")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Ensure core columns exist
    required_cols = ['score', 'words', 'model', 'strategy', 'temperature']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing column '{col}' in results")
            df[col] = None
    
    # Basic type conversions
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    if 'temperature' in df.columns:
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    
    # Create display name for models (provider + model if both exist)
    if 'provider' in df.columns and 'model' in df.columns:
        df['display_model'] = df.apply(
            lambda row: f"{row['model']}" if pd.notna(row['model']) 
            else f"{row['provider']}-unknown",
            axis=1
        )
    else:
        df['display_model'] = df.get('model', 'unknown')
    
    logger.info(f"Loaded {len(df)} results from {len(json_files)} files")
    return df


def prepare_data(df: pd.DataFrame, 
                 filter_outliers: bool = True,
                 n_std: float = 3,
                 sample_size: Optional[int] = 500,
                 min_samples: int = 3) -> pd.DataFrame:
    """
    Prepare data for visualization with optional filtering and sampling.
    
    Args:
        df: Raw results DataFrame
        filter_outliers: Whether to remove outliers
        n_std: Number of standard deviations for outlier detection
        sample_size: Max samples per model (None for no limit)
        min_samples: Minimum samples required per model
    
    Returns:
        Cleaned DataFrame ready for visualization
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Remove rows with missing scores
    initial_count = len(df)
    df = df[df['score'].notna()]
    if len(df) < initial_count:
        logger.info(f"Removed {initial_count - len(df)} rows with missing scores")
    
    # Group by model for processing
    model_col = 'display_model' if 'display_model' in df.columns else 'model'
    
    if filter_outliers and 'score' in df.columns:
        # Remove outliers per model
        def remove_outliers(group):
            if len(group) < min_samples:
                return group
            mean = group['score'].mean()
            std = group['score'].std()
            if std > 0:
                mask = np.abs(group['score'] - mean) <= n_std * std
                return group[mask]
            return group
        
        df = df.groupby(model_col).apply(remove_outliers).reset_index(drop=True)
        logger.info(f"After outlier removal: {len(df)} results")
    
    if sample_size and 'score' in df.columns:
        # Sample to consistent size per model
        def sample_group(group):
            n = min(len(group), sample_size)
            if n >= min_samples:
                return group.sample(n=n, random_state=42)
            return group if len(group) >= min_samples else pd.DataFrame()
        
        df = df.groupby(model_col).apply(sample_group).reset_index(drop=True)
        logger.info(f"After sampling: {len(df)} results")
    
    # Remove models with too few samples
    model_counts = df[model_col].value_counts()
    valid_models = model_counts[model_counts >= min_samples].index
    df = df[df[model_col].isin(valid_models)]
    
    if len(valid_models) < len(model_counts):
        logger.info(f"Kept {len(valid_models)} models with >= {min_samples} samples")
    
    return df


def get_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for each model.
    
    Args:
        df: Results DataFrame
    
    Returns:
        Summary DataFrame with mean, std, count per model
    """
    model_col = 'display_model' if 'display_model' in df.columns else 'model'
    
    summary = df.groupby(model_col)['score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('sem', lambda x: x.std() / np.sqrt(len(x))),
        ('count', 'count'),
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    # Sort by mean score
    summary = summary.sort_values('mean', ascending=False)
    
    return summary


def get_best_runs(df: pd.DataFrame,
                  model_col: str = None,
                  score_col: str = 'score',
                  top_n: int = 1) -> pd.DataFrame:
    """
    Get the best run(s) for each model based on score.
    
    Args:
        df: DataFrame with model results
        model_col: Column name for models (auto-detected if None)
        score_col: Column name for scores
        top_n: Number of top runs to return per model
    
    Returns:
        DataFrame with best runs for each model
    """
    # Auto-detect model column
    if model_col is None:
        if 'display_model' in df.columns:
            model_col = 'display_model'
        elif 'model' in df.columns:
            model_col = 'model'
        else:
            raise ValueError("No model column found")
    
    # Get best runs for each model
    best_runs = []
    for model in df[model_col].unique():
        model_df = df[df[model_col] == model]
        # Sort by score and take top_n
        top_runs = model_df.nlargest(top_n, score_col)
        best_runs.append(top_runs)
    
    if best_runs:
        return pd.concat(best_runs, ignore_index=True)
    else:
        return pd.DataFrame()


def aggregate_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by experimental conditions.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame grouped by model, strategy, temperature
    """
    model_col = 'display_model' if 'display_model' in df.columns else 'model'
    
    # Group by available conditions
    group_cols = [col for col in [model_col, 'strategy', 'temperature'] if col in df.columns]
    
    if not group_cols:
        logger.warning("No grouping columns available")
        return df
    
    agg_df = df.groupby(group_cols).agg({
        'score': ['mean', 'std', 'count']
    }).round(2)
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    
    return agg_df


def load_verifiers_results(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load results from verifiers' results.jsonl format into DAT-Bench DataFrame.

    Converts verifiers' RolloutOutput records into the same schema used by
    the standalone experiment runner, so all existing visualization functions
    (ridge_plot, statistical_heatmap, etc.) work without modification.

    Args:
        path: Path to a verifiers results.jsonl file

    Returns:
        DataFrame compatible with prepare_data / ridge_plot / etc.
    """
    import re

    path = Path(path)
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract completion text
            completion = obj.get("completion", [])
            content = ""
            if completion:
                last_msg = completion[-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")

            # Parse words from completion text
            words = _parse_words_from_text(content)

            # Extract metadata
            metrics = obj.get("metrics", {})
            info = obj.get("info", {})
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except (json.JSONDecodeError, TypeError):
                    info = {}

            rows.append(
                {
                    "model": obj.get("task", "unknown"),
                    "score": metrics.get("_raw_dat_score", metrics.get("raw_dat_score", None)),
                    "words": words,
                    "num_valid_words": int(
                        metrics.get("_num_valid_words", metrics.get("num_valid_words", 0))
                    ),
                    "strategy": info.get("strategy", "none"),
                    "temperature": 0.7,  # from env default; override from metadata if available
                    "reward": obj.get("reward"),
                    "example_id": obj.get("example_id"),
                    "source_file": path.name,
                }
            )

    if not rows:
        logger.warning(f"No results loaded from {path}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["display_model"] = df["model"]

    logger.info(f"Loaded {len(df)} verifiers results from {path.name}")
    return df


def _parse_words_from_text(text: str) -> list:
    """Extract word list from raw completion text (shared helper)."""
    import re

    if not text:
        return []

    cleaned = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    cleaned = re.sub(r"^\d+[.):\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[-*\u2022]\s*", "", cleaned, flags=re.MULTILINE)

    words = []
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
        skip = {"provide", "diverse", "words", "nouns", "here", "list", "following"}
        if word and len(word) > 1 and not any(s in word for s in skip):
            words.append(word)

    if len(words) < 5 and "," in text:
        comma_words = [re.sub(r"[^\w\s-]", "", w).strip().lower() for w in text.split(",")]
        comma_words = [w for w in comma_words if w and len(w) > 1 and w.isalpha()]
        if len(comma_words) >= len(words):
            words = comma_words

    return words[:10]


def add_temperature_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temperature category column for easier filtering.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with temperature_category column
    """
    if 'temperature' not in df.columns:
        return df
    
    df = df.copy()
    
    # Define temperature categories
    def categorize_temp(temp):
        if pd.isna(temp):
            return None
        elif temp <= 0.5:
            return 'Low'
        elif temp <= 0.8:
            return 'Medium'
        else:
            return 'High'
    
    df['temperature_category'] = df['temperature'].apply(categorize_temp)
    
    return df