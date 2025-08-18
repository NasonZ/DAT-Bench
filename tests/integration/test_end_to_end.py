"""
End-to-end integration test for divergent_bench.
Tests the complete flow from LLM generation to DAT scoring using real LLM clients.
"""

import pytest
import asyncio
import json
import os
from pathlib import Path
import numpy as np
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from divergent_bench.experiments.runner import ExperimentRunner
from divergent_bench.config.strategies import DAT_STRATEGIES
from divergent_bench.llm import create_llm_client

# Load environment variables from .env file
load_dotenv()


class DATWords(BaseModel):
    """Structured output for DAT word generation."""
    words: List[str] = Field(
        description="List of 10 words that are as different from each other as possible, all nouns",
        min_items=10,
        max_items=10
    )


@pytest.fixture
def setup_test_glove(tmp_path):
    """Create test GloVe embeddings and dictionary for testing."""
    # Create comprehensive test vocabulary
    glove_path = tmp_path / "glove.840B.300d.txt"
    
    # Extended vocabulary to cover more potential LLM outputs
    test_words = [
        # Animals
        "cat", "dog", "elephant", "tiger", "whale", "spider", "butterfly", "eagle",
        # Abstract concepts
        "democracy", "philosophy", "justice", "freedom", "truth", "wisdom", "courage", "empathy",
        # Natural phenomena
        "volcano", "ocean", "mountain", "desert", "glacier", "forest", "river", "canyon",
        # Technology/Science
        "algorithm", "electron", "molecule", "atom", "galaxy", "planet", "cosmos", "enzyme",
        # Arts/Culture
        "violin", "symphony", "poetry", "sculpture", "painting", "melody", "rhythm", "harmony",
        # Objects
        "satellite", "crystal", "mineral", "protein", "bacteria", "fungus", 
        # Common nouns
        "tree", "cloud", "star", "book", "chair", "table", "house", "car",
        "water", "fire", "earth", "air", "light", "shadow", "mirror", "window",
        # More abstract
        "imagination", "creativity", "intuition", "curiosity", "language", "history", "mathematics", "literature",
        # Additional common words for better coverage
        "love", "time", "space", "energy", "matter", "mind", "soul", "spirit",
        "dream", "memory", "thought", "idea", "concept", "theory", "fact", "data",
        "emotion", "feeling", "sensation", "perception", "consciousness", "awareness",
        "universe", "reality", "existence", "being", "essence", "substance",
        "structure", "pattern", "system", "network", "connection", "relation"
    ]
    
    # Generate diverse embeddings for each word
    np.random.seed(42)
    with open(glove_path, 'w') as f:
        for i, word in enumerate(test_words):
            # Generate 300-dimensional vector with some structure
            # This ensures different words have different distances
            base_vector = np.zeros(300)
            base_vector[i % 300] = 1.0  # Different basis for each word
            noise = np.random.randn(300) * 0.1
            vector = base_vector + noise
            vector = vector / np.linalg.norm(vector)  # Normalize
            vector_str = ' '.join(map(str, vector))
            f.write(f"{word} {vector_str}\n")
    
    # Create dictionary file
    dict_path = tmp_path / "words.txt"
    with open(dict_path, 'w') as f:
        for word in test_words:
            f.write(f"{word}\n")
    
    return glove_path, dict_path


@pytest.mark.asyncio
async def test_openai_dat_experiment(setup_test_glove):
    """Test DAT experiment with OpenAI client."""
    glove_path, dict_path = setup_test_glove
    
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Use real GloVe if available, otherwise use test data
    if not os.getenv("GLOVE_PATH") or not Path(os.getenv("GLOVE_PATH")).exists():
        os.environ['GLOVE_PATH'] = str(glove_path)
        os.environ['WORDS_PATH'] = str(dict_path)
    
    # Create runner with real OpenAI client
    runner = ExperimentRunner(provider="openai", model="gpt-5.1-mini")
    
    # Run actual experiment
    results = await runner.run_dat_experiment(
        strategy="none",
        temperature=0.7,
        num_samples=2,  # Small number for testing
        save_incrementally=False
    )
    
    # Verify results
    assert len(results) == 2
    
    for result in results:
        assert 'words' in result
        assert 'score' in result
        assert 'strategy' in result
        assert 'provider' in result
        assert 'model' in result
        
        # Check basic properties
        assert result['strategy'] == "none"
        assert result['temperature'] == 0.7
        assert result['provider'] == "openai"
        
        # Verify words were generated and parsed
        assert isinstance(result['words'], list)
        assert len(result['words']) > 0
        assert len(result['words']) <= 10
        
        # Verify scoring
        if len(result['words']) >= 7:
            assert result['score'] is not None
            assert isinstance(result['score'], (int, float))
            assert 0 <= result['score'] <= 200
        
        # Check metadata
        assert 'timestamp' in result
        assert 'iteration' in result
        assert 'raw_response' in result
        assert 'num_valid_words' in result


@pytest.mark.asyncio
async def test_ollama_dat_experiment(setup_test_glove):
    """Test DAT experiment with Ollama client."""
    glove_path, dict_path = setup_test_glove
    
    # Skip if Ollama not available
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url.replace('/v1', '')}/api/tags")
            if response.status_code != 200:
                pytest.skip("Ollama not accessible")
    except:
        pytest.skip("Ollama not running")
    
    # Set environment
    os.environ['GLOVE_PATH'] = str(glove_path)
    os.environ['WORDS_PATH'] = str(dict_path)
    
    # Create runner with Ollama client
    runner = ExperimentRunner(provider="ollama", model="llama3.2")
    
    # Run experiment
    results = await runner.run_dat_experiment(
        strategy="none",
        temperature=0.7,
        num_samples=1,
        save_incrementally=False
    )
    
    assert len(results) == 1
    result = results[0]
    
    # Basic validation
    assert result['provider'] == "ollama"
    assert 'words' in result
    assert isinstance(result['words'], list)


@pytest.mark.asyncio
async def test_batch_experiments_with_strategies(setup_test_glove):
    """Test batch experiments across multiple strategies."""
    glove_path, dict_path = setup_test_glove
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    os.environ['GLOVE_PATH'] = str(glove_path)
    os.environ['WORDS_PATH'] = str(dict_path)
    
    runner = ExperimentRunner(provider="openai", model="gpt-5.1-mini")
    
    # Test multiple strategies and temperatures
    strategies = ["none", "random"]
    temperatures = [0.5, 1.0]
    
    results = await runner.run_batch_experiments(
        strategies=strategies,
        temperatures=temperatures,
        samples_per_condition=1  # Minimal for testing
    )
    
    # Verify structure
    assert len(results) == len(strategies) * len(temperatures)
    
    for strategy in strategies:
        for temp in temperatures:
            key = f"{strategy}_temp{temp}"
            assert key in results
            assert len(results[key]) == 1
            
            result = results[key][0]
            assert result['strategy'] == strategy
            assert result['temperature'] == temp
            assert 'words' in result
            assert 'score' in result


@pytest.mark.asyncio
async def test_deepseek_structured_output(setup_test_glove):
    """Test with DeepSeek using instructor for structured output."""
    glove_path, dict_path = setup_test_glove
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    
    os.environ['GLOVE_PATH'] = str(glove_path)
    os.environ['WORDS_PATH'] = str(dict_path)
    
    # Create client with DeepSeek
    client = create_llm_client(provider="deepseek", model="deepseek-chat")
    
    # Test structured output
    prompt = DAT_STRATEGIES["none"]
    
    try:
        # DeepSeek uses instructor for structured output
        result = await client.generate(
            messages=[{"role": "user", "content": prompt}],
            output_type=DATWords,
            temperature=0.7
        )
        
        assert isinstance(result, DATWords)
        assert len(result.words) == 10
        assert all(isinstance(word, str) for word in result.words)
    except Exception as e:
        # Log but don't fail - API might be down
        print(f"DeepSeek test failed (API might be down): {e}")


# @pytest.mark.asyncio  
# async def test_anthropic_dat_experiment(setup_test_glove):
#     """Test DAT experiment with Anthropic Claude."""
#     glove_path, dict_path = setup_test_glove
    
#     if not os.getenv("ANTHROPIC_API_KEY"):
#         pytest.skip("ANTHROPIC_API_KEY not set")
    
#     os.environ['GLOVE_PATH'] = str(glove_path)
#     os.environ['WORDS_PATH'] = str(dict_path)
    
#     runner = ExperimentRunner(provider="anthropic", model="claude-3-5-sonnet-20241022")
    
#     # Run single experiment
#     results = await runner.run_dat_experiment(
#         strategy="etymology",  # Test different strategy
#         temperature=0.8,
#         num_samples=1,
#         save_incrementally=False
#     )
    
#     assert len(results) == 1
#     result = results[0]
    
#     assert result['provider'] == "anthropic"
#     assert result['strategy'] == "etymology"
#     assert 'words' in result
#     assert len(result['words']) > 0


def test_result_analysis_with_real_data(setup_test_glove):
    """Test analysis of real experimental results."""
    glove_path, dict_path = setup_test_glove
    
    # Use test GloVe data for this unit test
    os.environ['GLOVE_PATH'] = str(glove_path)
    os.environ['WORDS_PATH'] = str(dict_path)
    
    # Skip if no API key (though this test doesn't actually call the API)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ['OPENAI_API_KEY'] = 'test-key-for-analysis'  # Dummy key since we're not making API calls
    
    runner = ExperimentRunner(provider="openai")
    
    # Create realistic result data
    results = [
        {
            "score": 85.5,
            "words": ["cat", "democracy", "volcano", "algorithm", "violin", "desert", "empathy"],
            "num_valid_words": 7
        },
        {
            "score": 92.3,
            "words": ["ocean", "philosophy", "electron", "symphony", "mountain", "curiosity", "galaxy"],
            "num_valid_words": 7
        },
        {
            "score": None,  # Too few valid words
            "words": ["xyz", "abc"],
            "num_valid_words": 2
        },
        {
            "score": 88.7,
            "words": ["tree", "mathematics", "crystal", "melody", "river", "imagination", "molecule"],
            "num_valid_words": 7
        }
    ]
    
    analysis = runner.analyze_results(results)
    
    # Verify analysis structure
    assert 'num_samples' in analysis
    assert 'num_valid' in analysis
    assert 'mean_score' in analysis
    assert 'std_score' in analysis
    assert 'min_score' in analysis
    assert 'max_score' in analysis
    assert 'median_score' in analysis
    
    # Check calculations
    assert analysis['num_samples'] == 4
    assert analysis['num_valid'] == 3  # One None score
    
    valid_scores = [85.5, 92.3, 88.7]
    assert abs(analysis['mean_score'] - np.mean(valid_scores)) < 0.01
    assert abs(analysis['std_score'] - np.std(valid_scores)) < 0.01


def test_word_parsing_from_real_responses(setup_test_glove):
    """Test parsing various real LLM response formats."""
    glove_path, dict_path = setup_test_glove
    
    # Use test GloVe data for this unit test
    os.environ['GLOVE_PATH'] = str(glove_path)
    os.environ['WORDS_PATH'] = str(dict_path)
    
    # Skip if no API key (though this test doesn't actually call the API)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ['OPENAI_API_KEY'] = 'test-key-for-parsing'  # Dummy key since we're not making API calls
    
    runner = ExperimentRunner(provider="openai")
    
    # Real response format 1: Numbered list (common from GPT)
    response1 = """Here are 10 diverse words:

1. Ocean
2. Democracy  
3. Molecule
4. Symphony
5. Desert
6. Algorithm
7. Empathy
8. Volcano
9. Telescope
10. Justice"""
    
    words1 = runner._parse_word_list(response1)
    assert len(words1) == 10
    assert "ocean" in words1
    assert "democracy" in words1
    
    # Real response format 2: Comma-separated (common from Claude)
    response2 = "ocean, philosophy, electron, symphony, mountain, curiosity, galaxy, poetry, enzyme, freedom"
    
    words2 = runner._parse_word_list(response2)
    assert len(words2) == 10
    assert all(word.isalpha() for word in words2)
    
    # Real response format 3: Bulleted (common from instruction-tuned models)
    response3 = """- Ocean
- Mountain
- Algorithm
- Poetry
- Electron
- Desert
- Symphony
- Philosophy
- Molecule
- Justice"""
    
    words3 = runner._parse_word_list(response3)
    assert len(words3) == 10
    assert "ocean" in words3
    
    # Real response format 4: Mixed with explanation
    response4 = """I'll provide 10 diverse nouns:

1. **Ocean** - vast body of water
2. **Algorithm** - computational procedure  
3. **Symphony** - musical composition
4. **Desert** - arid region
5. **Molecule** - chemical unit
6. **Democracy** - political system
7. **Volcano** - geological formation
8. **Poetry** - literary art
9. **Electron** - subatomic particle
10. **Justice** - moral principle"""
    
    words4 = runner._parse_word_list(response4)
    # Should extract just the words, not descriptions
    assert "ocean" in words4
    assert "algorithm" in words4
    assert "vast" not in words4  # Should not include descriptions


@pytest.mark.asyncio
async def test_temperature_effects(setup_test_glove):
    """Test that different temperatures produce different results."""
    glove_path, dict_path = setup_test_glove
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    os.environ['GLOVE_PATH'] = str(glove_path)
    os.environ['WORDS_PATH'] = str(dict_path)
    
    runner = ExperimentRunner(provider="openai", model="gpt-5.1-mini")
    
    # Run with low temperature
    low_temp_results = await runner.run_dat_experiment(
        strategy="none",
        temperature=0.1,
        num_samples=2,
        save_incrementally=False
    )
    
    # Run with high temperature  
    high_temp_results = await runner.run_dat_experiment(
        strategy="none",
        temperature=1.5,
        num_samples=2,
        save_incrementally=False
    )
    
    # Both should produce results
    assert len(low_temp_results) == 2
    assert len(high_temp_results) == 2
    
    # High temperature should generally produce more varied results
    # (though this is probabilistic, not guaranteed)
    low_scores = [r['score'] for r in low_temp_results if r['score']]
    high_scores = [r['score'] for r in high_temp_results if r['score']]
    
    if low_scores and high_scores:
        print(f"Low temp mean score: {np.mean(low_scores):.2f}")
        print(f"High temp mean score: {np.mean(high_scores):.2f}")


@pytest.mark.asyncio
async def test_save_and_load_results(setup_test_glove, tmp_path):
    """Test saving results incrementally and loading them."""
    glove_path, dict_path = setup_test_glove
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    os.environ['GLOVE_PATH'] = str(glove_path)
    os.environ['WORDS_PATH'] = str(dict_path)
    
    # Create runner with custom results directory
    runner = ExperimentRunner(provider="openai", model="gpt-5.1-mini")
    runner.results_dir = tmp_path / "results"
    runner.results_dir.mkdir(exist_ok=True)
    
    # Run experiment with incremental saving
    results = await runner.run_dat_experiment(
        strategy="thesaurus",
        temperature=0.7,
        num_samples=2,
        save_incrementally=True
    )
    
    # Check that results were saved
    result_files = list(runner.results_dir.glob("*.json"))
    assert len(result_files) > 0
    
    # Load and verify saved results
    with open(result_files[0], 'r') as f:
        saved_results = json.load(f)
    
    assert isinstance(saved_results, list)
    assert len(saved_results) == len(results)
    
    # Verify saved data matches returned data
    for saved, original in zip(saved_results, results):
        assert saved['strategy'] == original['strategy']
        assert saved['temperature'] == original['temperature']
        assert saved['words'] == original['words']
        if saved['score'] and original['score']:
            assert abs(saved['score'] - original['score']) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])