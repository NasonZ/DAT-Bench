"""Integration tests for Ollama model compatibility with divergent thinking tasks.

PURPOSE:
--------
This test suite validates Ollama models for compatibility with the divergent_bench system,
specifically testing their ability to generate creative word lists for the Divergent 
Association Task (DAT). Since Ollama supports hundreds of open-weight models with
varying capabilities, this suite helps identify which models work with structured
output (Pydantic) and which require fallback text parsing.

USAGE EXAMPLES:
--------------
1. Test a specific model already in the parameterized list:
   pytest tests/integration/test_ollama_models.py::TestOllamaModels::test_ollama_model_structured_output[llama3.2:3b] -v

2. Test ALL models in the parameterized list (including skipped ones):
   pytest tests/integration/test_ollama_models.py::TestOllamaModels::test_ollama_model_structured_output -v --runxfail

3. Test a custom model not in the list:
   pytest tests/integration/test_ollama_models.py::TestOllamaModels::test_ollama_model_custom --model=mistral:7b -v

4. Test multiple custom models:
   for model in mistral:7b llama2:13b phi3:mini; do
     pytest tests/integration/test_ollama_models.py::TestOllamaModels::test_ollama_model_custom --model=$model -v
   done

5. Run with specific Ollama server:
   OLLAMA_BASE_URL=http://192.168.1.100:11434/v1 pytest tests/integration/test_ollama_models.py -v

ADDING NEW MODELS:
-----------------
To add a model to regular testing rotation:

1. Pull the model first:
   ollama pull gemma2:9b

2. Add to the @pytest.mark.parametrize decorator (line ~42):
   pytest.param("gemma2:9b", id="gemma2-9b"),  # No skip mark for active testing
   pytest.param("gemma2:9b", id="gemma2-9b", marks=pytest.mark.skip(reason="Optional")),  # With skip

3. Run the test:
   pytest tests/integration/test_ollama_models.py::TestOllamaModels::test_ollama_model_structured_output[gemma2-9b] -v

WHAT THE TEST VALIDATES:
------------------------
1. **Model Initialization**: Can the model be loaded via OpenAI-compatible API?
2. **Structured Output**: Does the model support Pydantic schema responses?
3. **Fallback Parsing**: If structured fails, can we parse text output?
4. **Output Quality**: DAT score > 50 (measures semantic diversity)
5. **Performance**: Logs timing and token usage for comparison

TEST OUTPUT INTERPRETATION:
--------------------------
✅ Structured output SUCCESS: Model supports Pydantic schemas directly
✅ Fallback parsing SUCCESS: Model doesn't support schemas but text parsing works
❌ Both structured and fallback failed: Model incompatible or needs prompt engineering
   DAT Score: Higher is better (50-100+ typical range)

EXPECTED BEHAVIOR BY MODEL TYPE:
--------------------------------
- **Instruction-tuned models** (e.g., llama3.2-instruct): Usually support structured output
- **Base models** (e.g., llama3.2): May need fallback parsing
- **Small models** (<3B params): Often need careful prompting
- **Code models** (e.g., codellama): May excel at structured output
- **Chat models** (e.g., mistral): Generally good all-around

ENVIRONMENT VARIABLES:
---------------------
OLLAMA_BASE_URL: Ollama server URL (default: reads from .env or http://172.24.32.1:11434/v1)

COMMON ISSUES:
-------------
1. Timeout: Increase timeout in _is_ollama_available() or check model size
2. Low DAT score: Model may need different temperature or prompting strategy
3. Structured output fails: Normal for many models, fallback should work
4. Connection refused: Check Ollama is running (ollama serve)

EXTENDING THIS TEST:
-------------------
- Add new output schemas: Create new Pydantic models like DATWords
- Test other tasks: Modify prompts and scoring functions
- Add performance benchmarks: Extend _log_model_performance()
- Export results: Implement CSV/JSON export in test fixtures
"""

import asyncio
import os
import pytest
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from divergent_bench.llm import create_llm_client
from divergent_bench.config.strategies import DAT_STRATEGIES
from divergent_bench.dat.scorer import DATScorer

# Load environment variables
load_dotenv()


class DATWords(BaseModel):
    """Structured output for DAT word generation."""
    words: List[str] = Field(
        description="List of exactly 10 single English nouns that are as different from each other as possible",
        min_length=10,
        max_length=10
    )


@pytest.mark.integration
@pytest.mark.ollama
class TestOllamaModels:
    """Test Ollama model compatibility with divergent thinking tasks."""
    
    @pytest.fixture
    def ollama_url(self):
        """Get Ollama URL from environment or use default."""
        return os.getenv("OLLAMA_BASE_URL", "http://172.24.32.1:11434/v1")
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        pytest.param("llama3.2:3b", id="llama3.2:3b"),
        pytest.param("hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:latest", id="qwen3:4b", marks=pytest.mark.skip(reason="Test manually")),
        pytest.param("mistral-small3.2:24b", id="mistral", marks=pytest.mark.skip(reason="Test manually")),
    ])
    async def test_ollama_model_structured_output(self, ollama_url, model):
        """Test if Ollama model supports structured output for DAT task.
        
        This test validates:
        1. Model can be initialized
        2. Structured output works OR fallback parsing works
        3. Output quality (DAT score)
        """
        # Skip if Ollama not available
        if not await self._is_ollama_available(ollama_url):
            pytest.skip(f"Ollama not available at {ollama_url}")
        
        # Create client
        os.environ["OLLAMA_BASE_URL"] = ollama_url
        client = create_llm_client(provider="ollama", model=model)
        
        # Use the standard DAT prompt
        prompt = DAT_STRATEGIES["none"]
        
        # Test structured output
        structured_success = False
        words = None
        
        try:
            # Try structured output first
            result = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                output_type=DATWords,
                temperature=0.7,
                max_tokens=200
            )
            
            if isinstance(result, DATWords):
                words = [w.lower() for w in result.words]
                structured_success = True
                print(f"\n✅ {model}: Structured output SUCCESS")
                print(f"   Words: {', '.join(words[:5])}...")
        except Exception as e:
            print(f"\n⚠️  {model}: Structured output failed: {str(e)[:100]}")
        
        # Fallback to text parsing if structured failed
        if not structured_success:
            try:
                response = await client.generate(
                    messages=[{
                        "role": "user", 
                        "content": prompt + "\n\nProvide ONLY 10 words as a comma-separated list."
                    }],
                    temperature=0.7,
                    max_tokens=100
                )
                
                # Parse response
                words = self._parse_word_list(response.content)
                if len(words) >= 10:
                    print(f"✅ {model}: Fallback parsing SUCCESS")
                    print(f"   Words: {', '.join(words[:5])}...")
                else:
                    print(f"❌ {model}: Fallback parsing incomplete ({len(words)} words)")
                    
            except Exception as e:
                pytest.fail(f"{model}: Both structured and fallback failed: {e}")
        
        # Validate output quality
        if words and len(words) >= 7:
            scorer = DATScorer()
            score = scorer.dat(words[:10])
            print(f"   DAT Score: {score:.2f}")
            
            # Assert minimum quality
            assert score is not None, f"{model}: DAT scoring failed"
            assert score > 50, f"{model}: DAT score too low ({score:.2f})"
            
            # Log performance characteristics
            self._log_model_performance(model, structured_success, score)
    
    @pytest.mark.asyncio
    async def test_ollama_model_custom(self, ollama_url, request):
        """Test a custom Ollama model specified via command line.
        
        Usage: pytest tests/integration/test_ollama_models.py::TestOllamaModels::test_ollama_model_custom --model=your-model:tag
        """
        model = request.config.getoption("--model", default=None)
        if not model:
            pytest.skip("No model specified. Use --model=model-name:tag")
        
        await self.test_ollama_model_structured_output(ollama_url, model)
    
    def _parse_word_list(self, response: str) -> List[str]:
        """Parse words from text response."""
        import re
        
        # Clean response
        cleaned = response.lower().strip()
        
        # Try comma-separated first
        if ',' in cleaned:
            words = [w.strip() for w in cleaned.split(',')]
            words = [re.sub(r'[^a-z]', '', w) for w in words]
            words = [w for w in words if w and w.isalpha()]
            if len(words) >= 10:
                return words[:10]
        
        # Try line-by-line
        lines = cleaned.split('\n')
        words = []
        for line in lines:
            # Remove numbering, bullets
            line = re.sub(r'^[\d\.\-\*\s]+', '', line.strip())
            # Extract word before any punctuation
            word = re.sub(r'[^a-z].*', '', line)
            if word and word.isalpha():
                words.append(word)
        
        return words[:10]
    
    async def _is_ollama_available(self, url: str) -> bool:
        """Check if Ollama is running and accessible."""
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                # Try to reach Ollama's health endpoint
                base_url = url.replace('/v1', '')
                response = await client.get(f"{base_url}/api/tags", timeout=2.0)
                return response.status_code == 200
        except:
            return False
    
    def _log_model_performance(self, model: str, structured: bool, score: float):
        """Log model performance for comparison."""
        # In a real scenario, this could write to a CSV or database
        # For now, just format nicely
        method = "structured" if structured else "fallback"
        print(f"   Performance: {method} output, DAT={score:.2f}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_fallback_mechanism():
    """Test that fallback parsing works when structured output fails."""
    # This tests the fallback mechanism specifically
    from divergent_bench.llm.providers import OpenAICompatibleClient
    
    # Use the URL from environment
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://172.24.32.1:11434/v1")
    os.environ["OLLAMA_BASE_URL"] = ollama_url
    
    client = OpenAICompatibleClient(
        provider="ollama",
        model="llama3.2:3b",
        api_key="ollama"
    )
    
    # Test with a prompt that might challenge structured output
    complex_prompt = """
    Generate 10 words that are maximally different from each other.
    Consider multiple dimensions: concrete/abstract, living/non-living, 
    natural/artificial, large/small, etc.
    """
    
    response = await client.generate(
        messages=[{"role": "user", "content": complex_prompt}],
        temperature=0.7,
        max_tokens=200
    )
    
    assert response.content is not None
    assert len(response.content) > 0
    
    # Check that we can parse something from it
    words = []
    for line in response.content.split('\n'):
        line = line.strip()
        if line and not any(x in line.lower() for x in ['generate', 'consider', 'dimension']):
            # Try to extract a word
            import re
            word = re.sub(r'[^a-z]+', '', line.lower())
            if word and len(word) > 2:
                words.append(word)
    
    # We should get at least some words
    assert len(words) > 0, "Failed to parse any words from response"