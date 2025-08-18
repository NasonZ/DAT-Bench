# Testing Guide for Divergent Bench

## Overview

The divergent_bench test suite is organized into two main categories:
- **Unit Tests**: Fast, isolated tests with mocked dependencies
- **Integration Tests**: Real-world tests with actual services (APIs, Ollama, etc.)

```
tests/
├── conftest.py              # Shared pytest configuration and fixtures
├── unit/                    # Fast, isolated tests
│   ├── test_dat_scorer.py  # DAT scoring logic
│   ├── test_metrics.py     # DSI and LZiv metrics
│   └── test_model_detection.py # Model routing logic
└── integration/            # Tests with real dependencies
    ├── test_end_to_end.py  # Full workflow tests
    ├── test_structured_output.py # OpenAI structured output
    ├── test_api_endpoints.py # API parameter handling
    ├── test_modern_instructor.py # Instructor implementation
    └── test_ollama_models.py # Ollama model compatibility
```

## Running Tests

### Quick Start
```bash
# Run all unit tests (fast, no external dependencies)
pytest tests/unit/ -v

# Run specific unit test
pytest tests/unit/test_model_detection.py -v

# Run integration tests (requires API keys/services)
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=divergent_bench --cov-report=html

# Run specific test by name pattern
pytest tests/ -k "test_model_detection" -v
```

### Custom Options
```bash
# Test specific Ollama model
pytest tests/integration/test_ollama_models.py --model=mistral:7b -v

# Run including skipped tests
pytest tests/ --runxfail -v

# Verbose output with stdout
pytest tests/ -vvs

# Stop on first failure
pytest tests/ -x
```

## Unit Tests

### Purpose
Unit tests validate individual components in isolation using mocks and fixtures. They should:
- Run quickly (<1 second per test)
- Not require external services
- Use mocks for dependencies
- Test edge cases and error conditions

### Current Unit Tests

#### test_dat_scorer.py
Tests the Divergent Association Task scoring algorithm:
- Word vector loading and caching
- Semantic distance calculations
- Score computation and normalization
- Edge cases (missing words, invalid inputs)

#### test_metrics.py
Tests creativity metrics:
- DSI (Diversity of Semantic Integration) calculations
- LZiv (Lempel-Ziv) complexity measurements
- Statistical computations

#### test_model_detection.py
Tests LLM model routing and capability detection:
- Pattern matching for model names
- API endpoint routing decisions
- Parameter restrictions by model type
- Critical bug regression tests

### How to Extend Unit Tests

#### 1. Adding Tests to Existing Files
```python
# In test_model_detection.py
def test_new_model_series(self, mock_client):
    """Test detection of new model series."""
    assert mock_client._is_reasoning_model("claude-5-opus") == True
    assert mock_client._uses_completion_tokens("claude-5-opus") == True
```

#### 2. Creating New Unit Test Files
```python
# tests/unit/test_prompt_strategies.py
import pytest
from unittest.mock import Mock, patch
from divergent_bench.config.strategies import StrategyManager

class TestPromptStrategies:
    """Test prompt strategy selection and formatting."""
    
    @pytest.fixture
    def strategy_manager(self):
        """Create a strategy manager instance."""
        return StrategyManager()
    
    def test_strategy_selection(self, strategy_manager):
        """Test that correct strategy is selected."""
        strategy = strategy_manager.get_strategy("creative")
        assert "creative" in strategy.lower()
    
    @pytest.mark.parametrize("strategy_name,expected_keywords", [
        ("logical", ["step", "reason"]),
        ("creative", ["imagine", "unique"]),
        ("analytical", ["analyze", "consider"]),
    ])
    def test_strategy_keywords(self, strategy_manager, strategy_name, expected_keywords):
        """Test that strategies contain expected keywords."""
        strategy = strategy_manager.get_strategy(strategy_name)
        for keyword in expected_keywords:
            assert keyword in strategy.lower()
```

#### 3. Using Mocks Effectively
```python
# Mock external dependencies
@patch('divergent_bench.llm.providers.AsyncOpenAI')
def test_api_call_with_mock(self, mock_openai):
    """Test API calls without real API."""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="test"))]
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    # Test the functionality
    client = OpenAICompatibleClient(provider="openai", api_key="test")
    result = client.generate(messages=[{"role": "user", "content": "test"}])
    
    # Verify behavior
    assert result.content == "test"
    mock_openai.return_value.chat.completions.create.assert_called_once()
```

## Integration Tests

### Purpose
Integration tests validate real-world functionality with actual services. They should:
- Test complete workflows
- Use real APIs and services (with test accounts/data)
- Validate compatibility with external systems
- Test error recovery and retries

### Current Integration Tests

#### test_end_to_end.py
Full workflow testing:
- Complete DAT task execution
- Multiple model providers
- Result persistence
- Error handling

#### test_structured_output.py
OpenAI structured output validation:
- Pydantic model responses
- Function calling
- Response parsing
- O-series model compatibility

#### test_ollama_models.py
Ollama model compatibility:
- Model initialization
- Structured output support
- Fallback parsing
- Performance benchmarking

### How to Extend Integration Tests

#### 1. Adding New Model Providers
```python
# tests/integration/test_anthropic_models.py
import pytest
import os
from divergent_bench.llm import create_llm_client

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")
class TestAnthropicModels:
    """Test Anthropic Claude models."""
    
    @pytest.mark.asyncio
    async def test_claude_structured_output(self):
        """Test Claude with structured output."""
        client = create_llm_client(provider="anthropic", model="claude-3-opus")
        # ... test implementation
```

#### 2. Adding New Test Scenarios
```python
# Add to existing test file
@pytest.mark.asyncio
async def test_parallel_generation(self):
    """Test parallel generation with multiple models."""
    import asyncio
    
    models = ["gpt-4", "gpt-3.5-turbo"]
    tasks = []
    
    for model in models:
        client = create_llm_client(provider="openai", model=model)
        task = client.generate(messages=[{"role": "user", "content": "test"}])
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    assert len(results) == len(models)
```

#### 3. Adding Performance Benchmarks
```python
# tests/integration/test_performance.py
import time
import pytest
from statistics import mean, stdev

@pytest.mark.integration
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarking tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4"])
    async def test_response_time(self, model, benchmark_results):
        """Benchmark model response times."""
        times = []
        
        for _ in range(5):
            start = time.time()
            # ... make API call
            elapsed = time.time() - start
            times.append(elapsed)
        
        benchmark_results[model] = {
            "mean": mean(times),
            "stdev": stdev(times),
            "min": min(times),
            "max": max(times)
        }
        
        # Assert reasonable performance
        assert mean(times) < 10.0  # Less than 10 seconds average
```

## Best Practices

### 1. Test Organization
- Group related tests in classes
- Use descriptive test names that explain what's being tested
- Keep tests focused on single behaviors
- Use fixtures for common setup

### 2. Parameterized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
    pytest.param("test3", "result3", marks=pytest.mark.skip(reason="Not ready")),
])
def test_multiple_cases(input, expected):
    assert process(input) == expected
```

### 3. Fixtures
```python
# In conftest.py or test file
@pytest.fixture
def sample_dat_words():
    """Provide sample DAT words for testing."""
    return ["ocean", "bicycle", "philosophy", "sandwich", 
            "galaxy", "whisper", "concrete", "melody", 
            "electron", "justice"]

@pytest.fixture(scope="session")
def glove_embeddings():
    """Load GloVe embeddings once per test session."""
    return load_embeddings()
```

### 4. Markers
```python
# Mark slow tests
@pytest.mark.slow
def test_expensive_operation():
    pass

# Skip based on condition
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gpu_operation():
    pass

# Mark as expected to fail
@pytest.mark.xfail(reason="Known issue, fix in progress")
def test_broken_feature():
    pass
```

### 5. Test Data Management
```python
# Use fixtures for test data
@pytest.fixture
def test_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate 10 creative words."}
    ]

# Use tmp_path for temporary files
def test_file_operations(tmp_path):
    test_file = tmp_path / "test.json"
    test_file.write_text('{"test": "data"}')
    # ... test with temporary file
```

## Environment Setup

### Required Environment Variables
```bash
# .env file for tests
OPENAI_API_KEY=sk-test-key
ANTHROPIC_API_KEY=sk-ant-test-key
OLLAMA_BASE_URL=http://localhost:11434/v1
GLOVE_PATH=/path/to/glove.840B.300d.txt
```

### Test-Specific Configuration
```python
# tests/conftest.py
import os
import pytest

@pytest.fixture(autouse=True)
def test_environment():
    """Set up test environment."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    yield
    # Cleanup after tests
    os.environ.pop("TESTING", None)
```

## Continuous Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -e .[test]
      - run: pytest tests/unit/ --cov=divergent_bench

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -e .[test]
      - run: pytest tests/integration/ -v
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Debugging Tests

### Useful pytest options
```bash
# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf

# Run tests matching keyword
pytest -k "model_detection"

# Show slowest tests
pytest --durations=10
```

### Using debugger in tests
```python
def test_complex_logic():
    result = complex_function()
    
    # Drop into debugger
    import pdb; pdb.set_trace()
    
    # Or use pytest's debugger
    pytest.set_trace()
    
    assert result == expected
```

## Coverage Reports

```bash
# Generate coverage report
pytest tests/ --cov=divergent_bench --cov-report=html

# View report
open htmlcov/index.html

# Fail if coverage below threshold
pytest tests/ --cov=divergent_bench --cov-fail-under=80
```

## Common Patterns

### Testing Async Code
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected

# Or use asyncio fixture
def test_with_asyncio(event_loop):
    result = event_loop.run_until_complete(async_function())
    assert result == expected
```

### Testing Exceptions
```python
def test_raises_exception():
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_raises("bad input")
```

### Testing Logging
```python
def test_logging(caplog):
    with caplog.at_level(logging.INFO):
        function_that_logs()
    
    assert "Expected message" in caplog.text
```

## Contributing Tests

When contributing new tests:

1. **Follow existing patterns**: Look at similar tests for guidance
2. **Document complex tests**: Add docstrings explaining what's being tested
3. **Keep tests independent**: Each test should run in isolation
4. **Use meaningful assertions**: Include helpful error messages
5. **Clean up resources**: Use fixtures and context managers
6. **Consider performance**: Mock expensive operations in unit tests
7. **Test edge cases**: Don't just test the happy path

## Questions?

For questions about testing, please:
1. Check existing tests for examples
2. Review pytest documentation
3. Open an issue on GitHub with the `testing` label