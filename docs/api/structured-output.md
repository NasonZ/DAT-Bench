# Structured Output Implementation Guide

## Overview

This document explains how structured output works in the Divergent Association Task LLM module. The system supports extracting structured, type-safe data from LLMs using Pydantic models through two distinct approaches: native OpenAI parsing and the Instructor library.

## Architecture

### Dual-Method Approach

The system intelligently routes structured output requests through two methods:

1. **Native Parse Method** (`_native_parse_generate`)
   - Used for: OpenAI and Ollama
   - Leverages OpenAI's native `beta.chat.completions.parse()` API
   - Direct integration without external dependencies
   - Best performance for supported providers

2. **Instructor Method** (`_instructor_generate`)
   - Used for: Anthropic, DeepSeek, Gemini, and custom endpoints
   - Leverages the Instructor library for universal structured output
   - Two sub-approaches:
     - `instructor.from_provider()`: Modern approach for standard providers
     - `instructor.from_openai()`: Fallback for custom/local endpoints

### Routing Logic

```python
# In providers.py generate() method:
if output_type and self._supports_native_parse(current_model):
    # Route 1: Native parse for OpenAI/Ollama
    return await self._native_parse_generate(params, output_type)
elif output_type:
    # Route 2: Instructor for all other providers
    return await self._instructor_generate(params, output_type)
else:
    # Route 3: Regular completion (no structured output)
    return await self._regular_generate(params)
```

## Provider-Specific Behavior

### OpenAI
- **Method**: Native `parse()`
- **Async**: Yes
- **Example models**: `gpt-4o-mini`, `gpt-4.1-mini`, `o4-mini`
- **Special handling**: Reasoning models (o1, o4) have parameter restrictions

### Ollama
- **Method**: Native `parse()`
- **Async**: Yes
- **Example models**: `llama3.2`, `mistral`, `codellama`
- **Configuration**: Requires `OLLAMA_BASE_URL` or defaults to `http://localhost:11434/v1`

### DeepSeek
- **Method**: Instructor with `from_provider()`
- **Async**: No (sync client from `from_provider`)
- **Example models**: `deepseek-chat`
- **API Key**: `DEEPSEEK_API_KEY`

### Anthropic
- **Method**: Instructor with `from_provider()`
- **Async**: No (sync client from `from_provider`)
- **Example models**: `claude-3-5-sonnet-20241022`
- **API Key**: `ANTHROPIC_API_KEY`
- **Note**: Requires `anthropic` package for `from_provider` to work

### Gemini
- **Method**: Instructor with `from_provider()`
- **Async**: No (sync client from `from_provider`)
- **Example models**: `gemini-pro`
- **API Key**: `GEMINI_API_KEY`

### OpenRouter/Custom Endpoints
- **Method**: Instructor with `from_openai()`
- **Async**: Yes (preserves async from AsyncOpenAI)
- **Example models**: `openai/gpt-4o-mini`, custom model names
- **Configuration**: Custom `base_url` required

## Implementation Details

### Native Parse Implementation

```python
async def _native_parse_generate(self, params: Dict, output_type: Type[BaseModel]) -> Any:
    """Generate using native parse() API."""
    parse_params = dict(params)
    parse_params.pop('stream', None)  # parse() doesn't support streaming
    
    # Handle tool_choice: only include if tools are also present
    if 'tool_choice' in parse_params and 'tools' not in parse_params:
        parse_params.pop('tool_choice', None)
    
    response = await self._client.beta.chat.completions.parse(
        **parse_params,
        response_format=output_type
    )
    
    if response.choices and response.choices[0].message.parsed is not None:
        return response.choices[0].message.parsed
    else:
        raise ValueError("Structured output parsing failed - no parsed content")
```

### Instructor Implementation

```python
async def _instructor_generate(self, params: Dict, output_type: Type[BaseModel]) -> Any:
    """Generate using Instructor for structured output."""
    model_name = params.get("model", self.model)
    
    if self.provider in ["anthropic", "deepseek", "gemini"] and not self._base_url:
        # Modern approach for standard providers
        provider_string = f"{self.provider}/{model_name}"
        instructor_client = instructor.from_provider(
            provider_string,
            api_key=self._client.api_key
        )
        # from_provider returns a sync client
        result = instructor_client.chat.completions.create(
            **params,
            response_model=output_type,
            max_retries=3  # Automatic validation retries
        )
    else:
        # Fall back to from_openai for custom endpoints
        instructor_client = instructor.from_openai(self._client)
        # from_openai with AsyncOpenAI returns an async client
        result = await instructor_client.chat.completions.create(
            **params,
            response_model=output_type,
            max_retries=3
        )
    
    return result  # Already parsed and validated!
```

## Usage Examples

### Basic Structured Output

```python
from pydantic import BaseModel, Field
from llm import create_llm_client

class UserInfo(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years")
    email: str = Field(description="User's email address")

# Create client (will use appropriate method based on provider)
client = create_llm_client(provider="openai")  # Uses native parse

# Generate structured output
result = await client.generate(
    messages=[
        {"role": "user", "content": "Extract user info from: John Doe, 30 years old, john@example.com"}
    ],
    output_type=UserInfo
)

print(result.name)   # "John Doe"
print(result.age)    # 30
print(result.email)  # "john@example.com"
```

### Complex Nested Structures

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]
    primary_language: str = Field(default="English")

# Works with both native and instructor methods
client = create_llm_client(provider="deepseek")  # Uses instructor

result = await client.generate(
    messages=[{"role": "user", "content": "Create a person with multiple addresses"}],
    output_type=Person
)
```

### Provider-Specific Configuration

```python
# OpenAI with native parse
openai_client = create_llm_client(
    provider="openai",
    model="gpt-4o-mini"
)

# DeepSeek with instructor
deepseek_client = create_llm_client(
    provider="deepseek",
    model="deepseek-chat"
)

# Custom endpoint with instructor fallback
custom_client = create_llm_client(
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-3-opus"
)
```

## Adding New Providers

To add a new provider:

1. **Add to PROVIDER_CONFIGS** in `providers.py`:
```python
PROVIDER_CONFIGS = {
    "new_provider": {
        "base_url": "https://api.newprovider.com/v1",
        "supports_tools": True,
    },
    # ...
}
```

2. **Determine routing method**:
   - If provider has native structured output support → Add to `_supports_native_parse()`
   - If standard provider with known endpoint → Will use `instructor.from_provider()`
   - If custom endpoint → Will use `instructor.from_openai()`

3. **Add default model** in `_get_default_model()`:
```python
defaults = {
    "new_provider": "model-name",
    # ...
}
```

4. **Add API key handling** in `_create_client()`:
```python
env_vars = {
    "new_provider": "NEW_PROVIDER_API_KEY",
    # ...
}
```

## Environment Configuration

### Required Environment Variables

```bash
# .env file
LLM_PROVIDER=openai  # or anthropic, deepseek, ollama, etc.
LLM_MODEL=gpt-4o-mini

# Provider-specific API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Local/Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Python Dependencies

```toml
# pyproject.toml
dependencies = [
    "openai>=1.0.0",      # Required for all providers
    "instructor>=1.0.0",   # Required for non-OpenAI providers
    "pydantic>=2.0.0",     # Required for structured output
    "anthropic",           # Optional: for Anthropic's from_provider
]
```

## Troubleshooting

### Common Issues

1. **"TypeError: object can't be used in 'await' expression"**
   - Cause: Mixing sync/async clients
   - Solution: Check if provider uses sync (`from_provider`) or async (`from_openai`)

2. **"ConfigurationError: The anthropic package is required"**
   - Cause: Missing provider-specific package for `from_provider`
   - Solution: Install `anthropic` package or use `from_openai` fallback

3. **"Structured output parsing failed - no parsed content"**
   - Cause: Model doesn't support structured output or malformed schema
   - Solution: Verify model capabilities and Pydantic model definition

4. **"No API key found"**
   - Cause: Missing environment variable
   - Solution: Set appropriate API key in `.env` file

### Debugging Structured Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed tracing
client = create_llm_client(provider="deepseek")

# Test with simple model first
class SimpleTest(BaseModel):
    message: str

# Gradually increase complexity
result = await client.generate(
    messages=[{"role": "user", "content": "Say hello"}],
    output_type=SimpleTest
)
```

## Performance Considerations

### Native Parse (OpenAI/Ollama)
- **Pros**: 
  - Fastest performance
  - No additional dependencies
  - Direct API integration
- **Cons**:
  - Limited to OpenAI-compatible providers
  - Less flexibility for validation

### Instructor Method
- **Pros**:
  - Universal compatibility
  - Automatic retries on validation failure
  - Rich validation and transformation features
- **Cons**:
  - Additional dependency
  - Slight overhead from library
  - Sync/async complexity with `from_provider`

## Best Practices

1. **Always use Pydantic Field descriptions**:
```python
class Output(BaseModel):
    value: int = Field(description="The numeric value")  # Good
    value: int  # Less effective
```

2. **Start with simple models and iterate**:
```python
# Start simple
class SimpleOutput(BaseModel):
    result: str

# Then add complexity
class ComplexOutput(BaseModel):
    result: str
    confidence: float = Field(ge=0, le=1)
    metadata: Dict[str, Any]
```

3. **Handle provider-specific quirks**:
```python
if client.provider == "openai" and client._is_reasoning_model(model):
    # Reasoning models have restrictions
    params.pop("temperature", None)
```

4. **Use appropriate models for structured output**:
   - OpenAI: `gpt-4o-mini`, `gpt-4-turbo`
   - Anthropic: `claude-3-5-sonnet`
   - DeepSeek: `deepseek-chat`
   - Ollama: Any model with JSON training

## Testing Structured Output

Run the test suite to verify your implementation:

```bash
# Basic test
uv run python test_modern_instructor.py

# Integration tests
uv run pytest tests/integration/test_structured_output.py
uv run pytest tests/integration/test_deepseek_structured.py
```

## References

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Instructor Library](https://python.useinstructor.com)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [DeepSeek API](https://platform.deepseek.com/api-docs)
- [Anthropic API](https://docs.anthropic.com/claude/reference)