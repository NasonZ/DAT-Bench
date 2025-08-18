"""
Multi-provider LLM support via OpenAI-compatible interface

Supports OpenAI, Anthropic, DeepSeek, Gemini, and more through
a unified OpenAI client interface with Instructor for structured outputs.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union, Type
from pydantic import BaseModel

from openai import AsyncOpenAI, AsyncAzureOpenAI

from .client import BaseLLMClient, LLMResponse, LLMMessage

logger = logging.getLogger(__name__)


# Provider configurations
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": None,
        "supports_tools": True,
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "supports_tools": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "supports_tools": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "supports_tools": True,
    },
    "local": {
        "base_url": None,  # Will use LOCAL_MODEL_URL or OLLAMA_BASE_URL env var
        "supports_tools": True,
    },
    "ollama": {
        "base_url": None,  # Will use OLLAMA_BASE_URL env var
        "supports_tools": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "supports_tools": True,
    }
}


# Model capabilities registry - single source of truth
MODEL_CAPABILITIES = {
    # Pattern order matters! Specific patterns must come before general ones
    'reasoning_api': ['gpt-5', 'o1-', 'o3-', 'o4-', 'o1', 'o3', 'o4'],
    'reasoning_restricted': ['o1-', 'o3-', 'o4-', 'o1', 'o3', 'o4'],  # No temperature etc
    'completion_tokens': ['gpt-5', 'o1-', 'o3-', 'o4-', 'o1', 'o3', 'o4'],

    # How to extend for other providers?
}


class OpenAICompatibleClient(BaseLLMClient):
    """
    Multi-provider LLM client using OpenAI-compatible interface.
    
    Supports structured outputs and tool calling where available.
    """
    
    @property
    def supports_structured_output(self) -> bool:
        """Check if this client supports structured outputs."""
        return True  # All OpenAI-compatible APIs support structured output
    
    @property
    def supports_tools(self) -> bool:
        """Check if this client supports tool calling."""
        return getattr(self, 'provider_config', {}).get("supports_tools", True)
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize client for specified provider.
        
        Args:
            provider: Provider name (openai, anthropic, deepseek, etc.)
            api_key: API key (will use env var if not provided)
            model: Model name (provider-specific)
            base_url: Override base URL
            **kwargs: Additional configuration
        """
        # Get provider config
        if provider not in PROVIDER_CONFIGS:
            raise ValueError(f"Unknown provider: {provider}. Supported: {list(PROVIDER_CONFIGS.keys())}")
            
        self.provider = provider
        self.provider_config = PROVIDER_CONFIGS[provider]
        
        # Set defaults
        if not model:
            model = self._get_default_model()
        
        super().__init__(model=model, **kwargs)
        
        # Store base_url for instructor routing decision
        self._base_url = base_url
        
        # Initialize OpenAI client
        self._client = self._create_client(api_key, base_url)
        
        # Store last response for usage tracking
        self.last_response = None
        
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-5.1-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-pro",
            "local": "llama3.2",
            "ollama": "llama3.2",
            "openrouter": "openai/gpt-4o-mini"
        }
        return defaults.get(self.provider, "gpt-4o-mini")
        
    def _create_client(self, api_key: Optional[str], base_url: Optional[str]) -> AsyncOpenAI:
        """Create the OpenAI client instance."""
        # Get API key from env if not provided
        if not api_key:
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", 
                "deepseek": "DEEPSEEK_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "local": None,  # No key needed
                "ollama": None,  # No key needed
                "openrouter": "OPENROUTER_API_KEY"
            }
            env_var = env_vars.get(self.provider, f"{self.provider.upper()}_API_KEY")
            if env_var:
                api_key = os.getenv(env_var)
            
            if not api_key and self.provider not in ["local", "ollama"]:
                raise ValueError(f"No API key found. Set {env_var} environment variable.")
                
        # Get base URL
        if not base_url:
            base_url = self.provider_config.get("base_url")
            
        # Special handling for ollama/local models
        if self.provider in ["local", "ollama"] and not base_url:
            # Check for Ollama-specific env var first
            if self.provider == "ollama":
                base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")
            
            # Fallback to generic local model URL
            if not base_url:
                base_url = os.getenv("LOCAL_MODEL_URL") or "http://localhost:11434"
            
            # Ensure base_url ends with /v1 for OpenAI compatibility
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            
            if not api_key:
                api_key = "ollama"  # Dummy key for local models
                
        # Create client
        client_kwargs = {
            "api_key": api_key,
        }
        
        if base_url:
            client_kwargs["base_url"] = base_url
        
        # Add timeout for local/ollama models which might be slow
        if self.provider in ["local", "ollama"]:
            import httpx
            # Use longer timeout for Ollama's slow responses
            client_kwargs["timeout"] = httpx.Timeout(
                connect=10.0,     # 10 seconds to establish connection
                read=300.0,       # 5 minutes to read response (Ollama can be slow)
                write=10.0,       # 10 seconds to write request
                pool=10.0         # 10 seconds to acquire connection from pool
            )
            # Disable retries since Ollama is just slow, not failing
            client_kwargs["max_retries"] = 0
            
        return AsyncOpenAI(**client_kwargs)
        
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if model is a reasoning model with restricted parameters."""
        return any(model.startswith(pattern) for pattern in MODEL_CAPABILITIES['reasoning_restricted'])
    
    def _is_reasoning_api_model(self, model: str) -> bool:
        """Check if model should use the Responses API instead of Chat Completions."""
        # GPT-5 series and o-series models use the Responses API
        return self.provider == "openai" and any(model.startswith(pattern) for pattern in MODEL_CAPABILITIES['reasoning_api'])
    
    def _uses_completion_tokens(self, model: str) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        # GPT-5 series and reasoning models use max_completion_tokens
        return any(model.startswith(pattern) for pattern in MODEL_CAPABILITIES['completion_tokens'])
    
    def _supports_native_parse(self, model: str) -> bool:
        """Check if model supports native parse() API.
        
        OpenAI and Ollama support native parse().
        """
        return self.provider in ["openai", "ollama"]
    
    @property
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return self.provider_config.get("supports_tools", True)
    
    def get_last_usage(self) -> Optional[Dict[str, Any]]:
        """Get usage info from last response, including reasoning tokens if available."""
        if not self.last_response or not hasattr(self.last_response, 'usage'):
            return None
        
        usage = self.last_response.usage
        if not usage:
            return None
            
        # Convert to dict and include detailed token breakdown
        usage_dict = usage.model_dump() if hasattr(usage, 'model_dump') else dict(usage)
        
        # Extract reasoning tokens if available (GPT-5/o4 models)
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            details = usage.completion_tokens_details
            if hasattr(details, 'reasoning_tokens'):
                usage_dict['reasoning_tokens'] = details.reasoning_tokens
                
        return usage_dict
    
    def get_last_reasoning_summary(self) -> Optional[str]:
        """Get reasoning summary from last response if available."""
        if hasattr(self, '_last_reasoning_summary'):
            return self._last_reasoning_summary
        return None
        
    async def generate(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        model: Optional[str] = None,
        output_type: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        Generate response using OpenAI-compatible API.
        
        Returns structured output instance if output_type is provided,
        otherwise returns LLMResponse.
        """
        # Normalize messages
        messages = self._normalize_messages(messages)
        
        # Build parameters
        current_model = model or self.model
        params = {
            "model": current_model,
            "messages": messages,
            **kwargs
        }
        
        # Handle model-specific parameter requirements
        if self.provider == "openai":
            if self._uses_completion_tokens(current_model):
                # GPT-5 and reasoning models use max_completion_tokens
                if max_tokens is not None:
                    params["max_completion_tokens"] = max_tokens
                elif self.default_max_tokens:
                    params["max_completion_tokens"] = self.default_max_tokens
                # Remove max_tokens if present
                params.pop("max_tokens", None)
                
                # Reasoning models have additional restrictions
                if self._is_reasoning_model(current_model):
                    # Remove restricted parameters for reasoning models
                    restricted_params = ['temperature', 'top_p', 'presence_penalty', 
                                       'frequency_penalty', 'logit_bias']
                    for param in restricted_params:
                        params.pop(param, None)
                else:
                    # GPT-5 models can use temperature
                    params["temperature"] = temperature or self.default_temperature
            else:
                # Normal OpenAI models
                params["temperature"] = temperature or self.default_temperature
                params["max_tokens"] = max_tokens or self.default_max_tokens
        else:
            # Non-OpenAI providers
            params["temperature"] = temperature or self.default_temperature
            params["max_tokens"] = max_tokens or self.default_max_tokens
        
        # Add tools if provided and supported
        if tools and self.supports_tools:
            params["tools"] = tools
            if "tool_choice" not in kwargs:
                params["tool_choice"] = "auto"
                
        try:
            # Route 0: Reasoning API for GPT-5/o-series models
            if self._is_reasoning_api_model(current_model):
                return await self._reasoning_generate(params, output_type)
            
            # Route 1: Native parse for OpenAI/Ollama with structured output
            elif output_type and self._supports_native_parse(current_model):
                return await self._native_parse_generate(params, output_type)
                
            # Route 2: Instructor for all other providers with structured output
            elif output_type:
                return await self._instructor_generate(params, output_type)
                
            # Route 3: Regular completion (no structured output)
            else:
                return await self._regular_generate(params)
                
        except Exception as e:
            logger.error(f"LLM generation failed for {self.provider}/{current_model}: {e}")
            raise
    
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
        
        # Store response for usage tracking
        self.last_response = response
        
        if response.choices and response.choices[0].message.parsed is not None:
            return response.choices[0].message.parsed
        else:
            raise ValueError("Structured output parsing failed - no parsed content")
    
    async def _instructor_generate(self, params: Dict, output_type: Type[BaseModel]) -> Any:
        """Generate using Instructor for structured output."""
        try:
            import instructor
        except ImportError:
            raise ImportError("Instructor is required for structured output with non-OpenAI providers. Install with: pip install instructor")
        
        # Use modern from_provider for standard providers with known endpoints
        # Fall back to from_openai for custom/local endpoints
        model_name = params.get("model", self.model)
        
        if self.provider in ["anthropic", "deepseek", "gemini"] and not self._base_url:
            # Modern approach for standard providers
            provider_string = f"{self.provider}/{model_name}"
            instructor_client = instructor.from_provider(
                provider_string,
                api_key=self._client.api_key
            )
            # from_provider returns a sync client, so we need to use the sync method
            result = instructor_client.chat.completions.create(
                **params,
                response_model=output_type,
                max_retries=3  # Automatic validation retries
            )
        else:
            # Fall back to from_openai for custom endpoints, local models, openrouter
            instructor_client = instructor.from_openai(self._client)
            # from_openai with AsyncOpenAI returns an async client
            result = await instructor_client.chat.completions.create(
                **params,
                response_model=output_type,
                max_retries=3  # Automatic validation retries
            )
        
        return result  # Already parsed and validated!
    
    async def _regular_generate(self, params: Dict) -> LLMResponse:
        """Generate regular completion without structured output."""
        response = await self._client.chat.completions.create(**params)
        
        # Extract message
        message = response.choices[0].message
        
        # Build response using standardized method
        llm_response = self._build_llm_response(
            content=message.content or "",
            raw_response=response,
            model=response.model,
            usage=response.usage
        )
        
        # Add tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            llm_response.tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
            
        return llm_response
    
    def _build_llm_response(
        self,
        content: str,
        raw_response: Any,
        model: Optional[str] = None,
        usage: Optional[Any] = None,
        reasoning_summary: Optional[str] = None
    ) -> LLMResponse:
        """Build standardized LLMResponse with consistent handling.
        
        Args:
            content: The text content of the response
            raw_response: The original response object
            model: Model name (extracted from response or params)
            usage: Usage statistics (will be normalized)
            reasoning_summary: Optional reasoning summary for metadata
        """
        # Normalize usage data
        usage_dict = None
        if usage:
            if hasattr(usage, 'model_dump'):
                usage_dict = usage.model_dump()
            elif isinstance(usage, dict):
                usage_dict = usage
            else:
                # Try to convert to dict
                try:
                    usage_dict = dict(usage)
                except:
                    pass
        
        # Build the response
        response = LLMResponse(
            content=content,
            raw_response=raw_response,
            model=model or getattr(raw_response, 'model', None) or self.model,
            usage=usage_dict
        )
        
        # Add reasoning summary to metadata if provided
        if reasoning_summary:
            response.metadata = response.metadata or {}
            response.metadata['reasoning_summary'] = reasoning_summary
            
        return response
    
    def _parse_structured_from_text(self, content_text: str, output_type: Type[BaseModel]) -> Optional[BaseModel]:
        """Parse structured output from text response.
        
        Tries multiple strategies:
        1. JSON array extraction 
        2. Line-by-line parsing for list types
        
        This is mainly for the Responses API which doesn't yet support structured output directly.
        """
        import re
        import json
        
        # Check if the output type expects a list field (like DATWords.words)
        # This is a generic approach that works for any model with a list field
        list_fields = [
            field_name for field_name, field_info in output_type.__fields__.items()
            if hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ == list
        ]
        
        if not list_fields:
            # Can't parse non-list structured output from text reliably
            return None
            
        # Use the first list field found
        list_field = list_fields[0]
        
        # Strategy 1: Try to extract JSON array
        json_match = re.search(r'\[([^\]]+)\]', content_text)
        if json_match:
            try:
                words_str = '[' + json_match.group(1) + ']'
                items = json.loads(words_str)
                if isinstance(items, list) and len(items) >= 10:
                    return output_type(**{list_field: items[:10]})
            except:
                pass
        
        # Strategy 2: Line-by-line parsing
        lines = content_text.split('\n')
        items = []
        for line in lines:
            # Remove common list markers: numbers, bullets, dashes
            cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line.strip())
            if cleaned and len(cleaned) > 1:
                # Extract the main word/item (before any separator like dash or colon)
                item = cleaned.split('-')[0].split(':')[0].strip().lower()
                if item and item.isalpha():
                    items.append(item)
        
        if len(items) >= 10:
            return output_type(**{list_field: items[:10]})
            
        return None
    
    def _extract_response_content(self, response) -> tuple[str, str]:
        """Extract content and reasoning summary from Responses API output.
        
        Handles both object-based and dict-based response formats transparently.
        """
        content_text = ""
        reasoning_summary = ""
        
        if not (hasattr(response, 'output') and response.output):
            return content_text, reasoning_summary
            
        for item in response.output:
            # Normalize access: convert object to dict if needed
            item_data = item if isinstance(item, dict) else {
                'type': getattr(item, 'type', None),
                'content': getattr(item, 'content', []),
                'summary': getattr(item, 'summary', [])
            }
            
            if item_data.get('type') == 'message':
                # Extract message content
                for content_item in item_data.get('content', []):
                    # Again normalize: dict or object
                    if isinstance(content_item, dict):
                        if content_item.get('type') == 'output_text':
                            content_text = content_item.get('text', '')
                    elif hasattr(content_item, 'type') and content_item.type == 'output_text':
                        content_text = getattr(content_item, 'text', '')
                        
            elif item_data.get('type') == 'reasoning':
                # Extract reasoning summary
                for summary_item in item_data.get('summary', []):
                    if isinstance(summary_item, dict):
                        if summary_item.get('type') == 'summary_text':
                            reasoning_summary = summary_item.get('text', '')
                    elif hasattr(summary_item, 'type') and summary_item.type == 'summary_text':
                        reasoning_summary = getattr(summary_item, 'text', '')
                        
        return content_text, reasoning_summary
    
    async def _reasoning_generate(self, params: Dict, output_type: Optional[Type[BaseModel]] = None) -> Union[LLMResponse, Any]:
        """Generate using the Responses API for reasoning models."""
        # Clear previous reasoning summary
        self._last_reasoning_summary = None
        
        # Transform chat completions params to responses API format
        responses_params = {
            "model": params["model"],
            "input": params["messages"],  # 'messages' becomes 'input'
            "reasoning": {
                "effort": params.get("reasoning_effort", "medium")
            }
        }
        
        # Add reasoning summary if requested (requires org verification)
        if params.get("reasoning_summary"):
            responses_params["reasoning"]["summary"] = params["reasoning_summary"]
        
        # Handle max_output_tokens (not max_completion_tokens)
        if "max_completion_tokens" in params:
            responses_params["max_output_tokens"] = params["max_completion_tokens"]
        elif "max_tokens" in params:
            responses_params["max_output_tokens"] = params["max_tokens"]
        
        # Temperature is supported for GPT-5 but not for o-series reasoning models
        if params.get("temperature") and params["model"].startswith("gpt-5"):
            # GPT-5 models only support temperature=1.0
            responses_params["temperature"] = 1.0
        
        # Note: Responses API doesn't support structured output yet
        # We'll need to parse the text output for DAT words
        
        try:
            # Call the Responses API
            response = await self._client.responses.create(**responses_params)
            
            # Store for usage tracking
            self.last_response = response
            
            # Parse the output array to extract content and reasoning
            content_text, reasoning_summary = self._extract_response_content(response)
            
            # Handle structured output by parsing the text if needed
            if output_type and content_text:
                parsed_output = self._parse_structured_from_text(content_text, output_type)
                if parsed_output:
                    # Store reasoning summary for later retrieval
                    if reasoning_summary:
                        self._last_reasoning_summary = reasoning_summary
                    return parsed_output
            
            # Build LLMResponse using standardized method
            llm_response = self._build_llm_response(
                content=content_text or getattr(response, 'output_text', ""),
                raw_response=response,
                model=getattr(response, 'model', params.get("model")),
                usage=getattr(response, 'usage', None),
                reasoning_summary=reasoning_summary
            )
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Reasoning API generation failed: {e}")
            # Fallback to chat completions API
            logger.info("Falling back to chat completions API")
            # Remove reasoning-specific params before fallback
            fallback_params = dict(params)
            fallback_params.pop("reasoning_effort", None)
            fallback_params.pop("reasoning_summary", None)
            return await self._regular_generate(fallback_params)


def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> OpenAICompatibleClient:
    """
    Factory function to create LLM client.
    
    Args:
        provider: Provider name (uses LLM_PROVIDER env var if not specified)
        model: Model name (uses LLM_MODEL env var if not specified)
        **kwargs: Additional client configuration
        
    Returns:
        Configured LLM client
    """
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "openai")
    
    if not model:
        model = os.getenv("LLM_MODEL")
        
    return OpenAICompatibleClient(provider=provider, model=model, **kwargs)