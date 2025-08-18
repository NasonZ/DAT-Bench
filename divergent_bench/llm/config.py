"""
LLM Configuration Management

Handles loading configuration from environment variables and
provides structured config objects.
"""

import os
from dataclasses import dataclass
from typing import Optional, Union
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str = "openai"
    model: str = "gpt-4.1-nano"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 32000
    
    # Optional multi-model support
    fast_model: Optional[str] = None
    reasoning_model: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create config from environment variables."""
        # Load .env file if it exists
        load_dotenv(override=True)
        
        # Get provider and model
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-4.1-nano")
        
        # Get API key based on provider
        api_key_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "local": None,  # No API key needed
            "openrouter": "OPENROUTER_API_KEY"
        }
        
        api_key = None
        if provider in api_key_env_vars and api_key_env_vars[provider]:
            api_key = os.getenv(api_key_env_vars[provider])
            
        # Get other settings
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "32000"))
        
        # Optional multi-model support (from DRG pattern)
        fast_model = os.getenv("FAST_MODEL")
        reasoning_model = os.getenv("REASONING_MODEL")
        
        # Base URL override
        base_url = None
        if provider == "local":
            base_url = os.getenv("LOCAL_MODEL_URL", os.getenv("OLLAMA_API_BASE"))
            
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            fast_model=fast_model,
            reasoning_model=reasoning_model
        )


@dataclass
class MultiModelConfig:
    """Configuration for multi-model setups (DRG pattern)."""
    reasoning: LLMConfig
    main: LLMConfig
    fast: LLMConfig
    
    @classmethod
    def from_env(cls) -> 'MultiModelConfig':
        """Create multi-model config from environment."""
        load_dotenv(override=True)
        
        # Reasoning model (complex tasks)
        reasoning = LLMConfig(
            provider=os.getenv("REASONING_MODEL_PROVIDER", "openai"),
            model=os.getenv("REASONING_MODEL", "o4-mini"),
            temperature=0.7
        )
        
        # Main model (quality outputs)
        main = LLMConfig(
            provider=os.getenv("MAIN_MODEL_PROVIDER", "openai"),
            model=os.getenv("MAIN_MODEL", "gpt-5.1-mini"),
            temperature=0.7
        )
        
        # Fast model (quick operations)
        fast = LLMConfig(
            provider=os.getenv("FAST_MODEL_PROVIDER", "openai"),
            model=os.getenv("FAST_MODEL", "gpt-4.1-nano"),
            temperature=0.7
        )
        
        # Load API keys
        for config in [reasoning, main, fast]:
            if config.provider == "openai":
                config.api_key = os.getenv("OPENAI_API_KEY")
            elif config.provider == "anthropic":
                config.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif config.provider == "deepseek":
                config.api_key = os.getenv("DEEPSEEK_API_KEY")
                
        return cls(reasoning=reasoning, main=main, fast=fast)


def load_llm_config(multi_model: bool = False) -> Union[LLMConfig, MultiModelConfig]:
    """
    Load LLM configuration from environment.
    
    Args:
        multi_model: Whether to load multi-model config (DRG pattern)
        
    Returns:
        LLMConfig or MultiModelConfig instance
    """
    if multi_model:
        return MultiModelConfig.from_env()
    else:
        return LLMConfig.from_env()