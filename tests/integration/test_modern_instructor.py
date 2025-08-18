#!/usr/bin/env python
"""Test the updated modern instructor implementation"""

import asyncio
import os
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from llm import create_llm_client

# Load environment variables from .env file
load_dotenv()

class TestOutput(BaseModel):
    """Test structured output"""
    message: str = Field(description="A greeting message")
    number: int = Field(description="A lucky number")

async def test_providers():
    """Test different providers with structured output"""
    
    # Test configurations with appropriate models
    test_cases = [
        # Provider that should use native parse
        {"provider": "openai", "model": "gpt-4.1-mini", "expected_method": "native"},
        {"provider": "ollama", "model": "llama3.2:3b", "expected_method": "native"},
        
        # Providers that should use instructor with from_provider
        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "expected_method": "instructor_from_provider"},
        {"provider": "deepseek", "model": "deepseek-chat", "expected_method": "instructor_from_provider"},
        
        # Provider that should use instructor with from_openai (custom endpoints)
        {"provider": "openrouter", "model": "openai/gpt-4o-mini", "expected_method": "instructor_from_openai"},
    ]
    
    for test in test_cases:
        provider = test["provider"]
        model = test["model"]
        print(f"\nTesting {provider} with {model}...")
        
        try:
            # Create client with specific model
            client = create_llm_client(provider=provider, model=model)
            
            # Test structured output
            result = await client.generate(
                messages=[{"role": "user", "content": "Say hello and give me the number 42"}],
                output_type=TestOutput
            )
            
            print(f"✓ {provider} worked! Result: {result}")
            print(f"  - Message: {result.message}")
            print(f"  - Number: {result.number}")
            
        except Exception as e:
            print(f"✗ {provider} failed: {type(e).__name__}: {e}")
            # This is expected for providers without API keys configured

if __name__ == "__main__":
    # Note: This will only work for providers with configured API keys
    print("Testing modern instructor implementation...")
    print("=" * 60)
    asyncio.run(test_providers())