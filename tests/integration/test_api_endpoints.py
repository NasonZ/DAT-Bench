import asyncio
import os
import inspect
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class TestOutput(BaseModel):
    """Simple test output"""
    message: str = Field(description="A test message")
    number: int = Field(description="A test number")

async def test_api_endpoints():
    """Test API endpoints and parameter handling"""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    
    print(f"Testing with model: {model}")
    print("=" * 60)
    
    # 1. Check parse() method parameters
    print("\n1. Inspecting parse() method:")
    parse_method = client.beta.chat.completions.parse
    parse_sig = inspect.signature(parse_method)
    print(f"   Signature: {parse_sig}")
    print(f"   Parameters: {list(parse_sig.parameters.keys())}")
    
    # Get docstring if available
    if parse_method.__doc__:
        print(f"   Docstring snippet: {parse_method.__doc__[:200]}...")
    
    # 2. Check create() method parameters
    print("\n2. Inspecting create() method:")
    create_method = client.chat.completions.create
    create_sig = inspect.signature(create_method)
    print(f"   Parameters: {list(create_sig.parameters.keys())}")
    
    # 3. Test parse() with various parameters
    print("\n3. Testing parse() parameter acceptance:")
    
    messages = [{"role": "user", "content": "Say hello and number 42"}]
    
    # Test cases with different parameter combinations
    test_cases = [
        {
            "name": "Basic (model, messages)",
            "params": {
                "model": model,
                "messages": messages,
            }
        },
        {
            "name": "With temperature",
            "params": {
                "model": model,
                "messages": messages,
                "temperature": 0.5,
            }
        },
        {
            "name": "With max_tokens",
            "params": {
                "model": model,
                "messages": messages,
                "max_tokens": 100,
            }
        },
        {
            "name": "With stream=False",
            "params": {
                "model": model,
                "messages": messages,
                "stream": False,
            }
        },
        {
            "name": "With n=1",
            "params": {
                "model": model,
                "messages": messages,
                "n": 1,
            }
        },
        {
            "name": "With tool_choice (should fail)",
            "params": {
                "model": model,
                "messages": messages,
                "tool_choice": "auto",
            }
        },
        {
            "name": "With top_p",
            "params": {
                "model": model,
                "messages": messages,
                "top_p": 0.9,
            }
        },
        {
            "name": "With presence_penalty",
            "params": {
                "model": model,
                "messages": messages,
                "presence_penalty": 0.1,
            }
        },
        {
            "name": "With frequency_penalty",
            "params": {
                "model": model,
                "messages": messages,
                "frequency_penalty": 0.1,
            }
        },
    ]
    
    for test_case in test_cases:
        try:
            print(f"\n   Testing: {test_case['name']}")
            response = await client.beta.chat.completions.parse(
                **test_case['params'],
                response_format=TestOutput
            )
            print(f"      ✓ Success! Parsed: {response.choices[0].message.parsed}")
        except Exception as e:
            error_msg = str(e)
            if "400" in error_msg:
                # Extract the parameter name from error message
                import re
                param_match = re.search(r"'(\w+)'", error_msg)
                param_name = param_match.group(1) if param_match else "unknown"
                print(f"      ✗ Failed: Parameter '{param_name}' not accepted")
            else:
                print(f"      ✗ Failed: {type(e).__name__}: {error_msg[:80]}...")
    
    # 4. Test reasoning model constraints (if applicable)
    if "o4-mini" in model:
        print("\n4. Testing o4-mini model constraints:")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7
            )
            print("   ✗ Temperature accepted (unexpected for o1)")
        except Exception as e:
            print(f"   ✓ Temperature rejected as expected: {type(e).__name__}")
    
    # 5. Check what parameters are actually used in AGE's codebase
    print("\n5. Common parameters usage:")
    print("   - From runners/agents: model, temperature, max_tokens, tools, tool_choice")
    print("   - From examples: stream, n, presence_penalty, frequency_penalty")
    print("   - Provider-specific: top_p, stop, logit_bias, user")

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())