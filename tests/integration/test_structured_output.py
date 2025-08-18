import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class TestOutput(BaseModel):
    """Simple test output"""
    message: str = Field(description="A test message")
    number: int = Field(description="A test number")

class KnowledgeGapOutput(BaseModel):
    """Output from the Knowledge Gap Agent - matching the research example"""
    research_complete: bool = Field(description="Whether the research is complete enough to end the loop")
    outstanding_gaps: List[str] = Field(description="List of knowledge gaps that still need to be addressed")

async def test_structured_output():
    """Test if structured output works with the current model"""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    model = os.getenv("LLM_MODEL", "o4-mini")

    model = "o4-mini"

    print(f"Testing model: {model}")
    
    # Test 1: Direct parse method with simple output
    try:
        print("\nTest 1: Using parse() with simple Pydantic model...")
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": "Say hello and give me the number 42"}],
            response_format=TestOutput
        )
        print(f"✓ Success! Result: {response.choices[0].message.parsed}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: Parse with KnowledgeGapOutput (matching the failing case)
    try:
        print("\nTest 2: Using parse() with KnowledgeGapOutput...")
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are analyzing research gaps."},
                {"role": "user", "content": "Are there any gaps in understanding AI safety? List them."}
            ],
            response_format=KnowledgeGapOutput
        )
        print(f"✓ Success! Result: {response.choices[0].message.parsed}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Try with additional parameters (like the actual code does)
    try:
        print("\nTest 3: Using parse() with extra parameters...")
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": "Say hello and give me the number 42"}],
            response_format=TestOutput,
            temperature=0.7,
            max_tokens=1000,
            # This might be the issue - extra params that parse() doesn't accept
            tool_choice="auto"  # This would cause an error
        )
        print(f"✓ Success! Result: {response.choices[0].message.parsed}")
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")
    
    # Test 4: Function calling approach (alternative)
    try:
        print("\nTest 4: Using function calling for structured output...")
        function_schema = {
            "name": "output_knowledge_gaps",
            "description": "Output the knowledge gap analysis",
            "parameters": KnowledgeGapOutput.model_json_schema()
        }
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are analyzing research gaps."},
                {"role": "user", "content": "Are there any gaps in understanding AI safety? List them."}
            ],
            tools=[{"type": "function", "function": function_schema}],
            tool_choice={"type": "function", "function": {"name": "output_knowledge_gaps"}}
        )
        
        if response.choices[0].message.tool_calls:
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            print(f"✓ Success! Result: {args}")
        else:
            print("✗ No tool calls in response")
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_structured_output())