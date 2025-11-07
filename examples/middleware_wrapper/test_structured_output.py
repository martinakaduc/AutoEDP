"""
Test structured output with the wrapper server.

This script tests whether vLLM properly supports JSON mode and structured outputs.
"""

import asyncio
import json
from openai import AsyncOpenAI


async def test_json_mode():
    """Test basic JSON mode."""
    print("\n" + "=" * 70)
    print("Test 1: JSON Mode (response_format={type: 'json_object'})")
    print("=" * 70)

    client = AsyncOpenAI(base_url="http://localhost:8082/v1", api_key="dummy")

    try:
        response = await client.chat.completions.create(
            model="vllm:qwen-2.5-omni-7b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Always respond with valid JSON.",
                },
                {
                    "role": "user",
                    "content": "Tell me about Python in JSON format with keys: name, type, created_year",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=200,
        )

        content = response.choices[0].message.content
        print(f"Response: {content}")

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"✅ Valid JSON!")
            print(f"Parsed: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_json_schema():
    """Test JSON schema (OpenAI's new structured output feature)."""
    print("\n" + "=" * 70)
    print("Test 2: JSON Schema (response_format with schema)")
    print("=" * 70)

    client = AsyncOpenAI(base_url="http://localhost:8082/v1", api_key="dummy")

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "clarification_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "need_clarification": {
                        "type": "boolean",
                        "description": "Whether clarification is needed",
                    },
                    "question": {
                        "type": "string",
                        "description": "Question to ask the user",
                    },
                    "verification": {
                        "type": "string",
                        "description": "Verification message",
                    },
                },
                "required": ["need_clarification", "question", "verification"],
                "additionalProperties": False,
            },
        },
    }

    try:
        response = await client.chat.completions.create(
            model="vllm:qwen-2.5-omni-7b",
            messages=[
                {
                    "role": "user",
                    "content": "I want to research climate change. Do you need clarification?",
                }
            ],
            response_format=schema,
            temperature=0.7,
            max_tokens=200,
        )

        content = response.choices[0].message.content
        print(f"Response: {content}")

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"✅ Valid JSON!")
            print(f"Parsed: {json.dumps(parsed, indent=2)}")

            # Check schema compliance
            if all(
                k in parsed for k in ["need_clarification", "question", "verification"]
            ):
                print("✅ Schema compliant!")
            else:
                print("❌ Missing required fields")

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Note: vLLM might not support JSON schema yet")
        import traceback

        traceback.print_exc()


async def test_tool_calling():
    """Test tool calling for structured output (alternative approach)."""
    print("\n" + "=" * 70)
    print("Test 3: Tool Calling (alternative for structured output)")
    print("=" * 70)

    client = AsyncOpenAI(base_url="http://localhost:8082/v1", api_key="dummy")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "provide_clarification_decision",
                "description": "Provide a decision on whether clarification is needed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "need_clarification": {
                            "type": "boolean",
                            "description": "Whether clarification is needed",
                        },
                        "question": {
                            "type": "string",
                            "description": "Question to ask the user if clarification is needed",
                        },
                        "verification": {
                            "type": "string",
                            "description": "Verification message if no clarification needed",
                        },
                    },
                    "required": ["need_clarification", "question", "verification"],
                },
            },
        }
    ]

    try:
        response = await client.chat.completions.create(
            model="vllm:qwen-2.5-omni-7b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant. When asked about research requests, use the tool to indicate if clarification is needed.",
                },
                {"role": "user", "content": "I want to research climate change."},
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "provide_clarification_decision"},
            },
            temperature=0.7,
            max_tokens=200,
        )

        message = response.choices[0].message
        print(f"Finish reason: {response.choices[0].finish_reason}")

        if message.tool_calls:
            print(f"✅ Tool called!")
            for tool_call in message.tool_calls:
                print(f"Tool: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")

                try:
                    args = json.loads(tool_call.function.arguments)
                    print(f"Parsed arguments: {json.dumps(args, indent=2)}")

                    if all(
                        k in args
                        for k in ["need_clarification", "question", "verification"]
                    ):
                        print("✅ All required fields present!")
                    else:
                        print("❌ Missing required fields")
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in tool arguments: {e}")
        else:
            print(f"❌ No tool calls. Content: {message.content}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Structured Output Support with vLLM + Wrapper Server")
    print("=" * 70)

    await test_json_mode()
    await test_json_schema()
    await test_tool_calling()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("1. If JSON mode works: vLLM supports basic JSON output")
    print("2. If JSON schema works: vLLM supports full structured output")
    print("3. If tool calling works: Use tool calling as fallback")
    print("\nRecommendation:")
    print("- If tool calling works best, modify deep_researcher.py to use")
    print("  tool calling instead of with_structured_output()")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
