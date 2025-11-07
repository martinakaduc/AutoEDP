#!/usr/bin/env python3
"""
Quick test to verify SSE parsing works correctly.
"""

import asyncio
from middleware import VLLMMiddleware


async def test_sse_parsing():
    """Test that SSE responses are parsed correctly."""

    print("=" * 70)
    print("Testing SSE Response Handling")
    print("=" * 70)

    async with VLLMMiddleware(
        base_url="http://localhost:8081/v1",
        log_dir="./logs/vllm",
        enable_file_logging=True,
    ) as middleware:

        try:
            print("\nSending test request to vLLM...")
            response = await middleware.chat_completion(
                messages=[
                    {"role": "user", "content": "Say 'Hello World' and nothing else."}
                ],
                model="vllm:qwen-2.5-omni-7b",
                temperature=0.1,
                max_tokens=10,
            )

            print("\n✅ Success!")
            print(f"\nResponse structure:")
            print(f"  ID: {response.get('id', 'N/A')}")
            print(f"  Model: {response.get('model', 'N/A')}")
            print(f"  Object: {response.get('object', 'N/A')}")

            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                message = choice.get("message", {})
                print(f"\nMessage:")
                print(f"  Role: {message.get('role', 'N/A')}")
                print(f"  Content: {message.get('content', 'N/A')}")
                print(f"  Finish Reason: {choice.get('finish_reason', 'N/A')}")

            if "usage" in response:
                usage = response["usage"]
                print(f"\nUsage:")
                print(f"  Prompt Tokens: {usage.get('prompt_tokens', 0)}")
                print(f"  Completion Tokens: {usage.get('completion_tokens', 0)}")
                print(f"  Total Tokens: {usage.get('total_tokens', 0)}")

            return True

        except Exception as e:
            print(f"\n❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            return False


async def test_through_wrapper():
    """Test through the wrapper server."""

    print("\n" + "=" * 70)
    print("Testing Through Wrapper Server")
    print("=" * 70)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("\n⚠️  OpenAI library not installed, skipping this test")
        return None

    try:
        client = AsyncOpenAI(
            base_url="http://localhost:8082/v1", api_key="dummy", timeout=30.0
        )

        print("\nSending request through wrapper...")
        response = await client.chat.completions.create(
            model="vllm:qwen-2.5-omni-7b",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            max_tokens=50,
        )

        print("\n✅ Success!")
        print(f"\nResponse:")
        print(f"  Content: {response.choices[0].message.content}")
        print(f"  Usage: {response.usage}")

        return True

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        print("\nMake sure wrapper server is running:")
        print(
            "  python wrapper_server.py --port 8082 --vllm-url http://localhost:8081/v1"
        )
        return False


async def main():
    """Run tests."""
    print("\n" + "=" * 70)
    print("SSE Response Handling Test Suite")
    print("=" * 70)
    print("\nThis will test:")
    print("1. Direct middleware SSE parsing")
    print("2. Through wrapper server (if running)")
    print("=" * 70)

    # Test 1: Direct middleware
    test1 = await test_sse_parsing()

    # Test 2: Through wrapper
    test2 = await test_through_wrapper()

    # Summary
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Direct Middleware:  {'✅ PASS' if test1 else '❌ FAIL'}")
    print(
        f"Through Wrapper:    {'✅ PASS' if test2 else '⊘ SKIP' if test2 is None else '❌ FAIL'}"
    )
    print("=" * 70)

    if test1:
        print("\n✅ SSE parsing is working!")
        print("\nYou can now use the wrapper server normally.")
        print("The middleware will automatically handle SSE responses from vLLM.")
    else:
        print("\n❌ SSE parsing failed. Check:")
        print("  1. Is vLLM running on port 8081?")
        print("  2. Is the model loaded?")
        print("  3. Check vLLM logs for errors")


if __name__ == "__main__":
    asyncio.run(main())
