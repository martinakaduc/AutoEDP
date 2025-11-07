#!/usr/bin/env python3
"""
Quick start script for testing vLLM middleware.

This script provides a simple interactive interface to test the middleware.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from middleware import VLLMMiddleware


async def test_connection(base_url: str):
    """Test connection to vLLM server."""
    print("Testing connection to vLLM server...")
    try:
        async with VLLMMiddleware(base_url=base_url, enable_file_logging=False) as mw:
            response = await mw.chat_completion(
                messages=[{"role": "user", "content": "Hello"}], max_tokens=10
            )
            print("✓ Connection successful!")
            return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


async def interactive_chat(base_url: str, log_dir: str):
    """Run an interactive chat session with logging."""
    print("\n" + "=" * 60)
    print("Interactive Chat with vLLM (type 'quit' to exit)")
    print("=" * 60)

    async with VLLMMiddleware(
        base_url=base_url, log_dir=log_dir, enable_file_logging=True
    ) as mw:

        conversation = [{"role": "system", "content": "You are a helpful assistant."}]

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                # Add to conversation
                conversation.append({"role": "user", "content": user_input})

                # Get response
                print("\nAssistant: ", end="", flush=True)
                response = await mw.chat_completion(
                    messages=conversation, temperature=0.7, max_tokens=500
                )

                assistant_message = response["choices"][0]["message"]["content"]
                print(assistant_message)

                # Add to conversation
                conversation.append({"role": "assistant", "content": assistant_message})

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

        # Show statistics
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        stats = mw.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")


async def run_test_queries(base_url: str, log_dir: str):
    """Run a set of test queries."""
    print("\n" + "=" * 60)
    print("Running Test Queries")
    print("=" * 60)

    test_cases = [
        {
            "name": "Simple question",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0.1,
        },
        {
            "name": "Creative writing",
            "messages": [{"role": "user", "content": "Write a haiku about coding."}],
            "temperature": 0.9,
        },
        {
            "name": "Technical explanation",
            "messages": [
                {"role": "system", "content": "You are a technical expert."},
                {"role": "user", "content": "Explain recursion in 2 sentences."},
            ],
            "temperature": 0.5,
        },
    ]

    async with VLLMMiddleware(
        base_url=base_url, log_dir=log_dir, enable_file_logging=True
    ) as mw:

        for i, test in enumerate(test_cases, 1):
            print(f"\n{i}. {test['name']}")
            print("-" * 40)

            try:
                response = await mw.chat_completion(
                    messages=test["messages"],
                    temperature=test["temperature"],
                    max_tokens=200,
                )

                content = response["choices"][0]["message"]["content"]
                print(f"Response: {content}")

                if "usage" in response:
                    print(f"Tokens: {response['usage']}")

            except Exception as e:
                print(f"Error: {e}")

        # Show statistics
        print("\n" + "=" * 60)
        print("Test Statistics")
        print("=" * 60)
        stats = mw.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick start script for vLLM middleware"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8081/v1", help="Base URL of vLLM server"
    )
    parser.add_argument(
        "--log-dir", default="./logs/vllm", help="Directory for log files"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "interactive", "check"],
        default="check",
        help="Mode: check connection, run tests, or interactive chat",
    )

    args = parser.parse_args()

    print("vLLM Middleware Quick Start")
    print("=" * 60)
    print(f"Server: {args.base_url}")
    print(f"Logs: {args.log_dir}")
    print("=" * 60)

    if args.mode == "check":
        # Just test connection
        success = asyncio.run(test_connection(args.base_url))
        if success:
            print("\n✓ Ready to use!")
            print("\nNext steps:")
            print(f"  1. Run tests: python {__file__} --mode test")
            print(f"  2. Interactive: python {__file__} --mode interactive")
        else:
            print("\n✗ Server not available")
            print("\nMake sure vLLM server is running:")
            print("  vllm serve Qwen/Qwen2.5-Omni-7B \\")
            print("      --served-model-name vllm:qwen-2.5-omni-7b \\")
            print("      --port 8081 \\")
            print("      --max-model-len 32768 \\")
            print("      --tensor-parallel-size 1 \\")
            print("      --enable-auto-tool-choice \\")
            print("      --tool-call-parser hermes")
            sys.exit(1)

    elif args.mode == "test":
        asyncio.run(run_test_queries(args.base_url, args.log_dir))

    elif args.mode == "interactive":
        asyncio.run(interactive_chat(args.base_url, args.log_dir))


if __name__ == "__main__":
    main()
