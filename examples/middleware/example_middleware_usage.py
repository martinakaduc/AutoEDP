"""
Example usage of the VLLMMiddleware for logging queries and responses.

This script demonstrates how to use the middleware to capture and log
interactions with the vLLM server.
"""

import asyncio
from middleware import VLLMMiddleware, load_logs


async def example_chat_completion():
    """Example of using middleware for chat completion."""

    async with VLLMMiddleware(
        base_url="http://localhost:8081/v1",
        log_dir="./logs/vllm",
        enable_file_logging=True,
    ) as middleware:

        # Example 1: Simple chat completion
        print("=" * 60)
        print("Example 1: Simple Chat Completion")
        print("=" * 60)

        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": "What are the key factors in climate change?"},
        ]

        try:
            response = await middleware.chat_completion(
                messages=messages,
                model="vllm:qwen-2.5-omni-7b",
                temperature=0.7,
                max_tokens=200,
            )

            print(f"\nResponse: {response['choices'][0]['message']['content']}")
            print(f"\nUsage: {response.get('usage', {})}")

        except Exception as e:
            print(f"Error: {e}")

        # Example 2: Multi-turn conversation
        print("\n" + "=" * 60)
        print("Example 2: Multi-turn Conversation")
        print("=" * 60)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a fun fact about Python programming."},
        ]

        try:
            response = await middleware.chat_completion(
                messages=conversation,
                model="vllm:qwen-2.5-omni-7b",
                temperature=0.8,
                max_tokens=150,
            )

            assistant_message = response["choices"][0]["message"]["content"]
            print(f"\nAssistant: {assistant_message}")

            # Continue the conversation
            conversation.append({"role": "assistant", "content": assistant_message})
            conversation.append(
                {"role": "user", "content": "That's interesting! Tell me more."}
            )

            response = await middleware.chat_completion(
                messages=conversation,
                model="vllm:qwen-2.5-omni-7b",
                temperature=0.8,
                max_tokens=150,
            )

            print(f"\nAssistant: {response['choices'][0]['message']['content']}")

        except Exception as e:
            print(f"Error: {e}")

        # Example 3: Using tools/function calling
        print("\n" + "=" * 60)
        print("Example 3: Function Calling")
        print("=" * 60)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather like in Paris?"}]

        try:
            response = await middleware.chat_completion(
                messages=messages,
                model="vllm:qwen-2.5-omni-7b",
                temperature=0.7,
                tools=tools,
                tool_choice="auto",
            )

            print(f"\nResponse: {response}")

        except Exception as e:
            print(f"Error: {e}")

        # Print statistics
        print("\n" + "=" * 60)
        print("Middleware Statistics")
        print("=" * 60)
        stats = middleware.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")


async def example_completions():
    """Example of using middleware for text completion."""

    async with VLLMMiddleware(
        base_url="http://localhost:8081/v1",
        log_dir="./logs/vllm",
        enable_file_logging=True,
    ) as middleware:

        print("=" * 60)
        print("Example: Text Completion")
        print("=" * 60)

        prompt = "Once upon a time in a land far away,"

        try:
            response = await middleware.completions(
                prompt=prompt,
                model="vllm:qwen-2.5-omni-7b",
                temperature=0.9,
                max_tokens=100,
            )

            print(f"\nPrompt: {prompt}")
            print(f"\nCompletion: {response['choices'][0]['text']}")

        except Exception as e:
            print(f"Error: {e}")


def example_load_logs():
    """Example of loading and analyzing logs."""

    print("=" * 60)
    print("Example: Loading and Analyzing Logs")
    print("=" * 60)

    try:
        records = load_logs("./logs/vllm")

        print(f"\nTotal records loaded: {len(records)}")

        if records:
            # Calculate statistics
            successful = [r for r in records if r.error is None]
            failed = [r for r in records if r.error is not None]

            avg_latency = (
                sum(r.latency_ms for r in successful if r.latency_ms) / len(successful)
                if successful
                else 0
            )

            print(f"Successful requests: {len(successful)}")
            print(f"Failed requests: {len(failed)}")
            print(f"Average latency: {avg_latency:.2f}ms")

            # Show last 3 records
            print("\nLast 3 records:")
            for record in records[-3:]:
                print(f"  - {record.timestamp}: {record.request_id}")
                print(f"    Model: {record.model}")
                print(
                    f"    Latency: {record.latency_ms:.2f}ms"
                    if record.latency_ms
                    else "    Error: " + str(record.error)
                )
        else:
            print("No records found. Run the chat completion examples first.")

    except Exception as e:
        print(f"Error loading logs: {e}")


async def main():
    """Main function to run all examples."""

    print("\n" + "=" * 60)
    print("VLLMMiddleware Examples")
    print("=" * 60 + "\n")

    # Run chat completion examples
    await example_chat_completion()

    print("\n")

    # Run completion example
    await example_completions()

    print("\n")

    # Load and analyze logs
    example_load_logs()


if __name__ == "__main__":
    asyncio.run(main())
