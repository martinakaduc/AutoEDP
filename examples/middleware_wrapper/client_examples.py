"""
Example client usage for the vLLM wrapper server.

This demonstrates how to use the wrapper server with:
1. OpenAI Python client
2. Direct HTTP requests
3. Streaming responses
"""

import asyncio
import httpx
from typing import List, Dict


def example_with_openai_client():
    """
    Example using the official OpenAI Python client.

    This is the recommended way to use the wrapper server.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI package not installed. Install with: pip install openai")
        return

    print("=" * 70)
    print("Example 1: Using OpenAI Client")
    print("=" * 70)

    # Initialize client pointing to the wrapper server
    client = OpenAI(
        base_url="http://localhost:8082/v1",
        api_key="dummy",  # vLLM doesn't require real API key
    )

    # Example 1: Simple chat completion
    print("\n1. Simple chat completion:")
    response = client.chat.completions.create(
        model="vllm:qwen-2.5-omni-7b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")

    # Example 2: Multi-turn conversation
    print("\n2. Multi-turn conversation:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about programming."},
    ]

    response = client.chat.completions.create(
        model="vllm:qwen-2.5-omni-7b", messages=messages, temperature=0.8
    )

    joke = response.choices[0].message.content
    print(f"Joke: {joke}")

    # Continue conversation
    messages.append({"role": "assistant", "content": joke})
    messages.append({"role": "user", "content": "Explain why it's funny."})

    response = client.chat.completions.create(
        model="vllm:qwen-2.5-omni-7b", messages=messages, temperature=0.7
    )

    print(f"Explanation: {response.choices[0].message.content}")

    # Example 3: Text completion
    print("\n3. Text completion:")
    response = client.completions.create(
        model="vllm:qwen-2.5-omni-7b",
        prompt="The future of AI is",
        max_tokens=50,
        temperature=0.9,
    )

    print(f"Completion: {response.choices[0].text}")


def example_with_openai_client_streaming():
    """
    Example using the official OpenAI Python client.

    This is the recommended way to use the wrapper server.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI package not installed. Install with: pip install openai")
        return

    print("=" * 70)
    print("Example 1.1: Using OpenAI Client with Streaming")
    print("=" * 70)

    # Initialize client pointing to the wrapper server
    client = OpenAI(
        base_url="http://localhost:8082/v1",
        api_key="dummy",  # vLLM doesn't require real API key
    )

    # Example 1: Simple chat completion
    print("\n1. Simple chat completion:")
    response = client.chat.completions.create(
        model="vllm:qwen-2.5-omni-7b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    print("Streaming Response:")
    for chunk in response:
        if hasattr(chunk, "choices"):
            delta = chunk.choices[0].delta
            if hasattr(delta, "content"):
                print(delta.content, end="", flush=True)
    print()

    # Example 2: Multi-turn conversation
    print("\n2. Multi-turn conversation:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about programming."},
    ]

    response = client.chat.completions.create(
        model="vllm:qwen-2.5-omni-7b", messages=messages, temperature=0.8, stream=True
    )

    print("Streaming Joke:")
    joke = ""
    for chunk in response:
        if hasattr(chunk, "choices"):
            delta = chunk.choices[0].delta
            if hasattr(delta, "content"):
                print(delta.content, end="", flush=True)
                joke += delta.content
    print()

    # Continue conversation
    messages.append({"role": "assistant", "content": joke})
    messages.append({"role": "user", "content": "Explain why it's funny."})

    response = client.chat.completions.create(
        model="vllm:qwen-2.5-omni-7b", messages=messages, temperature=0.7, stream=True
    )

    print("Streaming Explanation:")
    for chunk in response:
        if hasattr(chunk, "choices"):
            delta = chunk.choices[0].delta
            if hasattr(delta, "content"):
                print(delta.content, end="", flush=True)
    print()

    # Example 3: Text completion
    print("\n3. Text completion:")
    response = client.completions.create(
        model="vllm:qwen-2.5-omni-7b",
        prompt="The future of AI is",
        max_tokens=50,
        temperature=0.9,
        stream=True,
    )
    print("Streaming Completion:")
    for chunk in response:
        if hasattr(chunk, "choices"):
            print(chunk.choices[0].text, end="", flush=True)
    print()


async def example_with_httpx():
    """
    Example using httpx directly for more control.
    """
    print("\n" + "=" * 70)
    print("Example 2: Using httpx Directly")
    print("=" * 70)

    base_url = "http://localhost:8082"

    async with httpx.AsyncClient() as client:
        # Example 1: Check server health
        print("\n1. Health check:")
        response = await client.get(f"{base_url}/health")
        print(f"Status: {response.json()}")

        # Example 2: List models
        print("\n2. List models:")
        response = await client.get(f"{base_url}/v1/models")
        print(f"Models: {response.json()}")

        # Example 3: Chat completion
        print("\n3. Chat completion:")
        payload = {
            "model": "vllm:qwen-2.5-omni-7b",
            "messages": [
                {"role": "user", "content": "What is machine learning in one sentence?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        response = await client.post(f"{base_url}/v1/chat/completions", json=payload)

        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")

        # Example 4: Get server statistics
        print("\n4. Server statistics:")
        response = await client.get(f"{base_url}/stats")
        stats = response.json()
        print(f"Total requests: {stats['total_requests']}")
        print(f"Total errors: {stats['total_errors']}")

        # Example 5: Get logs summary
        print("\n5. Logs summary:")
        response = await client.get(f"{base_url}/logs/summary")
        summary = response.json()
        if "error" not in summary:
            print(f"Total logged: {summary.get('total_records', 0)}")
            print(f"Success rate: {summary.get('success_rate', 0):.1f}%")
            print(f"Avg latency: {summary.get('avg_latency_ms', 0):.2f}ms")


def example_with_langchain():
    """
    Example using LangChain with the wrapper server.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain-openai")
        return

    print("\n" + "=" * 70)
    print("Example 3: Using LangChain")
    print("=" * 70)

    # Initialize LangChain with the wrapper server
    llm = ChatOpenAI(
        base_url="http://localhost:8082/v1",
        api_key="dummy",
        model="vllm:qwen-2.5-omni-7b",
        temperature=0.7,
    )

    # Example 1: Simple invocation
    print("\n1. Simple invocation:")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What are the benefits of renewable energy?"),
    ]

    response = llm.invoke(messages)
    print(f"Response: {response.content}")

    # Example 2: Chain usage
    print("\n2. Using in a chain:")
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that explains concepts simply."),
            ("user", "{topic}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "What is quantum computing?"})
    print(f"Result: {result}")


async def example_batch_requests():
    """
    Example of making batch requests efficiently.
    """
    print("\n" + "=" * 70)
    print("Example 4: Batch Requests")
    print("=" * 70)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("OpenAI package not installed.")
        return

    client = AsyncOpenAI(base_url="http://localhost:8082/v1", api_key="dummy")

    # Create multiple requests
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
        "What is Julia?",
    ]

    print(f"\nProcessing {len(questions)} questions concurrently...")

    # Process all requests concurrently
    tasks = []
    for question in questions:
        task = client.chat.completions.create(
            model="vllm:qwen-2.5-omni-7b",
            messages=[{"role": "user", "content": question}],
            max_tokens=50,
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)

    # Print results
    for question, response in zip(questions, responses):
        answer = response.choices[0].message.content
        print(f"\nQ: {question}")
        print(f"A: {answer}")


def example_error_handling():
    """
    Example of proper error handling.
    """
    try:
        from openai import OpenAI, APIError, APIConnectionError
    except ImportError:
        print("OpenAI package not installed.")
        return

    print("\n" + "=" * 70)
    print("Example 5: Error Handling")
    print("=" * 70)

    client = OpenAI(
        base_url="http://localhost:8082/v1",
        api_key="dummy",
        timeout=30.0,
        max_retries=2,
    )

    try:
        response = client.chat.completions.create(
            model="vllm:qwen-2.5-omni-7b",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=100,
        )
        print(f"Success: {response.choices[0].message.content}")

    except APIConnectionError as e:
        print(f"Connection error: {e}")
        print("Make sure the wrapper server is running on http://localhost:8082")

    except APIError as e:
        print(f"API error: {e}")
        print(f"Status code: {e.status_code}")

    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("vLLM Wrapper Server - Client Examples")
    print("=" * 70)
    print("\nMake sure the wrapper server is running:")
    print("  python wrapper_server.py --port 8082")
    print("=" * 70)

    # Run synchronous examples
    example_with_openai_client()
    example_with_openai_client_streaming()
    example_with_langchain()
    example_error_handling()

    # Run async examples
    asyncio.run(example_with_httpx())
    asyncio.run(example_batch_requests())

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nCheck the logs directory for captured requests:")
    print("  ls -lh ./logs/vllm/")
    print("\nAnalyze logs:")
    print("  python analyze_logs.py analyze --log-dir ./logs/vllm")
    print("=" * 70)


if __name__ == "__main__":
    main()
