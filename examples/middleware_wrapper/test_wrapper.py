#!/usr/bin/env python3
"""
Test script for the vLLM wrapper server.

This script performs basic tests to verify the wrapper server is working correctly.
"""

import asyncio
import sys
import httpx
from typing import Dict, Any


class WrapperServerTest:
    """Test suite for the wrapper server."""

    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0

    def print_result(self, test_name: str, passed: bool, message: str = ""):
        """Print test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
        if message:
            print(f"       {message}")

        if passed:
            self.passed += 1
        else:
            self.failed += 1

    async def test_health_check(self) -> bool:
        """Test health check endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                passed = response.status_code == 200
                data = response.json()
                self.print_result(
                    "Health Check", passed, f"Status: {data.get('status', 'unknown')}"
                )
                return passed
        except Exception as e:
            self.print_result("Health Check", False, f"Error: {e}")
            return False

    async def test_server_info(self) -> bool:
        """Test server info endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/")
                passed = response.status_code == 200
                data = response.json()
                self.print_result(
                    "Server Info", passed, f"Version: {data.get('version', 'unknown')}"
                )
                return passed
        except Exception as e:
            self.print_result("Server Info", False, f"Error: {e}")
            return False

    async def test_list_models(self) -> bool:
        """Test list models endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/v1/models")
                passed = response.status_code == 200
                data = response.json()
                models = data.get("data", [])
                self.print_result(
                    "List Models", passed, f"Found {len(models)} model(s)"
                )
                return passed
        except Exception as e:
            self.print_result("List Models", False, f"Error: {e}")
            return False

    async def test_chat_completion(self) -> bool:
        """Test chat completion endpoint."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": "vllm:qwen-2.5-omni-7b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say 'test successful' in exactly 2 words.",
                        }
                    ],
                    "max_tokens": 10,
                    "temperature": 0.1,
                }

                response = await client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload
                )

                passed = response.status_code == 200
                if passed:
                    data = response.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    usage = data.get("usage", {})
                    self.print_result(
                        "Chat Completion",
                        True,
                        f"Response: '{content[:50]}...' | Tokens: {usage.get('total_tokens', 0)}",
                    )
                else:
                    self.print_result(
                        "Chat Completion", False, f"Status: {response.status_code}"
                    )

                return passed
        except Exception as e:
            self.print_result("Chat Completion", False, f"Error: {e}")
            return False

    async def test_text_completion(self) -> bool:
        """Test text completion endpoint."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": "vllm:qwen-2.5-omni-7b",
                    "prompt": "The capital of France is",
                    "max_tokens": 5,
                    "temperature": 0.1,
                }

                response = await client.post(
                    f"{self.base_url}/v1/completions", json=payload
                )

                passed = response.status_code == 200
                if passed:
                    data = response.json()
                    text = data.get("choices", [{}])[0].get("text", "")
                    self.print_result(
                        "Text Completion", True, f"Response: '{text[:50]}...'"
                    )
                else:
                    self.print_result(
                        "Text Completion", False, f"Status: {response.status_code}"
                    )

                return passed
        except Exception as e:
            self.print_result("Text Completion", False, f"Error: {e}")
            return False

    async def test_statistics(self) -> bool:
        """Test statistics endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/stats")
                passed = response.status_code == 200
                data = response.json()
                self.print_result(
                    "Statistics",
                    passed,
                    f"Requests: {data.get('total_requests', 0)}, Errors: {data.get('total_errors', 0)}",
                )
                return passed
        except Exception as e:
            self.print_result("Statistics", False, f"Error: {e}")
            return False

    async def test_logs_summary(self) -> bool:
        """Test logs summary endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/logs/summary")
                passed = response.status_code == 200
                data = response.json()

                if "error" in data:
                    self.print_result("Logs Summary", passed, f"Info: {data['error']}")
                else:
                    self.print_result(
                        "Logs Summary",
                        passed,
                        f"Records: {data.get('total_records', 0)}, Success rate: {data.get('success_rate', 0):.1f}%",
                    )
                return passed
        except Exception as e:
            self.print_result("Logs Summary", False, f"Error: {e}")
            return False

    async def test_concurrent_requests(self) -> bool:
        """Test handling concurrent requests."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Create 5 concurrent requests
                tasks = []
                for i in range(5):
                    payload = {
                        "model": "vllm:qwen-2.5-omni-7b",
                        "messages": [{"role": "user", "content": f"Count to {i+1}"}],
                        "max_tokens": 20,
                    }
                    task = client.post(
                        f"{self.base_url}/v1/chat/completions", json=payload
                    )
                    tasks.append(task)

                responses = await asyncio.gather(*tasks, return_exceptions=True)

                successful = sum(
                    1
                    for r in responses
                    if not isinstance(r, Exception) and r.status_code == 200
                )

                passed = successful == len(tasks)
                self.print_result(
                    "Concurrent Requests",
                    passed,
                    f"{successful}/{len(tasks)} requests successful",
                )
                return passed
        except Exception as e:
            self.print_result("Concurrent Requests", False, f"Error: {e}")
            return False

    async def run_all_tests(self) -> bool:
        """Run all tests."""
        print("=" * 70)
        print("vLLM Wrapper Server Test Suite")
        print("=" * 70)
        print(f"Testing: {self.base_url}")
        print("=" * 70)

        # Basic tests
        await self.test_health_check()
        await self.test_server_info()
        await self.test_list_models()

        # API tests
        await self.test_chat_completion()
        await self.test_text_completion()

        # Management tests
        await self.test_statistics()
        await self.test_logs_summary()

        # Load test
        await self.test_concurrent_requests()

        # Summary
        print("=" * 70)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("=" * 70)

        if self.failed == 0:
            print("✓ All tests passed!")
            return True
        else:
            print(f"✗ {self.failed} test(s) failed")
            return False


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM wrapper server")
    parser.add_argument(
        "--url", default="http://localhost:8082", help="Base URL of the wrapper server"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip completion tests)",
    )

    args = parser.parse_args()

    # Check if server is reachable
    try:
        async with httpx.AsyncClient() as client:
            await client.get(f"{args.url}/health", timeout=5.0)
    except Exception as e:
        print("=" * 70)
        print("ERROR: Cannot connect to wrapper server")
        print("=" * 70)
        print(f"URL: {args.url}")
        print(f"Error: {e}")
        print("\nMake sure the wrapper server is running:")
        print("  python wrapper_server.py --port 8082")
        print("=" * 70)
        sys.exit(1)

    # Run tests
    tester = WrapperServerTest(base_url=args.url)
    success = await tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
