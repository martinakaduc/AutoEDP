"""
Synchronous wrapper for VLLMMiddleware.

This module provides a synchronous interface to the VLLMMiddleware
for easier integration in non-async codebases.
"""

import asyncio
from functools import wraps
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from middleware import VLLMMiddleware as AsyncVLLMMiddleware


class VLLMMiddleware:
    """
    Synchronous wrapper for the async VLLMMiddleware.

    This class provides the same functionality as the async version
    but with synchronous methods for easier integration.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        log_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[str] = None,
        enable_file_logging: bool = True,
        log_format: str = "jsonl",
        max_log_size_mb: int = 100,
    ):
        """Initialize the synchronous middleware wrapper."""
        self._middleware = AsyncVLLMMiddleware(
            base_url=base_url,
            log_dir=log_dir,
            log_file=log_file,
            enable_file_logging=enable_file_logging,
            log_format=log_format,
            max_log_size_mb=max_log_size_mb,
        )
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "vllm:qwen-2.5-omni-7b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request (synchronous).

        See AsyncVLLMMiddleware.chat_completion for parameter details.
        """
        return self._run_async(
            self._middleware.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        )

    def completions(
        self,
        prompt: str,
        model: str = "vllm:qwen-2.5-omni-7b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a completion request (synchronous).

        See AsyncVLLMMiddleware.completions for parameter details.
        """
        return self._run_async(
            self._middleware.completions(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return self._middleware.get_statistics()

    def close(self):
        """Close the middleware and cleanup resources."""
        self._run_async(self._middleware.close())
        if self._loop and not self._loop.is_closed():
            self._loop.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Simple synchronous usage
    with VLLMMiddleware(
        base_url="http://localhost:8081/v1",
        log_dir="./logs/vllm",
    ) as middleware:

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"},
        ]

        try:
            response = middleware.chat_completion(
                messages=messages, temperature=0.7, max_tokens=200
            )

            print("Response:", response["choices"][0]["message"]["content"])
            print("\nStatistics:", middleware.get_statistics())

        except Exception as e:
            print(f"Error: {e}")
