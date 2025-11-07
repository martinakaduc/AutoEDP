"""
Middleware for capturing queries and responses from vLLM server.

This module provides middleware functionality to intercept, log, and store
interactions with the vLLM server for training and analysis purposes.

Command to start vLLM server:
vllm serve Qwen/Qwen2.5-Omni-7B \
    --served-model-name vllm:qwen-2.5-omni-7b \
    --port 8081 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import wraps

import httpx
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryRecord(BaseModel):
    """Model for storing query and response data."""

    timestamp: str = Field(description="ISO format timestamp of the request")
    request_id: Optional[str] = Field(
        default=None, description="Unique identifier for the request"
    )
    model: str = Field(description="Model name used for the query")
    messages: List[Dict[str, Any]] = Field(description="Input messages/prompts")
    response: Optional[Dict[str, Any]] = Field(
        default=None, description="Model response"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Request parameters"
    )
    latency_ms: Optional[float] = Field(
        default=None, description="Response latency in milliseconds"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if request failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class VLLMMiddleware:
    """
    Middleware class for capturing and logging vLLM server interactions.

    This middleware intercepts requests to the vLLM server, logs them,
    and optionally stores them to disk for training and analysis.
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
        """
        Initialize the vLLM middleware.

        Args:
            base_url: Base URL of the vLLM server
            log_dir: Directory to store logs (defaults to ./logs/vllm)
            enable_file_logging: Whether to write logs to files
            log_format: Format for log files ('jsonl' or 'json')
            max_log_size_mb: Maximum size of a single log file in MB
        """
        self.base_url = base_url.rstrip("/")
        self.enable_file_logging = enable_file_logging
        self.log_format = log_format
        self.log_file = log_file
        self.max_log_size_mb = max_log_size_mb

        # Setup log directory
        if log_dir is None:
            self.log_dir = Path("./logs/vllm")
        else:
            self.log_dir = Path(log_dir)

        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Logs will be stored in: {self.log_dir}")

        self.client = httpx.AsyncClient(timeout=300.0)
        self._request_count = 0

    def _get_log_file_path(self) -> Path:
        """Get the current log file path."""
        date_str = datetime.now().strftime("%Y%m%d")
        if self.log_file:
            assert self.log_file.endswith(
                f".{self.log_format}"
            ), "Log file extension must match log format"
            return self.log_dir / self.log_file
        return self.log_dir / f"vllm_queries_{date_str}.{self.log_format}"

    def _rotate_log_if_needed(self, log_path: Path) -> Path:
        """Rotate log file if it exceeds max size."""
        if log_path.exists():
            size_mb = log_path.stat().st_size / (1024 * 1024)
            if size_mb >= self.max_log_size_mb:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_path = self.log_dir / f"vllm_queries_{timestamp}.{self.log_format}"
                logger.info(f"Rotating log file: {log_path} -> {new_path}")
                return new_path
        return log_path

    def _write_log(self, record: QueryRecord) -> None:
        """Write a query record to the log file."""
        if not self.enable_file_logging:
            return

        try:
            log_path = self._get_log_file_path()
            log_path = self._rotate_log_if_needed(log_path)

            with open(log_path, "a", encoding="utf-8") as f:
                if self.log_format == "jsonl":
                    f.write(record.model_dump_json() + "\n")
                else:
                    json.dump(record.model_dump(), f, indent=2)
                    f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def _parse_sse_response(self, sse_text: str) -> Dict[str, Any]:
        """
        Parse Server-Sent Events (SSE) response from vLLM.

        vLLM sometimes returns streaming format even when stream=False.
        This method reconstructs the full response from SSE chunks.

        Args:
            sse_text: Raw SSE response text

        Returns:
            Reconstructed response dictionary
        """
        lines = sse_text.strip().split("\n")
        chunks = []

        for line in lines:
            if line.startswith("data: "):
                data_str = line[6:]  # Remove 'data: ' prefix

                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse SSE chunk: {data_str[:100]}")
                    continue

        if not chunks:
            raise ValueError("No valid SSE chunks found in response")

        # Reconstruct full response from chunks
        first_chunk = chunks[0]

        # For chat completions, reconstruct the message
        if "choices" in first_chunk and first_chunk["choices"]:
            full_content = ""
            role = "assistant"
            completion_tokens = 0

            for chunk in chunks:
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "role" in delta:
                        role = delta["role"]
                    if "content" in delta:
                        full_content += delta["content"]
                        completion_tokens += 1

            # Construct OpenAI-compatible response
            response = {
                "id": first_chunk.get("id", "unknown"),
                "object": "chat.completion",  # Change from chunk to completion
                "created": first_chunk.get("created", int(time.time())),
                "model": first_chunk.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": role, "content": full_content},
                        "finish_reason": chunks[-1]
                        .get("choices", [{}])[0]
                        .get("finish_reason", "stop"),
                    }
                ],
                "usage": chunks[-1].get(
                    "usage",
                    {
                        "prompt_tokens": 0,
                        "completion_tokens": completion_tokens,
                        "total_tokens": completion_tokens,
                    },
                ),
            }

            logger.info(f"Reconstructed response from {len(chunks)} SSE chunks")
            return response

        # Fallback: return first chunk if structure is unexpected
        logger.warning("Unexpected SSE response structure, returning first chunk")
        return first_chunk

    async def chat_completion(
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
        Send a chat completion request to the vLLM server with logging.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response dictionary from the vLLM server
        """
        start_time = time.time()
        self._request_count += 1
        request_id = (
            f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._request_count}"
        )

        # Prepare request payload - start with base params
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools is not None:
            payload["tools"] = tools

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # Initialize record
        record = QueryRecord(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            model=model,
            messages=messages,
            parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "tools": tools is not None,
                "tool_choice": tool_choice,
                **kwargs,
            },
        )

        try:
            # Send request
            logger.info(
                f"Sending request {request_id} to {self.base_url}/chat/completions"
            )
            logger.debug(f"Payload: {payload}")

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            # Log raw response for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")

            # Parse response with better error handling (non-streaming)
            try:
                # Check if response is in SSE (Server-Sent Events) format
                if stream:
                    # vLLM returned streaming format, need to parse SSE
                    logger.warning(
                        "Handling SSE format response as stream=True, parsing..."
                    )
                    response_data = self._parse_sse_response(response.text)
                else:
                    response_data = response.json()

            except json.JSONDecodeError as je:
                error_msg = (
                    f"JSON decode error: {je}. Response text: {response.text[:200]}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency_ms = (time.time() - start_time) * 1000

            # Update record
            record.response = response_data
            record.latency_ms = latency_ms

            logger.info(
                f"Request {request_id} completed in {latency_ms:.2f}ms - "
                f"Tokens: {response_data.get('usage', {})}"
            )

            if stream:
                return StreamingResponse(
                    content=response.aiter_bytes(),
                    media_type="text/event-stream",
                )
            return response_data

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            record.error = error_msg
            record.latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Request {request_id} failed: {error_msg}")
            self._write_log(record)
            raise

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            record.error = error_msg
            record.latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Request {request_id} failed: {error_msg}")
            logger.exception("Full traceback:")  # This will log the full stack trace
            self._write_log(record)
            raise

        finally:
            # Log the record for non-streaming requests (streaming already logged)
            self._write_log(record)

    async def completions(
        self,
        prompt: str,
        model: str = "vllm:qwen-2.5-omni-7b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a completion request to the vLLM server with logging.

        Args:
            prompt: Text prompt for completion
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Response dictionary from the vLLM server
        """
        start_time = time.time()
        self._request_count += 1
        request_id = (
            f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._request_count}"
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        record = QueryRecord(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            parameters={"temperature": temperature, "max_tokens": max_tokens, **kwargs},
        )

        try:
            logger.info(f"Sending completion request {request_id}")
            logger.debug(f"Payload: {payload}")

            response = await self.client.post(
                f"{self.base_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            # Parse response with better error handling
            try:
                # Check if response is in SSE (Server-Sent Events) format
                if stream:
                    # vLLM returned streaming format, need to parse SSE
                    logger.warning(
                        "Received SSE format response as stream=True, parsing..."
                    )
                    response_data = self._parse_sse_response(response.text)
                else:
                    response_data = response.json()

            except json.JSONDecodeError as je:
                error_msg = (
                    f"JSON decode error: {je}. Response text: {response.text[:200]}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            latency_ms = (time.time() - start_time) * 1000

            record.response = response_data
            record.latency_ms = latency_ms

            logger.info(f"Completion {request_id} completed in {latency_ms:.2f}ms")

            return response_data

        except Exception as e:
            error_msg = str(e)
            record.error = error_msg
            record.latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Completion {request_id} failed: {error_msg}")
            logger.exception("Full traceback:")
            raise

        finally:
            # Log the record for non-streaming requests (streaming already logged)
            self._write_log(record)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged requests.

        Returns:
            Dictionary with statistics about the logged requests
        """
        stats = {
            "total_requests": self._request_count,
            "log_directory": str(self.log_dir),
            "logging_enabled": self.enable_file_logging,
        }

        if self.enable_file_logging and self.log_dir.exists():
            log_files = list(self.log_dir.glob(f"*.{self.log_format}"))
            total_size = sum(f.stat().st_size for f in log_files)
            stats["log_files_count"] = len(log_files)
            stats["total_log_size_mb"] = total_size / (1024 * 1024)

        return stats


def load_logs(
    log_dir: Union[str, Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[QueryRecord]:
    """
    Load and parse log files from a directory.

    Args:
        log_dir: Directory containing log files
        start_date: Optional start date filter (YYYYMMDD format)
        end_date: Optional end date filter (YYYYMMDD format)

    Returns:
        List of QueryRecord objects
    """
    log_dir = Path(log_dir)
    records = []

    for log_file in sorted(log_dir.glob("*.jsonl")):
        # Filter by date if specified
        if start_date or end_date:
            date_str = log_file.stem.split("_")[-1]
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        records.append(QueryRecord(**data))
        except Exception as e:
            logger.error(f"Error loading {log_file}: {e}")

    return records


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        """Example usage of the VLLMMiddleware."""

        # Initialize middleware
        async with VLLMMiddleware(
            base_url="http://localhost:8081/v1",
            log_dir="./logs/vllm",
            enable_file_logging=True,
        ) as middleware:

            # Example chat completion
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ]

            try:
                response = await middleware.chat_completion(
                    messages=messages,
                    model="vllm:qwen-2.5-omni-7b",
                    temperature=0.7,
                    max_tokens=100,
                )

                print("Response:", response)
                print("\nStatistics:", middleware.get_statistics())

            except Exception as e:
                print(f"Error: {e}")

    asyncio.run(main())
