"""
Wrapper server for vLLM with OpenAI-compatible API and automatic logging.

This server acts as a proxy between clients and the vLLM server, providing:
- OpenAI-compatible API endpoints
- Automatic request/response logging via middleware
- Additional features like caching, rate limiting, etc.

Usage:
    python wrapper_server.py --port 8082 --vllm-url http://localhost:8081/v1

Then use with OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8082/v1", api_key="dummy")
"""

import argparse
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from middleware import VLLMMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Request/Response Models (OpenAI-compatible)
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "vllm:qwen-2.5-omni-7b"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    model: str = "vllm:qwen-2.5-omni-7b"
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm"


class WrapperServer:
    """
    Wrapper server that provides OpenAI-compatible API with logging.
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8081/v1",
        log_dir: str = "./logs/vllm",
        log_file: str = "queries.jsonl",
        enable_logging: bool = True,
        enable_cors: bool = True,
    ):
        """
        Initialize the wrapper server.

        Args:
            vllm_base_url: Base URL of the vLLM server
            log_dir: Directory for storing logs
            enable_logging: Whether to enable request/response logging
            enable_cors: Whether to enable CORS
        """
        self.vllm_base_url = vllm_base_url
        self.log_dir = log_dir
        self.log_file = log_file
        self.enable_logging = enable_logging

        # Initialize FastAPI app
        self.app = FastAPI(
            title="vLLM Wrapper Server",
            description="OpenAI-compatible API wrapper for vLLM with logging",
            version="1.0.0",
        )

        # Add CORS middleware
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Initialize middleware
        self.middleware = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "start_time": datetime.now().isoformat(),
        }

        # Setup routes
        self._setup_routes()

    async def startup(self):
        """Initialize middleware on startup."""
        self.middleware = VLLMMiddleware(
            base_url=self.vllm_base_url,
            log_dir=self.log_dir,
            log_file=self.log_file,
            enable_file_logging=self.enable_logging,
        )
        logger.info(f"Wrapper server started, proxying to {self.vllm_base_url}")

    async def shutdown(self):
        """Cleanup on shutdown."""
        if self.middleware:
            await self.middleware.close()
        logger.info("Wrapper server stopped")

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.on_event("startup")
        async def startup_event():
            await self.startup()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.shutdown()

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "vLLM Wrapper Server",
                "version": "1.0.0",
                "vllm_url": self.vllm_base_url,
                "logging_enabled": self.enable_logging,
                "uptime_seconds": (
                    datetime.now() - datetime.fromisoformat(self.stats["start_time"])
                ).total_seconds(),
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/v1/models")
        async def list_models():
            """List available models (OpenAI-compatible)."""
            # Return a simple model list
            models = [
                {
                    "id": "vllm:qwen-2.5-omni-7b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm",
                }
            ]
            return {"object": "list", "data": models}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """
            Chat completions endpoint (OpenAI-compatible).

            Accepts OpenAI-style requests and forwards them to vLLM.
            """
            self.stats["total_requests"] += 1

            try:
                # Convert messages to dict format, excluding None values
                messages = []
                for msg in request.messages:
                    message_dict = {"role": msg.role, "content": msg.content}
                    if msg.name is not None:
                        message_dict["name"] = msg.name
                    messages.append(message_dict)

                # Build kwargs with only non-None values
                kwargs = {}
                if request.top_p is not None:
                    kwargs["top_p"] = request.top_p
                if request.stop is not None:
                    kwargs["stop"] = request.stop
                if (
                    request.presence_penalty is not None
                    and request.presence_penalty != 0.0
                ):
                    kwargs["presence_penalty"] = request.presence_penalty
                if (
                    request.frequency_penalty is not None
                    and request.frequency_penalty != 0.0
                ):
                    kwargs["frequency_penalty"] = request.frequency_penalty
                if request.n is not None and request.n != 1:
                    kwargs["n"] = request.n
                if request.logit_bias is not None:
                    kwargs["logit_bias"] = request.logit_bias
                if request.user is not None:
                    kwargs["user"] = request.user
                if request.response_format is not None:
                    kwargs["response_format"] = request.response_format

                # Call vLLM through middleware
                response = await self.middleware.chat_completion(
                    messages=messages,
                    model=request.model,
                    temperature=(
                        request.temperature if request.temperature is not None else 0.7
                    ),
                    max_tokens=request.max_tokens,
                    stream=request.stream if request.stream is not None else False,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    **kwargs,
                )

                return response

            except Exception as e:
                self.stats["total_errors"] += 1
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            """
            Text completions endpoint (OpenAI-compatible).

            Accepts OpenAI-style requests and forwards them to vLLM.
            """
            self.stats["total_requests"] += 1

            try:
                # Build kwargs with only non-None values
                kwargs = {}
                if request.top_p is not None:
                    kwargs["top_p"] = request.top_p
                if request.stop is not None:
                    kwargs["stop"] = request.stop
                if (
                    request.presence_penalty is not None
                    and request.presence_penalty != 0.0
                ):
                    kwargs["presence_penalty"] = request.presence_penalty
                if (
                    request.frequency_penalty is not None
                    and request.frequency_penalty != 0.0
                ):
                    kwargs["frequency_penalty"] = request.frequency_penalty
                if request.n is not None and request.n != 1:
                    kwargs["n"] = request.n
                if request.logit_bias is not None:
                    kwargs["logit_bias"] = request.logit_bias
                if request.user is not None:
                    kwargs["user"] = request.user

                # Call vLLM through middleware
                response = await self.middleware.completions(
                    prompt=request.prompt,
                    model=request.model,
                    temperature=(
                        request.temperature if request.temperature is not None else 0.7
                    ),
                    max_tokens=request.max_tokens,
                    stream=request.stream if request.stream is not None else False,
                    **kwargs,
                )

                return response

            except Exception as e:
                self.stats["total_errors"] += 1
                logger.error(f"Completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            middleware_stats = {}
            if self.middleware:
                middleware_stats = self.middleware.get_statistics()

            return {
                **self.stats,
                "middleware": middleware_stats,
                "current_time": datetime.now().isoformat(),
            }

        @self.app.get("/logs/summary")
        async def logs_summary():
            """Get summary of logged requests."""
            if not self.middleware or not self.enable_logging:
                return {"error": "Logging is not enabled"}

            try:
                from middleware import load_logs
                from pathlib import Path

                records = load_logs(Path(self.log_dir))

                if not records:
                    return {"message": "No logs found", "count": 0}

                successful = [r for r in records if r.error is None]
                failed = [r for r in records if r.error is not None]

                latencies = [r.latency_ms for r in successful if r.latency_ms]
                avg_latency = sum(latencies) / len(latencies) if latencies else 0

                return {
                    "total_records": len(records),
                    "successful": len(successful),
                    "failed": len(failed),
                    "success_rate": (
                        len(successful) / len(records) * 100 if records else 0
                    ),
                    "avg_latency_ms": avg_latency,
                    "log_directory": str(self.log_dir),
                }

            except Exception as e:
                logger.error(f"Error loading logs: {e}")
                return {"error": str(e)}


def create_app(
    vllm_url: str = "http://localhost:8081/v1",
    log_dir: str = "./logs/vllm",
    log_file: str = "queries.jsonl",
    enable_logging: bool = True,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        vllm_url: Base URL of the vLLM server
        log_dir: Directory for storing logs
        enable_logging: Whether to enable request/response logging

    Returns:
        Configured FastAPI application
    """
    server = WrapperServer(
        vllm_base_url=vllm_url,
        log_dir=log_dir,
        log_file=log_file,
        enable_logging=enable_logging,
    )
    return server.app


def main():
    """Main entry point for the wrapper server."""
    parser = argparse.ArgumentParser(
        description="vLLM Wrapper Server with OpenAI-compatible API"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8082, help="Port to run the server on"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8081/v1",
        help="Base URL of the vLLM server",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs/vllm", help="Directory for storing logs"
    )
    parser.add_argument(
        "--log-file", type=str, default="queries.jsonl", help="Log file name"
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="Disable request/response logging",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Create app
    app = create_app(
        vllm_url=args.vllm_url,
        log_dir=args.log_dir,
        log_file=args.log_file,
        enable_logging=not args.disable_logging,
    )

    # Print startup info
    print("=" * 70)
    print("vLLM Wrapper Server")
    print("=" * 70)
    print(f"Server URL:        http://{args.host}:{args.port}")
    print(f"OpenAI API Base:   http://{args.host}:{args.port}/v1")
    print(f"vLLM Server:       {args.vllm_url}")
    print(f"Logging:           {'Enabled' if not args.disable_logging else 'Disabled'}")
    print(f"Log Directory:     {args.log_dir}")
    print("=" * 70)
    print("\nEndpoints:")
    print(f"  GET  /                      - Server info")
    print(f"  GET  /health                - Health check")
    print(f"  GET  /v1/models             - List models")
    print(f"  POST /v1/chat/completions   - Chat completions")
    print(f"  POST /v1/completions        - Text completions")
    print(f"  GET  /stats                 - Server statistics")
    print(f"  GET  /logs/summary          - Logs summary")
    print("=" * 70)
    print("\nUsage with OpenAI client:")
    print("  from openai import OpenAI")
    print(
        f"  client = OpenAI(base_url='http://localhost:{args.port}/v1', api_key='dummy')"
    )
    print("  response = client.chat.completions.create(...)")
    print("=" * 70)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
