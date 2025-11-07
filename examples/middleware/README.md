# vLLM Middleware

This directory contains middleware for capturing and logging queries and responses from the vLLM server.

## Overview

The `VLLMMiddleware` class provides a comprehensive solution for:
- **Request/Response Logging**: Automatically logs all interactions with the vLLM server
- **Performance Tracking**: Measures and records latency for each request
- **Error Handling**: Captures and logs errors for debugging
- **File Storage**: Stores logs in JSONL format for easy analysis
- **Log Rotation**: Automatically rotates log files based on size
- **Statistics**: Provides usage statistics and metrics

## Files

- `middleware.py`: Main middleware implementation
- `example_middleware_usage.py`: Example scripts demonstrating usage
- `README.md`: This documentation file

## Installation

The middleware requires the following dependencies (already included in `pyproject.toml`):
- `httpx>=0.24.0`
- `pydantic` (transitive dependency)

## Starting the vLLM Server

Before using the middleware, start the vLLM server:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
    --served-model-name vllm:qwen-2.5-omni-7b \
    --port 8081 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

## Usage

### Basic Usage

```python
import asyncio
from middleware import VLLMMiddleware

async def main():
    async with VLLMMiddleware(
        base_url="http://localhost:8081/v1",
        log_dir="./logs/vllm",
        enable_file_logging=True,
    ) as middleware:
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = await middleware.chat_completion(
            messages=messages,
            model="vllm:qwen-2.5-omni-7b",
            temperature=0.7,
            max_tokens=100
        )
        
        print(response)

asyncio.run(main())
```

### Configuration Options

```python
middleware = VLLMMiddleware(
    base_url="http://localhost:8081/v1",  # vLLM server URL
    log_dir="./logs/vllm",                 # Directory for log files
    enable_file_logging=True,              # Enable/disable file logging
    log_format="jsonl",                    # Log format (jsonl or json)
    max_log_size_mb=100,                   # Max log file size before rotation
)
```

### Chat Completion

```python
response = await middleware.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    model="vllm:qwen-2.5-omni-7b",
    temperature=0.7,
    max_tokens=200,
)
```

### Text Completion

```python
response = await middleware.completions(
    prompt="Once upon a time",
    model="vllm:qwen-2.5-omni-7b",
    temperature=0.9,
    max_tokens=100,
)
```

### Function Calling / Tool Use

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = await middleware.chat_completion(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    model="vllm:qwen-2.5-omni-7b",
    tools=tools,
    tool_choice="auto"
)
```

### Getting Statistics

```python
stats = middleware.get_statistics()
print(stats)
# Output:
# {
#     'total_requests': 10,
#     'log_directory': './logs/vllm',
#     'logging_enabled': True,
#     'log_files_count': 2,
#     'total_log_size_mb': 1.5
# }
```

## Log Format

Logs are stored in JSONL format (one JSON object per line). Each record includes:

```json
{
    "timestamp": "2025-10-21T10:30:45.123456",
    "request_id": "req_20251021_103045_1",
    "model": "vllm:qwen-2.5-omni-7b",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "response": {
        "id": "cmpl-123",
        "choices": [...],
        "usage": {...}
    },
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 100
    },
    "latency_ms": 234.56,
    "error": null,
    "metadata": {}
}
```

## Loading and Analyzing Logs

```python
from middleware import load_logs

# Load all logs
records = load_logs("./logs/vllm")

# Load logs for a specific date range
records = load_logs(
    "./logs/vllm",
    start_date="20251021",
    end_date="20251025"
)

# Analyze the records
successful = [r for r in records if r.error is None]
avg_latency = sum(r.latency_ms for r in successful) / len(successful)

print(f"Total requests: {len(records)}")
print(f"Success rate: {len(successful) / len(records) * 100:.1f}%")
print(f"Average latency: {avg_latency:.2f}ms")
```

## Examples

Run the example script to see the middleware in action:

```bash
cd src/train
python example_middleware_usage.py
```

This will:
1. Make several test requests to the vLLM server
2. Log all interactions to `./logs/vllm/`
3. Display statistics and sample responses
4. Load and analyze the logged data

## Use Cases

### Training Data Collection
Collect real-world queries and responses for fine-tuning:

```python
# All interactions are automatically logged
# Later, load logs and filter for high-quality examples
records = load_logs("./logs/vllm")
training_data = [
    {"prompt": r.messages, "completion": r.response}
    for r in records
    if r.error is None and r.latency_ms < 1000
]
```

### Performance Monitoring
Track model performance over time:

```python
records = load_logs("./logs/vllm", start_date="20251020")
latencies = [r.latency_ms for r in records if r.error is None]

print(f"Min latency: {min(latencies):.2f}ms")
print(f"Max latency: {max(latencies):.2f}ms")
print(f"Avg latency: {sum(latencies)/len(latencies):.2f}ms")
```

### Error Analysis
Identify and debug common errors:

```python
records = load_logs("./logs/vllm")
errors = [r for r in records if r.error is not None]

for error in errors:
    print(f"Request {error.request_id}: {error.error}")
```

### A/B Testing
Compare different model parameters:

```python
# Test different temperatures
temps = [0.5, 0.7, 0.9]
messages = [{"role": "user", "content": "Write a creative story."}]

for temp in temps:
    response = await middleware.chat_completion(
        messages=messages,
        temperature=temp,
    )
    # Responses are automatically logged with parameters
```

## Best Practices

1. **Log Directory Management**: Regularly archive or clean up old logs to save disk space
2. **Security**: Logs may contain sensitive information - store them securely
3. **Error Handling**: Always wrap API calls in try-except blocks
4. **Async Context**: Use the middleware with `async with` for proper cleanup
5. **Monitoring**: Regularly check statistics and error rates

## Troubleshooting

### Connection Errors

If you get connection errors, verify the vLLM server is running:

```bash
curl http://localhost:8081/v1/models
```

### Import Errors

Ensure all dependencies are installed:

```bash
pip install httpx pydantic
```

### Log File Issues

Check that the log directory is writable:

```bash
mkdir -p ./logs/vllm
chmod 755 ./logs/vllm
```

## License

MIT License - See LICENSE file in the repository root.
