# vLLM Wrapper Server

OpenAI-compatible API wrapper for vLLM with automatic request/response logging.

## Overview

The wrapper server acts as a proxy between clients and the vLLM server, providing:

- ✅ **OpenAI-compatible API** - Use with OpenAI client libraries
- ✅ **Automatic logging** - All requests/responses captured via middleware
- ✅ **Performance metrics** - Track latency, success rates, token usage
- ✅ **Easy integration** - Drop-in replacement for OpenAI API
- ✅ **Production ready** - CORS, health checks, statistics endpoints

## Architecture

```
┌─────────────────┐
│  Client Code    │
│  (OpenAI SDK)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Wrapper Server  │  ◄─── FastAPI + Middleware
│   Port: 8082    │       (Logging, metrics)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  vLLM Server    │
│   Port: 8081    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Log Files     │
│   (JSONL)       │
└─────────────────┘
```

## Quick Start

### 1. Start vLLM Server

```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
    --served-model-name vllm:qwen-2.5-omni-7b \
    --port 8081 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### 2. Install Dependencies

```bash
pip install -r requirements-wrapper.txt
# or
pip install fastapi uvicorn httpx pydantic
```

### 3. Start Wrapper Server

```bash
python wrapper_server.py --port 8082 --vllm-url http://localhost:8081/v1
```

### 4. Use with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="dummy"  # Not validated by vLLM
)

response = client.chat.completions.create(
    model="vllm:qwen-2.5-omni-7b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## API Endpoints

### OpenAI-Compatible Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat completions (OpenAI-compatible) |
| POST | `/v1/completions` | Text completions (OpenAI-compatible) |
| GET | `/v1/models` | List available models |

### Server Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server info and status |
| GET | `/health` | Health check |
| GET | `/stats` | Server statistics |
| GET | `/logs/summary` | Summary of logged requests |

## Configuration

### Command Line Options

```bash
python wrapper_server.py [OPTIONS]

Options:
  --host TEXT              Host to bind to [default: 0.0.0.0]
  --port INTEGER           Port to run on [default: 8082]
  --vllm-url TEXT          vLLM server URL [default: http://localhost:8081/v1]
  --log-dir TEXT           Log directory [default: ./logs/vllm]
  --disable-logging        Disable request/response logging
  --workers INTEGER        Number of worker processes [default: 1]
  --reload                 Enable auto-reload (development)
```

### Environment Variables

```bash
export VLLM_URL="http://localhost:8081/v1"
export LOG_DIR="./logs/vllm"
export HOST="0.0.0.0"
export PORT="8082"
```

## Usage Examples

### With OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="dummy"
)

# Chat completion
response = client.chat.completions.create(
    model="vllm:qwen-2.5-omni-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### With LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    base_url="http://localhost:8082/v1",
    api_key="dummy",
    model="vllm:qwen-2.5-omni-7b"
)

response = llm.invoke([
    HumanMessage(content="What is machine learning?")
])

print(response.content)
```

### With cURL

```bash
# Chat completion
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm:qwen-2.5-omni-7b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
  }'

# Get statistics
curl http://localhost:8082/stats

# Get logs summary
curl http://localhost:8082/logs/summary
```

## Deployment

### Using Systemd

1. Copy service file:
```bash
sudo cp vllm-wrapper.service /etc/systemd/system/
```

2. Edit paths in the service file:
```bash
sudo nano /etc/systemd/system/vllm-wrapper.service
```

3. Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm-wrapper
sudo systemctl start vllm-wrapper
sudo systemctl status vllm-wrapper
```

4. View logs:
```bash
sudo journalctl -u vllm-wrapper -f
```

### Using Docker Compose

1. Start all services:
```bash
docker-compose up -d
```

2. Check status:
```bash
docker-compose ps
docker-compose logs -f vllm-wrapper
```

3. Stop services:
```bash
docker-compose down
```

### With Monitoring (Prometheus + Grafana)

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access Grafana
# URL: http://localhost:3000
# Username: admin
# Password: admin

# Access Prometheus
# URL: http://localhost:9090
```

## Features

### Automatic Logging

All requests and responses are automatically logged to JSONL files:

```json
{
  "timestamp": "2025-10-21T10:30:45.123456",
  "request_id": "req_20251021_103045_1",
  "model": "vllm:qwen-2.5-omni-7b",
  "messages": [...],
  "response": {...},
  "latency_ms": 234.56,
  "error": null
}
```

### Statistics Endpoint

Get real-time statistics:

```bash
curl http://localhost:8082/stats
```

Response:
```json
{
  "total_requests": 150,
  "total_errors": 2,
  "start_time": "2025-10-21T10:00:00",
  "middleware": {
    "total_requests": 150,
    "log_files_count": 3,
    "total_log_size_mb": 5.2
  }
}
```

### Logs Summary

Get summary of logged requests:

```bash
curl http://localhost:8082/logs/summary
```

Response:
```json
{
  "total_records": 150,
  "successful": 148,
  "failed": 2,
  "success_rate": 98.67,
  "avg_latency_ms": 245.32
}
```

## Advanced Usage

### Batch Requests

```python
from openai import AsyncOpenAI
import asyncio

client = AsyncOpenAI(
    base_url="http://localhost:8082/v1",
    api_key="dummy"
)

async def process_batch():
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]
    
    tasks = [
        client.chat.completions.create(
            model="vllm:qwen-2.5-omni-7b",
            messages=[{"role": "user", "content": q}]
        )
        for q in questions
    ]
    
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in responses]

answers = asyncio.run(process_batch())
```

### Custom Request Headers

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="dummy",
    default_headers={
        "X-User-ID": "user123",
        "X-Session-ID": "session456"
    }
)
```

### Error Handling

```python
from openai import OpenAI, APIError, APIConnectionError

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="dummy",
    timeout=30.0,
    max_retries=3
)

try:
    response = client.chat.completions.create(...)
except APIConnectionError as e:
    print(f"Connection failed: {e}")
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
```

## Monitoring and Debugging

### Check Server Status

```bash
# Health check
curl http://localhost:8082/health

# Server info
curl http://localhost:8082/

# Statistics
curl http://localhost:8082/stats
```

### Analyze Logs

```bash
# View log files
ls -lh ./logs/vllm/

# Analyze with utility
python analyze_logs.py analyze --log-dir ./logs/vllm

# Export for training
python analyze_logs.py export --output training.jsonl

# Find slow requests
python analyze_logs.py slow --threshold 1000
```

### Server Logs

```bash
# If running with systemd
sudo journalctl -u vllm-wrapper -f

# If running with docker-compose
docker-compose logs -f vllm-wrapper

# If running directly
# Check terminal output
```

## Performance Tuning

### Worker Processes

```bash
# Single worker (default)
python wrapper_server.py --workers 1

# Multiple workers for better throughput
python wrapper_server.py --workers 4
```

### Resource Limits

Edit `vllm-wrapper.service`:
```ini
[Service]
LimitNOFILE=65535
MemoryLimit=4G
CPUQuota=200%
```

### Connection Pool

Adjust httpx client settings in `wrapper_server.py`:
```python
self.client = httpx.AsyncClient(
    timeout=300.0,
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100
    )
)
```

## Troubleshooting

### Connection Refused

```bash
# Check if wrapper server is running
curl http://localhost:8082/health

# Check if vLLM server is running
curl http://localhost:8081/v1/models

# Check ports
netstat -tlnp | grep -E '8081|8082'
```

### Import Errors

```bash
# Install missing dependencies
pip install fastapi uvicorn httpx pydantic

# Or use requirements file
pip install -r requirements-wrapper.txt
```

### Permission Denied (Logs)

```bash
# Create log directory
mkdir -p ./logs/vllm

# Set permissions
chmod 755 ./logs/vllm
```

### High Memory Usage

```bash
# Reduce workers
python wrapper_server.py --workers 1

# Set memory limit (systemd)
# Edit service file: MemoryLimit=2G

# Monitor usage
docker stats vllm-wrapper
```

## Best Practices

1. **Production Deployment**:
   - Use systemd or Docker for process management
   - Set appropriate resource limits
   - Enable monitoring and alerting
   - Use multiple workers for better throughput

2. **Security**:
   - Run behind a reverse proxy (nginx, traefik)
   - Enable HTTPS/TLS
   - Implement authentication if needed
   - Restrict network access

3. **Logging**:
   - Regularly archive old logs
   - Monitor disk space
   - Set up log rotation
   - Keep sensitive data secure

4. **Monitoring**:
   - Track success rates
   - Monitor latency trends
   - Set up alerts for errors
   - Review logs periodically

## Example Client Code

See `client_examples.py` for comprehensive examples including:
- OpenAI client usage
- LangChain integration
- Batch processing
- Error handling
- Direct HTTP requests

Run examples:
```bash
python client_examples.py
```

## License

MIT License - See repository root for details.

## Support

For issues or questions:
1. Check server logs: `curl http://localhost:8082/stats`
2. Review log files: `ls ./logs/vllm/`
3. Check vLLM server: `curl http://localhost:8081/health`
4. See examples: `python client_examples.py`

---

**Status**: Production Ready ✅
**Version**: 1.0.0
**Created**: October 21, 2025
