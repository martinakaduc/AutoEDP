# !/bin/bash

# Start vLLM server for testing
vllm serve Qwen/Qwen2.5-Omni-7B \
    --served-model-name vllm:qwen-2.5-omni-7b \
    --port 8081 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Test middleware with vLLM
python examples/middleware/example_middleware_usage.py

# Test wrapper server
python examples/middleware_wrapper/client_examples.py
python examples/middleware_wrapper/test_wrapper.py
python examples/middleware_wrapper/test_sse.py
python examples/middleware_wrapper/test_structured_output.py