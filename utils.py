from typing import Any, Dict, List, Optional

import os
import re
import json
import time
import psutil
import requests
import subprocess
import logging
import signal, platform
from openai._types import NOT_GIVEN

from deepresearch import start_deep_research_pipeline

logging.basicConfig(level=logging.INFO)


def run_server(cmd_string, cwd=None, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        if platform.system() == "Windows":
            return subprocess.Popen(
                cmd_string.split(),
                cwd=cwd,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            return subprocess.Popen(
                cmd_string.split(),
                cwd=cwd,
                env=env,
                preexec_fn=os.setsid,
            )
    except Exception as e:
        print(f"Error starting server: {e}")
        return None


def shutdown_server(process):
    try:
        if platform.system() == "Windows":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        print("Server shutdown successfully.")
    except Exception as e:
        print(f"Error shutting down server: {e}")


def get_model_configs(model_path: str, num_gpus: int = 1) -> Dict[str, Any]:
    if model_path == "Qwen/Qwen2.5-7B-Instruct":
        return {
            "model_path": model_path,
            "model_name": "vllm:qwen-2.5-7b-instruct",
            "max_model_len": 32768,
            "tensor_parallel_size": num_gpus,
            "tool_call_parser": "hermes",
        }
    elif model_path == "Qwen/Qwen2.5-Omni-7B":
        return {
            "model_path": model_path,
            "model_name": "vllm:qwen-2.5-omni-7b",
            "max_model_len": 32768,
            "tensor_parallel_size": num_gpus,
            "tool_call_parser": "hermes",
        }
    elif model_path == "Qwen/Qwen3-4B-Instruct-2507":
        return {
            "model_path": model_path,
            "model_name": "vllm:qwen-3-4b-instruct-2507",
            "max_model_len": 32768,
            "tensor_parallel_size": num_gpus,
            "tool_call_parser": "hermes",
        }
    elif model_path == "ibm-granite/granite-3.1-8b-instruct":
        return {
            "model_path": model_path,
            "model_name": "vllm:granite-3.1-8b-instruct",
            "max_model_len": 32768,
            "tensor_parallel_size": num_gpus,
            "tool_call_parser": "granite",
        }
    else:
        raise ValueError(f"Unsupported model path: {model_path}")


def check_health(url):
    server_ok = False
    while server_ok is False:
        try:
            # Send a GET request to the health check endpoint
            response = requests.get(url)

            # Check if the server is healthy
            if response.status_code == 200:
                server_ok = True
            else:
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            time.sleep(1)
    return server_ok


def initialize_servers(
    model_configs: Dict[str, Any],
    vllm_port: int,
    middleware_port: int,
    deeprs_port: int,
    deeprs_framework: str = "open_deep_research",
    middleware_workers: int = 1,
    log_dir: str = "./logs",
) -> Dict[str, Any]:
    # Start vLLM Server
    logging.info("Starting vLLM server...")
    vllm_pid = None
    vllm_pid = run_server(
        (
            f"vllm serve {model_configs['model_path']} "
            f"--served-model-name {model_configs['model_name']} "
            f"--port {vllm_port} "
            f"--max-model-len {model_configs['max_model_len']} "
            f"--tensor-parallel-size {model_configs['tensor_parallel_size']} "
            f"--enable-auto-tool-choice "
            f"--tool-call-parser {model_configs['tool_call_parser']}"
        )
    )

    check_health(f"http://localhost:{vllm_port}/health")

    # Start Middleware Server
    logging.info("Starting Middleware server...")
    middleware_pid = run_server(
        (
            f"python wrapper_server.py "
            f"--port {middleware_port} "
            f"--vllm-url http://localhost:{vllm_port}/v1 "
            f"--workers {middleware_workers} "
            f"--log-dir {log_dir} "
            "--log-file conversations.jsonl"
        )
    )

    time.sleep(5)  # Give middleware time to start

    # Start Deep Researcher Server
    logging.info("Starting Deep Researcher server...")
    deeprs_pid = start_deep_research_pipeline(
        deeprs_framework=deeprs_framework,
        deeprs_port=deeprs_port,
        middleware_port=middleware_port,
        model_name=model_configs["model_name"],
        run_server=run_server,
    )

    time.sleep(5)  # Give Deep Researcher time to start

    return {
        "vllm_pid": vllm_pid,
        "middleware_pid": middleware_pid,
        "deeprs_pid": deeprs_pid,
    }


def terminate_servers(pids: Dict[str, Any]):
    for server_name, pid in pids.items():
        logging.info(f"Terminating {server_name} server...")
        shutdown_server(pid)


def get_clients(
    vllm_port: int, deeprs_port: int, deeprs_framework: str = "open_deep_research"
):
    from openai import OpenAI

    openai_client = OpenAI(
        base_url=f"http://localhost:{vllm_port}/v1",
        api_key="abcd1234",  # Dummy key since no auth is needed
    )

    if deeprs_framework == "open_deep_research":
        from langgraph_sdk import get_client

        deeprs_client = get_client(url=f"http://localhost:{deeprs_port}")
    else:
        raise ValueError(f"Unsupported Deep Researcher framework: {deeprs_framework}")

    return openai_client, deeprs_client


async def get_response_from_llm(
    msg: str,
    client,
    model,
    system_message,
    msg_history=None,
    response_format=None,
    retry=10,
):
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    for attempt in range(retry):
        try:
            completion = client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                response_format=response_format,
            )
            if response_format is None:
                output = completion.choices[0].message["content"]
            else:
                annotation_response = completion.choices[0].message
                output = annotation_response.parsed
                if output is None:
                    raise ValueError("Failed to parse the model output.")

            new_msg_history = new_msg_history + [
                {"role": "assistant", "content": output}
            ]
            return output, new_msg_history

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(15 * (attempt + 1))  # Wait before retrying

    if response_format is not None and response_format != NOT_GIVEN:
        output = response_format()
    else:
        output = ""
    new_msg_history = new_msg_history + [{"role": "assistant", "content": output}]
    return output, new_msg_history


def process_generated_data(
    data_file: str,
    final_reports: List[str],
    save_path: str,
    tokenizer_name: Optional[str] = None,
):
    conversations = []
    with open(data_file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            conversations.append(json.loads(line))

    # Process conversations
    training_data = []
    for conversation in conversations:
        messages = conversation["messages"]
        response = conversation["response"]["choices"][0]["message"]["content"]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
            training_data.append({"messages": messages})

    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Tokenization check (optional)
        queries = []
        responses = []
        for data in training_data:
            query = tokenizer.apply_chat_template(
                data["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            response = data["messages"][-1]["content"] + tokenizer.eos_token

            queries.append(query)
            responses.append(response)

        training_data = {
            "query": queries,
            "response": responses,
        }

        # Save processed data
        csv_path = save_path.replace(".json", ".csv")
        import pandas as pd

        df = pd.DataFrame(training_data)
        df.to_csv(csv_path, index=False)

    else:
        # Save processed data
        with open(save_path, "w") as f:
            json.dump(training_data, f, indent=2)

    logging.info(f"Processed data saved to {save_path}")
