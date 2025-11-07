from typing import Callable

import os
import logging


def start_deep_research_pipeline(
    deeprs_framework: str,
    deeprs_port: int,
    middleware_port: int,
    model_name: str,
    run_server: Callable,
):
    cwd = os.environ.get("DEEP_RESEARCHER_CWD", "")
    if deeprs_framework == "open_deep_research":
        if not cwd:
            cwd = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "open_deep_research")
            )
        deeprs_pid = run_server(
            (
                "uvx --refresh --from langgraph-cli[inmem] --with-editable . "
                f"--python 3.11 langgraph dev --port {deeprs_port} --allow-blocking"
            ),
            cwd=cwd,
            env_vars={
                "SUMMARIZATION_MODEL": model_name,
                "RESEARCH_MODEL": model_name,
                "COMPRESSION_MODEL": model_name,
                "FINAL_REPORT_MODEL": model_name,
                "SUMMARIZATION_MODEL_BASE_URL": f"http://localhost:{middleware_port}/v1",
                "SUMMARIZATION_MODEL_PROVIDER": "openai",
                "RESEARCH_MODEL_BASE_URL": f"http://localhost:{middleware_port}/v1",
                "RESEARCH_MODEL_PROVIDER": "openai",
                "COMPRESSION_MODEL_BASE_URL": f"http://localhost:{middleware_port}/v1",
                "COMPRESSION_MODEL_PROVIDER": "openai",
                "FINAL_REPORT_MODEL_BASE_URL": f"http://localhost:{middleware_port}/v1",
                "FINAL_REPORT_MODEL_PROVIDER": "openai",
            },
        )
    else:
        raise ValueError(f"Unsupported Deep Researcher framework: {deeprs_framework}")

    return deeprs_pid


async def perform_deep_research(
    deeprs_framework: str,
    deeprs_client,
    research_question: str,
):
    if deeprs_framework == "open_deep_research":
        async for chunk in deeprs_client.runs.stream(
            None,  # Threadless run
            "Deep Researcher",  # Name of assistant. Defined in langgraph.json.
            input={
                "messages": [
                    {
                        "role": "human",
                        "content": research_question,
                    }
                ],
            },
            stream_mode="updates",
        ):
            json_data = chunk.data
            if "final_report_generation" in json_data:
                final_report = json_data["final_report_generation"]["final_report"]
                logging.info("=" * 20)
                logging.info(
                    f"Starting deep research for question: {research_question}"
                )
                logging.info("-" * 20)
                logging.info(f"Final Report: {final_report}")
                logging.info("=" * 20)
                return final_report
    else:
        raise ValueError(f"Unsupported Deep Researcher framework: {deeprs_framework}")
