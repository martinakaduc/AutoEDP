from typing import List

import os
import argparse
import asyncio
import logging
import json

import torch
from tqdm import tqdm
from deepresearch import perform_deep_research
from generate_ideas import generate_next_idea, check_idea_novelty
from grpo import run_grpo_training, export_grpo_model
from utils import (
    initialize_servers,
    terminate_servers,
    get_clients,
    get_model_configs,
    process_generated_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def collect_data(
    round_idx: int,
    topics: List[str],
    vllm_client,
    model_name: str,
    deeprs_client,
    deeprs_framework: str = "open_deep_research",
    n_questions: int = 1,
    num_reflections: int = 3,
    skip_novelty_check: bool = False,
    paper_search_engine: str = "semanticscholar",
    data_dir: str = "./data",
):
    base_dir = os.path.join(data_dir, "topics/{topic}")
    result_dir = os.path.join(data_dir, f"round_{round_idx}")
    os.makedirs(result_dir, exist_ok=True)
    n_questions_per_topic = n_questions // len(topics)

    ideas = []
    for topic in topics:
        topic_ideas = []
        topic_base_dir = base_dir.format(topic=topic)
        topic_result_file = f"ideas_{topic}.json"
        # Load previous ideas if not the first round
        n_prev_ideas = 0
        if round_idx > 0:
            pre_result_dir = f"data/round_{round_idx - 1}"
            prev_idea_archive = []
            with open(os.path.join(pre_result_dir, topic_result_file), "r") as f:
                seed_ideas = json.load(f)
            for seed_idea in seed_ideas:
                prev_idea_archive.append(json.dumps(seed_idea))

            topic_ideas.extend(prev_idea_archive)
            n_prev_ideas = len(prev_idea_archive)

        for _ in tqdm(range(n_questions_per_topic), desc=f"Generating {topic}"):
            topic_ideas = await generate_next_idea(
                base_dir=topic_base_dir,
                result_dir=result_dir,
                result_file=topic_result_file,
                client=vllm_client,
                model=model_name,
                prev_idea_archive=topic_ideas,
                num_reflections=num_reflections,
            )

        if not skip_novelty_check:
            topic_ideas = await check_idea_novelty(
                ideas=topic_ideas,
                base_dir=topic_base_dir,
                result_file=topic_result_file,
                result_dir=result_dir,
                client=vllm_client,
                model=model_name,
                engine=paper_search_engine,
            )

        # Only keep new ideas in this round
        ideas.extend(topic_ideas[n_prev_ideas:])

    final_reports = []
    for idea in tqdm(ideas, desc="Performing DeepResearch"):
        if "novel" in idea and not idea["novel"]:
            continue
        research_question = idea["question"]
        final_report = await perform_deep_research(
            deeprs_framework=deeprs_framework,
            deeprs_client=deeprs_client,
            research_question=research_question,
        )
        final_reports.append(final_report)
    return final_reports


def main(args):
    # Setting GPU
    num_gpus = torch.cuda.device_count()
    logging.info(f"Using {num_gpus} GPUs for training.")

    # Get model config
    model_configs = get_model_configs(args.model_path, num_gpus=num_gpus)

    # Start improvement rounds
    for round_idx in range(args.n_rounds):
        logging.info(f"=== Starting Improvement Round {round_idx + 1} ===")

        # Create round directory
        data_dir = os.path.join(args.data_dir, f"round_{round_idx}")
        os.makedirs(data_dir, exist_ok=True)

        # Initialize Servers
        server_pids = initialize_servers(
            vllm_port=args.vllm_port,
            middleware_port=args.middleware_port,
            deeprs_port=args.deeprs_port,
            deeprs_framework=args.deeprs_framework,
            model_configs=model_configs,
            log_dir=data_dir,
        )

        # Get Clients
        vllm_client, deeprs_client = get_clients(
            vllm_port=args.vllm_port,
            deeprs_port=args.deeprs_port,
            deeprs_framework=args.deeprs_framework,
        )

        # Collect Data
        logging.info("Starting data collection...")
        logging.info(f"Starting round {round_idx + 1}/{args.n_rounds}...")
        final_reports = asyncio.run(
            collect_data(
                round_idx=round_idx,
                topics=args.topics,
                vllm_client=vllm_client,
                model_name=model_configs["model_name"],
                deeprs_client=deeprs_client,
                deeprs_framework=args.deeprs_framework,
                n_questions=args.n_questions,
                num_reflections=args.num_reflections,
                skip_novelty_check=args.skip_novelty_check,
                paper_search_engine=args.paper_search_engine,
                data_dir=args.data_dir,
            )
        )

        # Terminate Inference Servers
        terminate_servers(server_pids)

        # Process generated conversations in to trainable data
        data_path = os.path.join(data_dir, "train")
        os.makedirs(data_path, exist_ok=True)
        process_generated_data(
            data_file=os.path.join(data_dir, "conversations.jsonl"),
            final_reports=final_reports,
            save_path=os.path.join(data_path, "data.json"),
            tokenizer_name=model_configs["model_path"],
        )

        # Run GRPO training
        logging.info("Starting GRPO training...")
        model_save_path = os.path.join(args.save_dir, f"round_{round_idx}")
        ckpt_path = os.path.join(model_save_path, "ckpts")
        os.makedirs(ckpt_path, exist_ok=True)

        run_grpo_training(
            model_path=model_configs["model_path"],
            reward_model_path=args.reward_model_path,
            save_path=model_save_path,
            ckpt_path=ckpt_path,
            data_path=os.path.join(data_dir, "train"),
            batch_size=args.batch_size,
            rollout_batch_size=args.batch_size,
            max_epochs=args.n_epochs,
            num_gpus=num_gpus,
        )

        # Export model
        export_grpo_model(
            model_path=model_configs["model_path"],
            lora_path=model_save_path,
            output_path=model_save_path,
        )

        # Update model path for next round
        model_configs["model_path"] = model_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument(
        "--reward_model_path", type=str, help="Path to the reward model file"
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8081,
        help="Port for the vLLM server",
    )
    parser.add_argument(
        "--middleware_port",
        type=int,
        default=8082,
        help="Port for the middleware server",
    )
    parser.add_argument(
        "--deeprs_port",
        type=int,
        default=2024,
        help="Port for the Deep Researcher server",
    )
    parser.add_argument(
        "--deeprs_framework",
        type=str,
        default="open_deep_research",
        help="Framework for Deep Researcher",
        choices=["open_deep_research"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory for data storage",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saves",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=1,
        help="Number of improvement rounds",
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        choices=["seir", "earthquake-prediction"],
        help="List of research topics to start with",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=5,
        help="Number of research questions to generate per round",
    )
    parser.add_argument(
        "--num_reflections",
        type=int,
        default=1,
        help="Number of reflections for question generation",
    )
    parser.add_argument(
        "--skip_novelty_check",
        action="store_true",
        help="Whether to skip novelty check for generated questions",
    )
    parser.add_argument(
        "--paper_search_engine",
        type=str,
        default="semanticscholar",
        help="Paper search engine to use for novelty check",
        choices=["semanticscholar", "openalex"],
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="Number of epochs for GRPO training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training the LLM",
    )
    args = parser.parse_args()
    main(args)
