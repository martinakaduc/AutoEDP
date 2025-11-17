# AutoEDP — Automatic Evolution of Deep Research Pipelines

AutoEDP is a framework for automatically evolving a deep research pipeline. It iterates through idea generation, novelty checking against the literature, deep-research execution, data capture, and GRPO fine-tuning to continuously improve an LLM-based research agent.

## Why AutoEDP

- Automates an end-to-end research loop: from ideation to literature vetting, structured investigation, and model improvement.
- Uses a modern stack: vLLM for fast inference, an OpenAI-compatible wrapper with logging, LangGraph-based "Deep Researcher" for orchestration, and OpenRLHF + Ray for GRPO training.
- Designed for iterative rounds to steadily refine both research questions and model behavior.

## Features

- Idea generation with self-reflection to improve question quality.
- Novelty checking via Semantic Scholar or OpenAlex to filter out known ideas.
- Pluggable Deep Research engine (Open Deep Research via LangGraph) for evidence collection and final report synthesis.
- OpenAI-compatible wrapper server that logs all prompts/responses (JSONL) for training data.
- Data processing into prompt/response pairs and CSV for tokenized workflows.
- GRPO training and LoRA export using OpenRLHF + Ray.
- Multi-round improvement loop driven by a single CLI (`main.py`).

## Architecture

```
┌─────────────────────────┐
│ Topics + Seed Ideas     │  data/topics/*/prompt.json, seed_ideas.json
└──────────┬──────────────┘
           │  (generate_ideas.py)
           ▼
┌─────────────────────────┐         ┌───────────────────────┐
│ Idea Generation +       │  ─────▶ │ Novelty Check         │  (Semantic Scholar / OpenAlex)
│ Self-Reflection         │         └───────────────────────┘
└──────────┬──────────────┘
           │  (deepresearch.py)
           ▼
┌────────────────────────────┐         ┌───────────────────────┐
│ Deep Researcher (LangGraph)│  ◀────  │ Middleware Wrapper    │  OpenAI API + logging
│ final report generation    │  ─────▶ │ (wrapper_server.py)   │  logs/vllm/*.jsonl
└──────────┬─────────────────┘         └───────────────────────┘
           │  (utils.process_generated_data)
           ▼
┌─────────────────────────┐
│ Train Data Builder      │  data/round_*/train/{data.json,csv}
└──────────┬──────────────┘
           │  (grpo.py, OpenRLHF + Ray)
           ▼
┌─────────────────────────┐
│ GRPO Fine-Tuning        │  saves/round_*/
│ + LoRA Export           │
└─────────────────────────┘
```

## Repository layout

- `main.py` — Orchestrates iterative rounds: servers → data collection → processing → GRPO → export.
- `generate_ideas.py` — Idea generation with multi-round self-reflection; novelty check via Semantic Scholar or OpenAlex.
- `deepresearch.py` — Starts and streams runs from the Deep Researcher (LangGraph) server.
- `middleware.py` / `wrapper_server.py` — OpenAI-compatible proxy to vLLM with request/response logging (JSONL) and stats.
- `sync_middleware.py` — Synchronous wrapper for the middleware.
- `utils.py` — Server lifecycle helpers (vLLM, wrapper, Deep Researcher), client creation, data processing utilities.
- `grpo.py` — GRPO training and LoRA merge using OpenRLHF + Ray.
- `data/` — Topics, seeds, and per-round outputs.
- `examples/` — Usage docs and wrappers; see `examples/middleware_wrapper/WRAPPER_SERVER.md` for wrapper details.

## Requirements

- Python 3.10+ (Deep Researcher invokes Python 3.11 via `uvx`)  
- macOS or Linux recommended; GPU(s) with CUDA for vLLM/OpenRLHF
- pip dependencies:
  - See `requirements.txt` (includes fastapi, uvicorn, httpx, pydantic, backoff, vllm, langgraph-sdk, OpenRLHF from Git)
- Additional tools:
  - vLLM >= 0.10.2
  - Ray (installed transitively by OpenRLHF) and system support to start/stop Ray
  - uv (for `uvx` command used to run LangGraph Deep Researcher)  
    - macOS: `brew install uv`  
    - Or see https://github.com/astral-sh/uv

Environment variables:
- `RAY_CACHE_DIR` — Required for GRPO. Path to a writable directory for Ray object spilling.
- `S2_API_KEY` — Required if using `--paper_search_engine semanticscholar` for novelty checks.
- `OPENALEX_MAIL_ADDRESS` — Optional; used if `--paper_search_engine openalex`.
- `DEEP_RESEARCHER_CWD` — Optional; path to your Open Deep Research project. If not set, AutoEDP looks for `../open_deep_research/` relative to this repo.

## Installation

```bash
# clone your project
# cd AutoEDP

# (Recommended) create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# install Python dependencies
pip install -r requirements.txt

# install uv for running Deep Researcher via langgraph-cli
brew install uv   # macOS; or see uv docs for other OSes
```

## Quickstart

1) Prepare environment variables

```bash
export RAY_CACHE_DIR="$HOME/.cache/ray"
# One of the two engines for novelty check
export S2_API_KEY="<your_semantic_scholar_api_key>"        # if using semanticscholar
export OPENALEX_MAIL_ADDRESS="you@example.com"             # if using openalex
# Optional: where your Open Deep Research project lives
export DEEP_RESEARCHER_CWD="/path/to/open_deep_research"
```

2) Run one improvement round end-to-end

```bash
python main.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --reward_model_path <hf_or_local_reward_model> \
  --topics seir earthquake-prediction \
  --n_rounds 1 \
  --n_questions 4 \
  --num_reflections 3 \
  --paper_search_engine semanticscholar \
  --batch_size 1 \
  --n_epochs 3
```

What this does:
- Starts vLLM, the OpenAI-compatible wrapper with logging, and the Deep Researcher service.
- Generates research questions for each topic with self-reflection.
- Optionally checks novelty against Semantic Scholar or OpenAlex.
- Runs Deep Researcher to produce a final report per idea.
- Converts logged conversations into training data and kicks off GRPO.
- Exports the LoRA-merged model; uses it as the starting point for the next round if `--n_rounds > 1`.

## Data & outputs

- Per round outputs: `data/round_{i}/`
  - `conversations.jsonl` — Logged prompt/response pairs (wrapper logs copied here)
  - `ideas_*.json` — Generated ideas per topic (with novelty flags)
  - `train/data.json` — Messages for OpenRLHF; if a tokenizer is known, also `train/data.csv` with `query,response`.
- Model checkpoints and exports: `saves/round_{i}/`

## CLI reference (main.py)

Key arguments:
- `--model_path` — Base HF model to serve in vLLM. Supported out of the box:
  - `Qwen/Qwen2.5-7B-Instruct` → served as `vllm:qwen-2.5-7b-instruct`
  - `Qwen/Qwen2.5-Omni-7B` → `vllm:qwen-2.5-omni-7b`
  - `Qwen/Qwen3-4B-Instruct-2507` → `vllm:qwen-3-4b-instruct-2507`
  - `ibm-granite/granite-3.1-8b-instruct` → `vllm:granite-3.1-8b-instruct`
  - To add another model, extend `utils.get_model_configs`.
- `--reward_model_path` — Reward model for OpenRLHF GRPO.
- `--topics` — One or more of: `seir`, `earthquake-prediction`.
- `--n_questions` — Total questions per round (split evenly across topics).
- `--num_reflections` — Refinement iterations during idea generation.
- `--skip_novelty_check` — Skip literature novelty gating.
- `--paper_search_engine` — `semanticscholar` (default) or `openalex`.
- `--n_rounds` — Number of iterative improvement rounds.
- Ports: `--vllm_port` (default 8081), `--middleware_port` (8082), `--deeprs_port` (2024).
- Paths: `--data_dir` (default `./data`), `--save_dir` (default `./saves`).
- Training: `--batch_size` (per-device micro batch), `--n_epochs`.

## Wrapper server & middleware

- The wrapper is a FastAPI server providing OpenAI-compatible endpoints and JSONL logging.
- See `examples/middleware_wrapper/WRAPPER_SERVER.md` for detailed usage, endpoints, and deployment.
- You can run it standalone:

```bash
python wrapper_server.py --port 8082 --vllm-url http://localhost:8081/v1 --log-dir ./logs/vllm
```

Then point the OpenAI client at `http://localhost:8082/v1`.

## Notes on Deep Researcher

- AutoEDP integrates the open-source "Open Deep Research" workflow via LangGraph CLI.
- The service is launched automatically using `uvx`:
  - Requires `uv` installed, and either:
    - `DEEP_RESEARCHER_CWD` set to the project directory, or
    - a sibling folder `../open_deep_research` available.
- Models for summarization/research/compression/final report all point to the same vLLM-served base model via the wrapper.

## Troubleshooting

- Missing `RAY_CACHE_DIR`: set it to a writable path (e.g., `~/.cache/ray`).
- Novelty check fails with Semantic Scholar: set `S2_API_KEY`.
- OpenAlex rate limits: set `OPENALEX_MAIL_ADDRESS` to identify your requests.
- Deep Researcher server doesn’t start: ensure `uv` is installed and `DEEP_RESEARCHER_CWD` is correct.
- vLLM OOM: reduce `--max-model-len`, use a smaller model, or fewer GPUs via `utils.get_model_configs`.

## Development

Helpful scripts and docs are under `examples/`:
- Middleware quickstart: `examples/middleware/quickstart.py`
- Wrapper docs: `examples/middleware_wrapper/WRAPPER_SERVER.md`
- Middleware README: `examples/middleware/README.md`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{autoedp2025,
  author = {Duc Q. Nguyen},
  title = {AutoEDP: Automatic Evolution of Deep Research Pipelines},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/martinakaduc/AutoEDP},
}
```

## License

MIT License. See `LICENSE`.

## Acknowledgements

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for GRPO training primitives
- [vLLM](https://github.com/vllm-project/vllm) for fast inference and OpenAI compatibility
- [LangGraph](https://github.com/langchain-ai/langgraph) & [Open Deep Research](https://github.com/langchain-ai/open_deep_research) for the research orchestration workflow
- [AI Scientist](https://github.com/SakanaAI/AI-Scientist) for inspiration on research question generation and refinement
- [DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench) for benchmarking dataset

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

For detailed contribution guidelines, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the author directly.
Email: nqduc@u.nus.edu

