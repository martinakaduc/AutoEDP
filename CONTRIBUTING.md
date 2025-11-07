# Contributing to AutoEDP

Thank you for your interest in contributing! AutoEDP is a research framework that automates the evolution of deep research pipelines. We welcome bug reports, feature requests, documentation improvements, and code contributions.

By participating in this project, you agree to abide by our Code of Conduct (we follow the spirit of the Contributor Covenant v2.1).

- Contributor Covenant: https://www.contributor-covenant.org/version/2/1/code_of_conduct/

## Ways to contribute

- Report bugs and request features via GitHub Issues
- Improve documentation (README, examples, comments, diagrams)
- Add new research topics (prompts + seed ideas)
- Enhance idea generation, novelty checks, or Deep Research integration
- Improve data processing and training (GRPO) pipeline
- Add support for additional base models in vLLM

## Development setup

Prerequisites:
- Python 3.10+ (Deep Researcher uses Python 3.11 via `uvx` under the hood)
- macOS or Linux recommended; NVIDIA GPUs for vLLM/OpenRLHF
- vLLM >= 0.10.2; Ray is pulled via OpenRLHF
- `uv` for running the LangGraph Deep Researcher (via `uvx`)

Setup steps:

```bash
# Fork the repo on GitHub, then clone your fork
# git clone https://github.com/<you>/AutoEDP.git
cd AutoEDP

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install uv (macOS example); see https://github.com/astral-sh/uv for other OSes
brew install uv
```

Environment variables (used by various parts of the pipeline):

```bash
export RAY_CACHE_DIR="$HOME/.cache/ray"                 # Required for GRPO/Ray
export S2_API_KEY="<your_semantic_scholar_api_key>"     # For novelty checks (semanticscholar)
export OPENALEX_MAIL_ADDRESS="you@example.com"          # For novelty checks (openalex)
export DEEP_RESEARCHER_CWD="/path/to/open_deep_research" # Optional; where Deep Researcher project lives
```

## Running the pipeline locally

Minimal end-to-end run (single round):

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

This will:
- Launch vLLM, the OpenAI-compatible wrapper (with logging), and the Deep Researcher service
- Generate and optionally novelty-check research questions
- Run Deep Researcher to produce final reports
- Convert conversations into training data and run GRPO
- Export the LoRA-merged model

See the README for more details.

## Project structure (quick reference)

- `main.py` — Orchestrates iterative rounds
- `generate_ideas.py` — Idea generation + self-reflection; novelty checks
- `deepresearch.py` — Deep Researcher startup and streaming interface
- `middleware.py` / `wrapper_server.py` — OpenAI-compatible wrapper around vLLM with logging
- `sync_middleware.py` — Sync wrapper for the middleware
- `utils.py` — Server lifecycle, client creation, data processing
- `grpo.py` — GRPO training + LoRA export via OpenRLHF + Ray
- `data/` — Topics, seeds, and per-round outputs
- `examples/` — Docs and scripts for middleware/wrapper usage

## Making changes

### Add/Update supported base models

To add a model that can be served by vLLM, extend `utils.get_model_configs()` with a new mapping:

- `model_path` (Hugging Face id or local path)
- `model_name` (served name, e.g., `vllm:qwen-2.5-7b-instruct`)
- `max_model_len`, `tensor_parallel_size`, and `tool_call_parser`

Ensure the model can load within available GPU memory.

### Add a new research topic

Create a folder under `data/topics/<topic>/` with:
- `prompt.json` — Contains `system` and `topic_description`
- `seed_ideas.json` — Array of seed idea objects

Then include the topic in the `--topics` CLI argument when running `main.py`.

### Novelty check engines

`generate_ideas.search_for_papers()` supports `semanticscholar` and `openalex`.
- Semantic Scholar: set `S2_API_KEY`
- OpenAlex: set `OPENALEX_MAIL_ADDRESS` (optional but recommended to avoid rate limits)

### Deep Researcher service

We integrate the open-source "Open Deep Research" LangGraph workflow.
- AutoEDP starts it via `uvx` and the LangGraph CLI
- Ensure `uv` is installed; set `DEEP_RESEARCHER_CWD` to point to your project or place it at `../open_deep_research`

### Wrapper/middleware development

Run the wrapper directly:

```bash
python wrapper_server.py --port 8082 --vllm-url http://localhost:8081/v1 --log-dir ./logs/vllm
```

OpenAI SDKs can target `http://localhost:8082/v1`. Logs are written to `./logs/vllm/*.jsonl`.
See `examples/middleware_wrapper/WRAPPER_SERVER.md` for endpoints and operations.

## Style and quality

- Follow PEP 8 and add type hints where practical
- Write clear docstrings and comments; keep public APIs stable
- Keep changes focused and incremental
- Documentation updates (README/examples) are part of the definition of done
- Optional (if installed): format with `black` and lint with `ruff`

## Pull request process

1. (Optional) Open an issue to discuss significant changes
2. Create a feature branch (`feature/<short-description>`)
3. Commit with clear messages; Conventional Commits are appreciated
4. Ensure you:
   - Update docs and examples if behavior or usage changes
   - Add or update small tests/examples where relevant
   - Manually sanity-check the main flows you touched (e.g., a short `main.py` run)
5. Open a pull request against `main` and fill out the PR description:
   - What changed and why
   - How it was tested (commands, configs)
   - Any follow-ups or known limitations

## Reporting bugs

When filing a bug report, please include:
- OS, Python version, GPU(s)
- Exact commands and arguments used
- Relevant environment variables (e.g., `RAY_CACHE_DIR`)
- Logs from the wrapper (`./logs/vllm`) and console output
- Expected vs. actual behavior

## Security

Do not include secrets in code or logs. The wrapper accepts a dummy API key for local use. Use environment variables for tokens/keys. If you discover a security vulnerability, please report it privately to the maintainer.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Contact

- GitHub Issues: https://github.com/martinakaduc/AutoEDP/issues
- Maintainer email: nqduc@u.nus.edu
