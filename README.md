# Knowledge Base Class Project (2025)
 - Artificial Intelligence Master Program
 - An-Najah National University
 - Instructor: Prof. Amjad Hawwash
 - Students: Razi Alsayyed & Shukri Khelfa

**Wikidata-based retrieval-augmented generation for reducing hallucinations in large language models.**

## What This Repository Does

- Runs a Wikidata RAG agent (`kb_project/wikidata_rag_agent.py`) that can call tools for:
  - Wikidata entity search
  - Wikidata property lookup
  - Wikipedia article retrieval
  - Custom read-only SPARQL queries
- Runs a prompt-only baseline (`kb_project/prompt_only_llm.py`) with no retrieval.
- Compares both approaches with hallucination evaluation (`run_benchmark.py`).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/docs) installed and running
- Network access for Wikidata/Wikipedia requests

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Create your environment file.

```bash
cp .env.example .env
```

4. Edit `.env` with your values, then make sure Ollama is available and pull the model(s) you configured.
`.env` is already ignored by git.

```bash
ollama signin
ollama pull qwen2.5:32b-instruct
```

## Environment Variables

All variables can be set in `.env` (auto-loaded by the app).

### Required only for specific features

- `OPENAI_API_KEY`
  - Required only when using `--llm-judge` (OpenAI-based LLM judge).
- `HF_TOKEN`
  - Optional but recommended for downloading HF-hosted models (e.g., Vectara/AIMon dependencies) with better rate limits.

### Optional

- Model choices in `.env`:
  - `LLM_MODEL` (general fallback)
  - `WIKIDATA_RAG_MODEL` (main RAG agent)
  - `PROMPT_ONLY_MODEL` (prompt-only baseline)
  - `RAGTRUTH_MODEL` (RAGTruth evaluator on Ollama)
  - `OPENAI_JUDGE_MODEL` (OpenAI judge model)
  - `RAG_RECURSION_LIMIT` (max LangGraph tool/reasoning steps per RAG run; default `40`)
  - `VECTARA_DEVICE` (device for Vectara evaluator: `auto|cuda|cpu|mps`)
  - `AIMON_DEVICE` (device for AIMon evaluator: `auto|cuda|cpu|mps`)

- `OLLAMA_HOST`
  - Use this if your Ollama server is not local/default (example: `http://your-host:11434`).
- `OLLAMA_PORT`
  - Optional fallback if `OLLAMA_HOST` is not set.
- `OLLAMA_API_KEY`
  - Use this for remote Ollama endpoints that require bearer auth.
  - Requests should include: `Authorization: Bearer <OLLAMA_API_KEY>`.

- LangSmith tracing (optional observability):
  - `LANGSMITH_TRACING=true`
  - `LANGSMITH_API_KEY=...`
  - `LANGSMITH_PROJECT=...`
  - `LANGSMITH_ENDPOINT=https://api.smith.langchain.com`

If LangSmith is not configured, the code runs without tracing.

## Run the Benchmark

From repository root:

```bash
python run_benchmark.py
```

This runs the built-in benchmark cases and saves:

- `benchmark_results.json`
- `benchmark_report.md`

Defaults now prioritize scientific comparability:
- Primary evaluation context mode: `ground_truth` (both models scored against curated references)
- Secondary metric: RAG retrieval-faithfulness against sanitized retrieved evidence
- Benchmark decoding temperature: `0.0` for both compared models

### Common benchmark variants

Run with OpenAI LLM judge (requires `OPENAI_API_KEY`):

```bash
python run_benchmark.py --llm-judge
```

Use RAGTruth dataset instead of built-in cases:

```bash
python run_benchmark.py --use-ragtruth-data
```

Control RAGTruth dataset split/size:

```bash
python run_benchmark.py --use-ragtruth-data --ragtruth-split test --ragtruth-limit 50
```

Disable RAGTruth evaluator:

```bash
python run_benchmark.py --no-ragtruth
```

Disable AIMon evaluator:

```bash
python run_benchmark.py --no-aimon
```

Use legacy combined context scoring mode:

```bash
python run_benchmark.py --eval-context-mode combined
```

Set benchmark decoding temperature:

```bash
python run_benchmark.py --benchmark-temperature 0.0
```

Set hallucination threshold:

```bash
python run_benchmark.py --threshold 0.5
```

## Run the Wikidata RAG Agent Directly (quick check)

```bash
python -c "from kb_project.wikidata_rag_agent import answer_question; print(answer_question('What is the capital of France?', verbose=False))"
```

## Project Layout

- `run_benchmark.py`: main benchmark CLI entry point
- `requirements.txt`: Python dependencies
- `kb_project/wikidata_rag_agent.py`: Wikidata RAG agent
- `kb_project/prompt_only_llm.py`: prompt-only baseline
- `kb_project/tools/`: retrieval/action tools
- `kb_project/benchmark/`: evaluation and reporting modules

## Notes

- Model defaults are defined in `kb_project/settings.py` and can be overridden in `.env`.
- Benchmark execution may download models/data depending on enabled evaluators.
- Runtime logs are written under `logs/`.
