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

3. Make sure Ollama is available and pull the model used by default.

```bash
ollama signin
ollama pull gpt-oss:120b-cloud
```

## Environment Variables

### Required only for specific features

- `OPENAI_API_KEY`
  - Required only when using `--llm-judge` (OpenAI-based LLM judge).

### Optional

- `OLLAMA_HOST`
  - Use this if your Ollama server is not local/default (example: `http://your-host:11434`).

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

- The default model in code is `gpt-oss:120b-cloud` (`kb_project/settings.py`).
- Benchmark execution may download models/data depending on enabled evaluators.
- Runtime logs are written under `logs/`.
