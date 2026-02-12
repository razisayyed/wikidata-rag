**Requirements & Setup Guide**

- **Project root**: `./` (run commands from project root)
- **Supported OS**: macOS (tested), Linux (likely compatible)
- **Recommended Python**: 3.10+ (3.11 is fine)

**Create & activate a virtual environment**
- Use `venv` or `conda`. Example (zsh):

```bash
# venv (recommended)
python -m venv .venv
source .venv/bin/activate

# or conda
conda create -n kbproject python=3.11 -y
conda activate kbproject
```

**1) System / external runtime: Ollama**
- Install Ollama (local LLM host) following their official instructions: https://ollama.com/docs
- Start Ollama daemon (on macOS typically runs automatically after install). If you need to run the server manually:

```bash
# If you installed Ollama, make sure the daemon is running
# Example (macOS/homebrew):
brew services start ollama  # optional depending on installation method
# Or just run ollama normally (it will spawn a local listener):
ollama models
```

- Register at ollama.com. You need to have a free account in order to use the cloud model we are using in our code.

- Login at your PC.

```bash
ollama signin
```

- Pull the required Ollama models used in this project. Run these with your Ollama CLI:

```bash
ollama pull gpt-oss:120b-cloud
```

**2) Python package dependencies**
- Install dependencies from the repository `requirements.txt` file.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- If you have a CUDA-enabled Linux GPU, install a CUDA-capable wheel using the official PyTorch instructions: https://pytorch.org/get-started/locally/

**3) Optional / external service credentials**
- If you want to enable the OpenAI-based LLM Judge (`--llm-judge`), set:
- This is optional to use openai to evaluate our model. We will not mention this in the paper.

```bash
export OPENAI_API_KEY="sk-..."
```

- LangSmith (tracing/observability): register at https://www.langsmith.com (or the LangSmith documentation) and create a project; you will receive an API key. Export it as environment variables used by the `langsmith` client used in the code.

Recommended environment variables for LangSmith (you will find them in the create project page):

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=lsv2_pt_...
export LANGSMITH_PROJECT=pr-...
```

- If you plan to use Ollama on a remote host or change defaults, set `OLLAMA_HOST`/`OLLAMA_PORT` accordingly (the project uses local defaults). Example (if remote Ollama API is at custom host):

```bash
export OLLAMA_HOST="http://your-ollama-host:11434"
```

**5) LangSmith - quick steps**
- Sign up / sign in at LangSmith
- Create a project for this repo (e.g., `KB-Project`) in the LangSmith UI
- Create or copy your LangSmith API key
- Export `LANGSMITH_API_KEY` and optionally set `LANGSMITH_PROJECT` as shown above

LangSmith client integration in the repo is optional — the code will fall back to a no-op tracer if `langsmith` isn't available.

**7) Run the benchmark / demos**
- Basic benchmark runner (saves `benchmark_results.json` and `benchmark_report.md`):

```bash
python run_benchmark.py --ragtruth
```

**8) Troubleshooting highlights**
- `ModuleNotFoundError: bs4`: install `beautifulsoup4` (see install step above)
- `langgraph` not installed: prompt-only agent will fall back to `ChatOllama` or raw invocation; install `langgraph` to get the full agent behavior
- `ollama` CLI/daemon not running: `ollama` commands will fail — ensure Ollama is installed and the daemon is available

**9) Results**

- Some results will be outputed on the console while the script is running.
- Two files will be created at the root of the project (containing all results):
  - benchmark_report.md
  - benchmark_results.json
- langsmith service will give you great insights on how actually our model (Grounded Model), Prompt Only model, and evaluator models work.

