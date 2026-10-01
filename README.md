# Autonomous Multi-Agent Bargain Spotter

A multi-agent deal hunter. It reads live deal feeds, estimates what each product
is really worth with an ensemble of pricing models, and sends a push notification
when something is selling well below its value.

## How it works

```
Scanner Agent ──▶ Ensemble Agent ──────────────────────────▶ Planning Agent
DealNews RSS      Preprocessor (Llama 3.2 via Ollama)           best discount > $50
+ GPT picks the     ├─ Specialist: fine-tuned Llama on Modal  10%   ├─ Pushover alert
  5 best-described  ├─ Frontier: GPT-5.1 + RAG over Chroma    80%   └─ saved to memory.json
  deals             └─ Neural Network: local PyTorch model    10%
```

1. **Scanner Agent** pulls the DealNews RSS feeds and asks `gpt-5-mini` to pick the
   five deals with the clearest descriptions and prices.
2. **Ensemble Agent** rewrites each description into a standard format with a
   local Llama 3.2 (via Ollama), then blends three price estimates:
   - **Specialist Agent**: a fine-tuned Llama 3.2 3B served on Modal (10%)
   - **Frontier Agent**: GPT-5.1 prompted with the five most similar products
     retrieved from a Chroma vector store (80%)
   - **Neural Network Agent**: a residual deep neural network running locally (10%)
3. **Planning Agent** keeps the deal with the biggest discount. If it is more than
   $50 below the estimate, the **Messaging Agent** sends a Pushover alert and the
   deal is saved to `memory.json` so it is not surfaced again.

The Gradio UI shows the deals found so far, a live agent log and a 3D t-SNE map of
the products in the vector store. A new scan runs every 5 minutes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/).

```bash
uv sync
cp .env.example .env          # then fill in your keys
```

**Local preprocessor**

```bash
ollama serve
ollama pull llama3.2
```

**Specialist model on Modal**

```bash
uv run modal token new
# create a Modal secret named "huggingface-secret" containing HF_TOKEN
uv run modal deploy pricer_service.py
```

**Neural network weights:** place `deep_neural_network.pth` in the repo root.

**Vector store:** build `products_vectorstore/` from a Hugging Face dataset of
priced products (fields: `title`, `category`, `price`, `full` / `summary`):

```bash
uv run python populate_vectorstore.py --dataset <hf-user>/<dataset>
```

## Running

```bash
uv run python price_is_right.py
```

The UI opens at http://127.0.0.1:7860. On Windows, set `PYTHONIOENCODING=utf-8`
if the console fails to print the coloured logs.

## Also included

- `agents/autonomous_planning_agent.py`: an alternative planner where GPT-5.1
  drives the workflow itself through tool calls (scan, estimate, notify), and
  Claude writes the alert text.
- `agents/evaluator.py`: a test harness that scores any price predictor against
  labelled items and charts the error.
- `MultiAgent.ipynb` and `results.ipynb`: UI prototypes and a model comparison
  chart. Install their extra dependencies with `uv sync --group notebooks`.
