# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A guardrailed, agentic RAG API scoped to **Kubernetes, Intel hardware, and enterprise networking** — FastAPI + LangGraph + NeMo Guardrails + Qdrant + Groq, traced with Pydantic Logfire. See [README.md](README.md) for the full architecture diagram and API reference; this file covers what the README doesn't.

## Commands

Setup:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run backend (FastAPI):
```bash
uvicorn app.main:app --reload --port 8000
```

Run frontend (Streamlit, separate terminal):
```bash
streamlit run ui/app.py
```

Ingest documents into Qdrant:
```bash
python -m app.ingestion.processor DATA          # populate from DATA/true_data + DATA/noisy_data
python -m app.ingestion.processor DATA --wipe   # drop + recreate the collection first
```

Run the RAGAS eval suite (needs `ollama serve` running locally with `qwen2.5:7b-instruct` + `nomic-embed-text` pulled — see below):
```bash
pytest tests/test_rag_eval.py -v -s
```
Requires live `GROQ_API_KEY`/`GEMINI_API_KEY`/`QDRANT_API_KEY`/`QDRANT_CLUSTER_ENDPOINT` (for the agent) and an already-ingested Qdrant collection — it queries the real stack, it doesn't mock it. There is no lint config in this repo (no `ruff`/`black`).

## Architecture notes

**Two-gate request flow.** Every `/query` call in [app/main.py](app/main.py) passes through NeMo Guardrails (`guard()`) *before* touching the LangGraph agent. If a rail fires, the LangGraph pipeline is skipped entirely and the rail's own response is returned — the agent's `responder` node is a second, independent line of defense against off-topic content that slips past the gate (see [app/guardrails/rails.py](app/guardrails/rails.py) docstring for why embeddings-only matching is used instead of LLM-generated canonical labels).

**Graph shape** ([app/agents/graph.py](app/agents/graph.py)): `planner` conditionally routes to either `retriever → responder` (technical queries) or straight to `responder` (conversational turns, e.g. greetings or follow-ups answerable from history). The routing signal is a magic string: `planner_node` sets `current_query = "CONVERSATIONAL"` and `route_planner` checks for that exact value — if you touch either function, keep the sentinel in sync.

**State is a flat `TypedDict`** ([app/agents/state.py](app/agents/state.py)); `message` is the only field with reducer semantics (`operator.add`, so it appends across nodes/turns) — every other field is fully overwritten by whatever a node returns. `MemorySaver` checkpoints this state per `thread_id`, which is how multi-turn memory works without an external store.

**Retrieval is three-stage**: [qdrant_service.py](app/services/retrieval/qdrant_service.py) runs a *hybrid* query — dense (semantic) candidates from `embeddings.py` and BM25 sparse (lexical) candidates from [sparse_embeddings.py](app/services/retrieval/sparse_embeddings.py) (local, via `fastembed`'s `Qdrant/bm25`), fused server-side with Reciprocal Rank Fusion (`models.FusionQuery(fusion=models.Fusion.RRF)`) into 15 candidates — then [ranking_service.py](app/services/retrieval/ranking_service.py) reranks with a local FlashRank cross-encoder down to the top 5. The Qdrant collection uses **named vectors** (`DENSE_VECTOR_NAME`/`SPARSE_VECTOR_NAME` in [app/config.py](app/config.py), currently `"dense"`/`"sparse"`) — a collection created before hybrid search shipped uses the old single unnamed vector and is schema-incompatible; it must be recreated with `--wipe` (see README "Ingesting Documents"). Because `search_enterprise_knowledge` swallows all exceptions into `[]` (see below), querying an un-migrated collection fails silently rather than erroring.

**Embeddings have a silent runtime fallback** ([app/services/retrieval/embeddings.py](app/services/retrieval/embeddings.py)): `_init()` probes Gemini (`gemini-embedding-2-preview`, 3072-dim) once at first use and falls back to a local `sentence-transformers` model (768-dim) if the probe fails. This means **the embedding dimension is decided at runtime, not statically** — the Qdrant collection is created with whichever dimension won the probe (`app/ingestion/processor.py::run_universal_ingestion`). Re-ingesting after an embedding-model change (or after a Gemini outage) without `--wipe` will silently produce dimension mismatches. Note the fallback model name is currently misspelled as `all-mpbet-base-v2` (should be `all-mpnet-base-v2`) in two places in that file — it will fail to load if Gemini is ever actually unreachable.

**Ingestion pipeline** ([app/ingestion/processor.py](app/ingestion/processor.py)): parse (format-specific loader under `app/ingestion/loaders/`) → chunk (`chunking/splitter.py`, ~1500 chars, paragraph-boundary splitting, no overlap) → cache to `processed_data/<source_type>/<filename>.json` → embed (dense + BM25 sparse) → upsert to Qdrant as named vectors. `source_type` is inferred from the subfolder name (`true`/`noisy`/`general`) unless passed explicitly — this label is stored as Qdrant payload metadata but isn't currently used to filter retrieval.

**Logfire must be configured before any other app import.** Both [app/main.py](app/main.py) and [app/ingestion/processor.py](app/ingestion/processor.py) call `logfire.configure(...)` at the very top of the file, ahead of importing `app.agents.*` — this is required for spans from those modules to be captured from process start, not an arbitrary ordering choice.

**Eval suite** ([tests/test_rag_eval.py](tests/test_rag_eval.py)): runs the golden Q&A set in `tests/eval_dataset.py` through the real `rag_agent` (still Groq/Gemini/Qdrant) and scores faithfulness/answer-relevancy/context-recall with RAGAS. The judge LLM/embeddings run **locally via Ollama** (`qwen2.5:7b-instruct` + `nomic-embed-text`), not Groq/Gemini — judging makes ~3x the LLM calls of the agent runs and reprocesses the same long context blocks, which alone exhausted Groq's free-tier daily token quota during development. `ragas` also has a known upstream bug — it unconditionally imports a `langchain_community.chat_models.vertexai` module that no longer exists — worked around by a `sys.modules` shim in [tests/conftest.py](tests/conftest.py) that must load before `ragas` is imported anywhere.

**Unused dependencies reserved for future work:** `requirements.txt` still includes `langfuse` and `portkey-ai` with no corresponding code (no Langfuse tracing calls, no Portkey-routed LLM calls — `responder.py` calls `ChatGroq` directly). Don't assume these are wired up just because they're installed.

## Config

All settings load from `.env` via [app/config.py](app/config.py) (`GROQ_API_KEY`, `GROQ_FALLBACK_API_KEY` — declared but not yet used for actual fallback routing, `GEMINI_API_KEY`, `QDRANT_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, `LOGFIRE_TOKEN`). `QDRANT_COLLECTION` is hardcoded to `"enterprise_rag"` in `Settings`, not read from the environment.
