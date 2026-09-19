# ProGrAd RAG — Production Grade Advanced RAG

A guardrailed, agentic Retrieval-Augmented Generation (RAG) API for enterprise IT documentation, scoped to **Kubernetes, Intel hardware, and enterprise networking**. Built with FastAPI, LangGraph, NeMo Guardrails, Qdrant, and Groq.

A user question is (1) screened by NeMo Guardrails for off-topic/jailbreak intent, (2) routed by a LangGraph planner to either a direct conversational answer or a retrieval pipeline (Qdrant hybrid search — dense + BM25 sparse, RRF-fused — → FlashRank cross-encoder rerank), and (3) synthesized into a final answer — all traced end-to-end with Pydantic Logfire.

## Architecture

```
                 ┌─────────────┐
   user query →  │  Streamlit  │  ui/app.py
                 │     UI      │
                 └──────┬──────┘
                        │ POST /query
                        ▼
                 ┌─────────────┐
                 │   FastAPI   │  app/main.py
                 └──────┬──────┘
                        ▼
              ┌──────────────────┐
              │  NeMo Guardrails │  off-topic / jailbreak / dialog gate
              │  (Colang v1.0)   │  app/guardrails/
              └─────────┬────────┘
                blocked │ clean
              ◄─────────┤
                        ▼
              ┌───────────────────────────────────────────────┐                            
              │   LangGraph RAG                               │  app/agents/graph.py
              │      Agent                                    │
              │                                               │
              │   planner ──┬──► retriever ──► responder      │
              │   (intent)  │    (Hybrid +      (Groq LLM)    │
              │             │     FlashRank)         │        │
              │             └────────────────────────┘        │
              │        (conversational path skips retrieval)  │
              └───────────────────────────────────────────────┘
```

- **Guardrail gate** ([app/guardrails](app/guardrails)) — NeMo Guardrails, embedding-similarity intent matching against example utterances, backed by a Groq LLM for anything that doesn't match a known pattern.
- **Planner** ([planner.py](app/agents/nodes/planner.py)) — classifies the query as `CONVERSATIONAL` (answerable from chat history / greeting) or technical, producing a refined search query for the latter.
- **Retriever** ([retriever.py](app/agents/nodes/retriever.py)) — hybrid search against Qdrant: dense (Gemini/sentence-transformers) semantic similarity and BM25 sparse (lexical) candidates fused with Reciprocal Rank Fusion, 15 fused candidates, reranked with FlashRank's local cross-encoder down to the top 5. Hybrid search catches exact keyword matches (resource kinds, flag names, error codes) that pure embedding similarity can miss.
- **Responder** ([responder.py](app/agents/nodes/responder.py)) — synthesizes the final answer from retrieved context (or chat history for conversational turns), scoped to stay on-topic even if something slips past the guardrail gate.
- **Memory** — LangGraph's `MemorySaver` checkpointer keeps per-`thread_id` conversation state across turns.
- **Observability** — Pydantic Logfire instruments every span (guardrail checks, retrieval, reranking, LLM calls) end-to-end.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Orchestration | LangGraph (stateful agent graph, `MemorySaver` checkpointing) |
| Guardrails | NVIDIA NeMo Guardrails (Colang v1.0, embeddings-only dialog rails) |
| LLM | Groq (`openai/gpt-oss-120b` for RAG, `openai/gpt-oss-20b` for the guardrail gate) |
| Embeddings | Google Gemini (`gemini-embedding-2-preview`), with a local `sentence-transformers` fallback |
| Vector DB | Qdrant (hybrid: named dense vector, cosine similarity + named BM25 sparse vector, RRF fusion) |
| Reranking | FlashRank (local ONNX cross-encoder, `ms-marco-MiniLM-L-6-v2`) |
| Document parsing | `pypdf` / `pdfplumber` (PDF), BeautifulSoup (HTML), `unstructured` (DOCX/PPTX) |
| Observability | Pydantic Logfire |

## Project Structure

```
app/
├── main.py                     # FastAPI app: /query, /graph endpoints
├── config.py                   # Settings loaded from .env
├── agents/
│   ├── graph.py                 # LangGraph StateGraph wiring (planner → retriever/responder)
│   ├── state.py                 # AgentState schema
│   └── nodes/
│       ├── planner.py           # Conversational vs. technical intent classification
│       ├── retriever.py         # Qdrant search + FlashRank rerank
│       └── responder.py         # Final answer synthesis
├── guardrails/
│   ├── rails.py                  # NeMo LLMRails singleton + guard() gate
│   └── colang_rules.py           # Colang flows, example utterances, YAML config
├── ingestion/
│   ├── processor.py              # CLI: parse → chunk → save locally → embed → index in Qdrant
│   ├── loaders/                  # Per-format text extraction (pdf, html, text, office)
│   └── chunking/splitter.py      # Paragraph-based chunking (~1500 chars/chunk)
└── services/retrieval/
    ├── embeddings.py             # Gemini embeddings w/ sentence-transformers fallback
    ├── sparse_embeddings.py      # Local BM25 sparse vectors (fastembed, Qdrant/bm25)
    ├── qdrant_service.py         # Hybrid (dense + sparse) query_points search, RRF fusion
    └── ranking_service.py        # FlashRank reranker

ui/app.py                        # Streamlit chat frontend ("Agent OS")
DATA/                             # Raw source documents for ingestion
├── true_data/                    #   genuine Kubernetes/enterprise-IT docs
└── noisy_data/                   #   synthetic distractor corpus (retrieval-precision testing)
processed_data/                   # Ingestion output cache (gitignored, regenerable)
```

## Prerequisites

- Python 3.10+
- A [Qdrant](https://qdrant.io) instance (cloud cluster or self-hosted) with its URL + API key
- A [Groq](https://console.groq.com) API key
- A [Google Gemini](https://ai.google.dev) API key (optional — falls back to a local `sentence-transformers` model if unset/unreachable)
- A [Logfire](https://logfire.pydantic.dev) token (optional — the app runs without it, just without tracing)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_key
GROQ_FALLBACK_API_KEY=
GEMINI_API_KEY=your_gemini_key
QDRANT_API_KEY=your_qdrant_key
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.qdrant.io
LOGFIRE_TOKEN=your_logfire_token
```

## Ingesting Documents

Populate Qdrant from a source directory (expects `true_data/` and `noisy_data/` subfolders, or pass an explicit source-type label for a flat directory):

```bash
python -m app.ingestion.processor DATA
```

Pass `--wipe` to drop and recreate the Qdrant collection first:

```bash
python -m app.ingestion.processor DATA --wipe
```

Each processed file is parsed, chunked (~1500 chars/chunk), embedded (dense + BM25 sparse), upserted into Qdrant, and also cached locally as JSON under `processed_data/<source_type>/<filename>.json`.

**Schema note:** the collection uses named vectors (`dense` + `sparse`) for hybrid search. A collection created before hybrid search was added uses the old single unnamed vector and is incompatible — run with `--wipe` once to recreate it under the new schema, or queries will fail closed (see "Known Limitations").

## Running

**Backend (FastAPI):**

```bash
uvicorn app.main:app --reload --port 8000
```

**Frontend (Streamlit), in a separate terminal:**

```bash
streamlit run ui/app.py
```

Set `BACKEND_URL` (default `http://localhost:8000`) if the API isn't running locally.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/query` | POST | `{"q": "...", "thread_id": "..."}` → runs the guardrail gate + RAG agent, returns the answer, thought process, and retrieved sources |
| `/graph` | GET | Renders the LangGraph agent's workflow as a Mermaid PNG |

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "How do I set up a Kubernetes HPA?", "thread_id": "demo"}'
```

## Guardrails

Off-topic and jailbreak detection ([colang_rules.py](app/guardrails/colang_rules.py)) uses **embeddings-only** intent matching — the raw query is compared by embedding similarity directly against example utterances (no LLM freeform generation involved), which is deliberately more reliable than letting a reasoning-tuned LLM try to complete a few-shot canonical-form pattern. Anything that doesn't match a known category falls through to the LangGraph pipeline, whose responder is itself scoped to refuse off-topic requests as a second line of defense.

## Evaluation

A RAGAS regression suite ([tests/test_rag_eval.py](tests/test_rag_eval.py)) runs a golden set of Kubernetes Q&A pairs ([tests/eval_dataset.py](tests/eval_dataset.py), grounded in `DATA/true_data`) through the full agent and scores the results for **faithfulness**, **answer relevancy**, and **context recall**.

The agent itself still runs on Groq/Gemini/Qdrant as usual, but the RAGAS *judge* (which scores the agent's answers) runs locally via [Ollama](https://ollama.ai) (`qwen2.5:7b-instruct` + `nomic-embed-text`) instead of an API — judging makes ~3x the LLM calls of the agent runs themselves and reprocesses the same long context blocks, which is enough on its own to exhaust Groq's free-tier daily token quota.

```bash
ollama pull qwen2.5:7b-instruct nomic-embed-text   # one-time
ollama serve                                        # if not already running
pytest tests/test_rag_eval.py -v -s
```

Requires `GROQ_API_KEY`, `GEMINI_API_KEY`, `QDRANT_API_KEY`, and `QDRANT_CLUSTER_ENDPOINT` for the agent, plus a running local Ollama instance for the judge. Assumes the target Qdrant collection is already populated (see [Ingesting Documents](#ingesting-documents) above) — the suite queries the existing index, it doesn't (re)ingest it. Runs automatically on push/PR via [.github/workflows/eval.yml](.github/workflows/eval.yml), which installs Ollama and pulls the judge models as part of the job.

## Known Limitations

- `app/services/retrieval/embeddings.py` references the sentence-transformers fallback model as `all-mpbet-base-v2` — this is a typo (should be `all-mpnet-base-v2`) and will fail to load if Gemini is unreachable.
- Groq's free tier caps `openai/gpt-oss-20b` at a low tokens-per-minute limit; guardrail checks can hit `429` rate-limit errors under moderate traffic.
- `requirements.txt` includes `langfuse` (production tracing) and an LLM gateway (`portkey-ai`) with no corresponding code in the repo yet — these are unused dependencies reserved for future work. `ragas` is now wired up (see [Evaluation](#evaluation) below).
- `qdrant_service.search_enterprise_knowledge` catches all exceptions and returns `[]` on any failure (schema mismatch, network error, auth failure), which the responder then answers around using its own general knowledge instead of retrieved context — a query against an un-migrated collection (see "Schema note" above) or an unreachable Qdrant cluster fails *silently* into an ungrounded answer rather than an error.
