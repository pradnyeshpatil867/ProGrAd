"""
RAGAS regression suite for the LangGraph RAG pipeline.

Runs the golden Q&A set in eval_dataset.py through the real agent (Qdrant
retrieval + FlashRank rerank + Groq synthesis, guardrails bypassed since this
checks RAG quality, not the guardrail gate) and scores the results with
RAGAS. Requires GROQ_API_KEY, GEMINI_API_KEY, QDRANT_API_KEY and
QDRANT_CLUSTER_ENDPOINT to be set (for the agent itself), and the Qdrant
collection to already be populated from DATA/true_data (see README:
"Ingesting Documents").

The RAGAS judge LLM/embeddings run locally via Ollama (not Groq/Gemini) —
judging an 8-question set makes 3x the LLM calls of the agent runs
themselves, and reprocesses the same long retrieved-context blocks, which
was enough on its own to blow through Groq's free-tier daily token quota
for openai/gpt-oss-120b. Requires `ollama serve` running locally with
`qwen2.5:7b-instruct` and `nomic-embed-text` pulled.
"""

import time
import uuid

import pytest
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextRecall
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.agents.graph import rag_agent
from tests.eval_dataset import EVAL_CASES

JUDGE_MODEL = "qwen2.5:7b-instruct"
JUDGE_EMBEDDING_MODEL = "nomic-embed-text"

# Starter thresholds for an 8-question golden set against real Kubernetes
# docs — tighten these as the retrieval/prompting pipeline improves.
MIN_FAITHFULNESS = 0.7
MIN_ANSWER_RELEVANCY = 0.7
MIN_CONTEXT_RECALL = 0.6


def run_agent(question: str) -> dict:
    """Invoke the RAG agent exactly as app.main.query() does, on a fresh thread."""
    initial_state = {
        "message": [{"role": "user", "content": question}],
        "current_query": question,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    return rag_agent.invoke(initial_state, config=config)


@pytest.fixture(scope="session")
def evaluation_result():
    samples = []
    for case in EVAL_CASES:
        output = run_agent(case["question"])
        # Groq's free tier has a low tokens-per-minute cap (see README "Known
        # Limitations") — space out agent calls so 8 back-to-back questions
        # don't trip it before we even get to judging.
        time.sleep(3)
        # retrieve_node prefixes each chunk with "CONTENT: " before handing it
        # to the responder — strip that back off for the ragas context field.
        contexts = [
            doc.removeprefix("CONTENT: ") for doc in output.get("documents", [])
        ]
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=output.get("final_answer") or "",
                retrieved_contexts=contexts or [""],
                reference=case["reference"],
            )
        )

    judge_llm = LangchainLLMWrapper(
        ChatOllama(
            model=JUDGE_MODEL,
            temperature=0,
            num_ctx=8192,  # retrieved contexts can run ~7500 chars across 5 chunks
            num_predict=2048,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=JUDGE_EMBEDDING_MODEL)
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(llm=judge_llm),
            # strictness=1 instead of the default 3: cuts self-consistency
            # sampling to a single generation, which matters here mainly for
            # speed on local CPU/GPU inference rather than API compatibility.
            AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
            LLMContextRecall(llm=judge_llm),
        ],
        # A local 7B model judging 8 questions is slow — keep concurrency low
        # so it doesn't compete with itself for CPU/GPU, with room to retry.
        run_config=RunConfig(max_workers=2, max_retries=3, max_wait=30),
    )

    df = result.to_pandas()
    print("\n" + df[["user_input", "faithfulness", "answer_relevancy", "context_recall"]].to_string())
    print("\nMean scores:")
    for metric in ("faithfulness", "answer_relevancy", "context_recall"):
        print(f"  {metric}: {df[metric].mean():.4f}")

    return df


def test_faithfulness(evaluation_result):
    assert evaluation_result["faithfulness"].mean() >= MIN_FAITHFULNESS


def test_answer_relevancy(evaluation_result):
    assert evaluation_result["answer_relevancy"].mean() >= MIN_ANSWER_RELEVANCY


def test_context_recall(evaluation_result):
    assert evaluation_result["context_recall"].mean() >= MIN_CONTEXT_RECALL
