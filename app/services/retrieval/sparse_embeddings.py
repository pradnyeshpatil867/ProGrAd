import logfire
from fastembed import SparseTextEmbedding
from qdrant_client.http import models

BM25_MODEL_NAME = "Qdrant/bm25"

_model: SparseTextEmbedding | None = None


def _init() -> SparseTextEmbedding:
    global _model
    if _model is None:
        with logfire.span("Loading BM25 sparse embedding model"):
            _model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
    return _model


def embed_query_sparse(query: str) -> models.SparseVector:
    """BM25 sparse vector for a search query (term-frequency weighted, no length norm)."""
    model = _init()
    sparse = next(model.query_embed(query))
    return models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())


def embed_texts_sparse(texts: list[str]) -> list[models.SparseVector]:
    """BM25 sparse vectors for documents (length-normalized term weighting)."""
    model = _init()
    with logfire.span("BM25 sparse embedding", count=len(texts)):
        return [
            models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
            for sparse in model.embed(texts)
        ]
