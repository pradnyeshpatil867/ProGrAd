import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from app.services.retrieval.embeddings import embed_query
from app.services.retrieval.sparse_embeddings import embed_query_sparse

# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Hybrid search: fuses dense (semantic) and BM25 sparse (lexical) candidates
    with Reciprocal Rank Fusion, so exact keyword matches (error codes, flag
    names, resource kinds) surface even when they're not embedding-similar.
    """
    try:
        dense_vector = embed_query(query)
        sparse_vector = embed_query_sparse(query)

        # Overfetch on each branch so RRF has enough candidates to fuse from
        # before trimming down to `limit`.
        prefetch_limit = limit * 3

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_vector, using=DENSE_VECTOR_NAME, limit=prefetch_limit
                ),
                models.Prefetch(
                    query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=prefetch_limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })

        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []
