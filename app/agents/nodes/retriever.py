import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Perform vector search and semantic reranking for technical querries.
    """
    
    query = state["current_query"]
    
    with logfire.span("knowledge retrieval"):
        logfire.info(f"searching qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieval {len(raw_results)} candidate from vector DB")
        
        doc_contents = [doc['content'] for doc in raw_results]
        
        with logfire.span("Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking completed. Kept top 5 most relevant chunks.")
        
        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]
        
    return {
        "documents": formatted_docs,
        "status": f"found technical context",
        "plan": state["plan"] + ["Context Retrieved"]
    }