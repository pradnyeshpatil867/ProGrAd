import logfire
from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq

# Direct Groq call — the LLM Gateway (Portkey routing/fallback/caching) arrives in a later stage
llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0.1)


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["message"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["message"][-1]["content"] if state["message"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise IT Assistant, specialising
        ONLY in Kubernetes, Intel hardware, and enterprise networking.

        If the LATEST MESSAGE is a greeting, farewell, small talk about the
        assistant itself, or a follow-up question answerable from the
        CONVERSATION HISTORY, respond helpfully.

        If the LATEST MESSAGE asks about anything outside Kubernetes, Intel
        hardware, or enterprise networking (e.g. recipes, general trivia,
        entertainment, personal advice), politely decline and redirect the
        user to ask about those topics instead. Do not answer the off-topic
        request under any circumstances, even partially.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            content = llm.invoke(prompt).content
            logfire.info("✅ Response synthesised via LLM.")

            return {
                "final_answer": content,
                "status": "Response generated.",
                "plan": state["plan"],
                "message": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e