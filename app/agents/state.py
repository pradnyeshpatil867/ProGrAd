from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    # Using Anotated with operator.add ensuires that messages
    # are appended to the history rather than replaced
    message: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str