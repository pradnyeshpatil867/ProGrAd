import sys
import types

# ragas (as of 0.4.3) unconditionally imports
# `langchain_community.chat_models.vertexai.ChatVertexAI` at module load time,
# but that submodule was removed from langchain-community — VertexAI chat
# support now lives in the standalone `langchain-google-vertexai` package.
# This crashes `import ragas` outright for anyone not using Vertex AI
# (see https://github.com/vibrantlabsai/ragas/issues/2745). Since this repo
# already depends on langchain-google-vertexai (as a NeMo Guardrails import
# shim), we register it under the old module path before ragas is imported.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    from langchain_google_vertexai import ChatVertexAI

    shim = types.ModuleType("langchain_community.chat_models.vertexai")
    shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = shim
