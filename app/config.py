import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "enterprise_rag"
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "openai/gpt-oss-120b"
    GRPQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

settings = Settings()
    