import os
from utils.secrets_manager import load_secrets_into_env

load_secrets_into_env()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LANCEDB_PATH = os.getenv("s3://droogbucket/lancedb")
