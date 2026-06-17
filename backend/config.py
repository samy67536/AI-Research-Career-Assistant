# backend/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # App
    app_name: str = "AI Research & Career Assistant"
    debug: bool = True
    port: int = 8000

    # LLM API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    deepseek_api_key: str = ""

    # Default LLM
    llm_provider: str = "gemini"         # claude | openai | gemini | deepseek
    llm_model: str = "gemini-1.5-flash"

    # Security
    secret_key: str = "change-this-secret-key"
    access_token_expire_minutes: int = 1440

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # File Storage
    upload_dir: str = "./data/uploads"
    vector_dir: str = "./data/vectors"
    max_file_size_mb: int = 50

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.vector_dir, exist_ok=True)
os.makedirs("./data", exist_ok=True)
