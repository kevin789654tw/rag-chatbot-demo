"""Configuration management for the ChatBot RAG demo."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)


class Settings(BaseSettings):
    """Application settings."""

    # OpenRouter & Query Model
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str
    QUERY_MODEL: str
    QUERY_MODEL_TEMPERATURE: float = Field(default=0)

    # JINA & Embedding Model
    JINA_API_KEY: str
    EMBEDDING_MODEL: str

    # FAISS Index
    FAISS_INDEX_PATH: Path
    FAISS_INDEX_NAME: str = Field(default="faiss_index")

    # Tokenizer
    TOKENIZER_PATH: Path
    PRETRAINED_MODEL: str
    PRETRAINED_MODEL_HASH: str

    # Chunking (by token)
    CHUNK_SIZE: int = Field(default=500)
    CHUNK_OVERLAP: int = Field(default=50)

    # RAG
    RETRIEVAL_TOP_K: int = Field(default=3)
    SIMILARITY_THRESHOLD: float = Field(default=0.8)

    # Memory
    MEMORY_WINDOW: int = Field(default=5)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def tokenizer_path(self) -> Path:
        """Return the path where the tokenizer for the specific model is stored.

        Returns:
            Path: The full directory path to the tokenizer corresponding to the
            selected pretrained model.
        """
        return self.TOKENIZER_PATH / self.PRETRAINED_MODEL


# Global settings instance
settings = Settings()
