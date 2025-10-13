from langchain_community.embeddings import JinaEmbeddings

from app.config import settings


class EmbeddingFactory:
    """Factory for creating embedding model instances."""

    @staticmethod
    def create_jina_embedding_model():
        """Create and initialize a Jina embedding model.

        Returns:
            JinaEmbeddings: An initialized embedding model instance.
        """
        return JinaEmbeddings(
            jina_api_key=settings.JINA_API_KEY, model_name=settings.EMBEDDING_MODEL
        )
