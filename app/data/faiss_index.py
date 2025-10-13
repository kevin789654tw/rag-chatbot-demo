from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore


class FAISSIndexManager:
    """Manager class for building, saving, and loading FAISS indexes."""

    @staticmethod
    def build(documents, embeddings) -> VectorStore:
        """Build a FAISS index from documents.

        Args:
            documents (List[Document]): List of documents to index.
            embeddings: Embedding model to use.

        Returns:
            FAISS: A FAISS vector store.
        """
        return FAISS.from_documents(documents, embeddings)

    @staticmethod
    def save(vector_db, path: Path, index_name: str) -> None:
        """Save FAISS index to local storage.

        Args:
            vector_db (FAISS): FAISS vector store instance.
            path (str): Path to save the index.
        """
        vector_db.save_local(path, index_name)

    @staticmethod
    def load(path: Path, embeddings, index_name: str = "index") -> VectorStore:
        """Load FAISS index from local storage.

        Args:
            path (Path): Path to the FAISS index directory.
            embeddings: Embedding model used for index.
            index_name (str): Name of the FAISS index file.

        Returns:
            FAISS: A FAISS vector store.
        """
        # NOTE: allow_dangerous_deserialization=True is required for loading pickled metadata.
        # Be cautious: pickle deserialization can execute arbitrary code.
        # Only enable this if the index files come from a trusted source.
        return FAISS.load_local(
            path,
            embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )
