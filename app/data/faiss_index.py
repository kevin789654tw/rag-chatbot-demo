from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings.base import BaseEmbeddings
from langchain_core.schema import Document
from langchain_core.vectorstores import VectorStore


class FAISSIndexManager:
    """Manager class for building, saving, and loading FAISS indexes."""

    @staticmethod
    def build(documents: List[Document], embeddings: BaseEmbeddings) -> VectorStore:
        """Build a FAISS index from documents.

        Args:
            documents (List[Document]): List of documents to index.
            embeddings: Embedding model to use.

        Returns:
            FAISS: A FAISS vector store.
        """
        return FAISS.from_documents(documents, embeddings)

    @staticmethod
    def save(vector_index: FAISS, path: Path, index_name: str) -> None:
        """Save FAISS index to local storage.

        Args:
            vector_index (FAISS): FAISS vector store instance.
            path (Path): Path to save the index directory.
            index_name (str): Name of the FAISS index file.

        Returns:
            None
        """
        vector_index.save_local(path, index_name)

    @staticmethod
    def load(
        path: Path, embeddings: BaseEmbeddings, index_name: str = "index"
    ) -> VectorStore:
        """Load FAISS index from local storage.

        Args:
            path (Path): Path to the FAISS index directory.
            embeddings: Embedding model used for index.
            index_name (str, optional): Name of the FAISS index file. Defaults to "index".

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
