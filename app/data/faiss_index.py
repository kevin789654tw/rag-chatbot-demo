from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS


class FAISSIndexManager:
    """Manager class for building, saving, and loading FAISS indexes."""

    @staticmethod
    def build(
        documents: List[Document], embeddings: Embeddings, batch_size: int = 512
    ) -> FAISS:
        """Build a FAISS index from documents in batches to avoid API limits.

        Args:
            documents (List[Document]): List of documents to index.
            embeddings (Embeddings): Embedding model to use.
            batch_size (int, optional): Max number of documents per embedding batch. Defaults to 512.

        Returns:
            FAISS: A FAISS vector index.
        """

        # Ensure `vector_index` is a valid FAISS object before calling `merge_from` for the first time
        vector_index = FAISS.from_documents(documents[:batch_size], embeddings)

        for i in range(batch_size, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_vector_index = FAISS.from_documents(batch_docs, embeddings)

            # merge multiple FAISS indexes into one for unified retrieval
            vector_index.merge_from(batch_vector_index)

        return vector_index

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
    def load(path: Path, embeddings: Embeddings, index_name: str = "index") -> FAISS:
        """Load FAISS index from local storage.

        Args:
            path (Path): Path to the FAISS index directory.
            embeddings (Embeddings): Embedding model used for index.
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
