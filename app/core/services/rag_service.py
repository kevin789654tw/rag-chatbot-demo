from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.core.services.similarity import SimilarityConverter


class RAGService:
    """Service for retrieving documents and building context for RAG."""

    @staticmethod
    def retrieve_docs(
        queries: List[str], vector_index: FAISS, retrieval_top_k: int
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents from FAISS index for list of queries.

        Args:
            queries (List[str]): List of query strings to retrieve documents for.
            vector_index (FAISS): FAISS index instance used for similarity search.
            retrieval_top_k (int): Number of top similar documents to retrieve per query.

        Returns:
            List[Tuple[Document, float]]: List of documents with similarity scores.
        """
        docs_with_scores_list = []
        for q in queries:
            docs_with_scores = vector_index.similarity_search_with_score(
                q, k=retrieval_top_k
            )
            docs_with_scores_list.extend(docs_with_scores)

        # remove duplicate documents
        seen_contents = set()
        docs_with_scores = []
        for doc, score in docs_with_scores_list:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                docs_with_scores.append((doc, score))

        return docs_with_scores

    @staticmethod
    def build_contexts_and_sources(
        docs_with_scores: List[Tuple[Document, float]],
        similarity_threshold: float,
        source_type: str | None = None,
        qa_file_path: str | None = None,
        dataset_name: str | None = None,
    ) -> Tuple[str, str]:
        """Build context string and source references from retrieved documents.

        Args:
            docs_with_scores (List[Tuple[Document, float]]):
                Retrieved documents with scores.
            similarity_threshold (float):
                Minimum similarity required for document to be included.
            source_type (str | None, optional):
                Type of data source ('csv', 'huggingface', or None). Defaults to None.
            qa_file_path (str | None, optional):
                Path to the CSV file if source_type is 'csv'. Defaults to None.
            dataset_name (str | None, optional):
                Dataset name if source_type is 'huggingface'. Defaults to None.

        Returns:
            Tuple[str, str]:
                - context_text (str): Context string for LLM input.
                - sources_text (str): Formatted source references string.
        """
        if source_type == "csv":
            source_name = str(qa_file_path)
        elif source_type == "huggingface":
            source_name = f"https://huggingface.co/datasets/{dataset_name}"
        else:
            source_name = "...(Unknown)"

        contexts, sources = [], []
        for i, (doc, score) in enumerate(docs_with_scores):
            sim_score = SimilarityConverter.score_to_similarity(score)
            if sim_score >= similarity_threshold:
                contexts.append(
                    f"[Segment {i+1}] \n"
                    f"Question: {doc.page_content} \n"
                    f"Answer: {doc.metadata['answer']}"
                )
                # TODO: limit `doc.metadata['answer']` length to prevent overly long source display
                sources.append(
                    f"📖 **Source {i+1}:** \n"
                    f"(from {source_name}) \n"
                    f"> Question: {doc.page_content} \n"
                    f"> Answer: {doc.metadata['answer']}"
                )
            else:
                break

        context_text = (
            "\n---\n".join(contexts) if contexts else "No relevant documents found."
        )

        sources_text = "📚 **You can refer to the following sources:** \n"
        sources_text = (
            sources_text + "\n\n".join(sources)
            if sources
            else sources_text + "Sorry, I couldn't find any relevant documents 😕"
        )

        return context_text, sources_text
