import os

from app.config import settings
from app.core.services.similarity import SimilarityConverter
from app.data.csv_loader import CSVLoader
from app.data.embedding import EmbeddingFactory
from app.data.faiss_index import FAISSIndexManager


def main():
    # Step 1: Load CSV as Documents
    if not os.path.exists(settings.QA_FILE_PATH):
        raise FileNotFoundError(f"CSV file not found: {settings.QA_FILE_PATH}")

    loader = CSVLoader(
        settings.QA_FILE_PATH, settings.QUES_PATTERN, settings.ANS_PATTERN
    )
    try:
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    if not documents:
        raise ValueError("CSV is empty, cannot create index")

    # Step 2: Embeddings
    embedding_model = EmbeddingFactory.create_jina_embedding_model()

    # Step 3: Build FAISS Index
    if os.path.exists(settings.FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        vector_index = FAISSIndexManager.load(
            settings.FAISS_INDEX_PATH, embedding_model
        )
    else:
        print("Creating FAISS index...")
        vector_index = FAISSIndexManager.build(documents, embedding_model)
        FAISSIndexManager.save(
            vector_index, settings.FAISS_INDEX_PATH, settings.FAISS_INDEX_NAME
        )
        print(f"Saved FAISS index to {settings.FAISS_INDEX_PATH}")

    # Step 4: Test Query
    query = "How many types of Eucalyptus are grown around the world?"
    docs_with_scores = vector_index.similarity_search_with_score(
        query, k=settings.RETRIEVAL_TOP_K
    )

    for doc, score in docs_with_scores:
        score = SimilarityConverter.score_to_similarity(score)
        if score >= settings.SIMILARITY_THRESHOLD:
            print("\n")
            print("Found Question:", doc.page_content)
            print("Corresponding Answer:", doc.metadata["answer"])
            print("Similarity Score:", score)
        else:
            print("\n")
            print("No relevant question found.")

    if not docs_with_scores:
        print("No relevant question found.")


if __name__ == "__main__":
    main()
