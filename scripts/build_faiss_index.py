import os

from app.config import settings
from app.core.services.similarity import SimilarityConverter
from app.data.chunker import Chunker
from app.data.data_loader import DataLoader
from app.data.embedding import EmbeddingFactory
from app.data.faiss_index import FAISSIndexManager
from app.data.tokenizer import TokenizerManager


def main():
    # Step 1: Load Data as Documents
    print("Loading dataset...")
    loader = DataLoader()
    try:
        match settings.SOURCE_TYPE:
            case "csv":
                print(f"Source type: CSV - {settings.QA_FILE_PATH.name}")
                if not os.path.exists(settings.QA_FILE_PATH):
                    raise FileNotFoundError(
                        f"Dataset not found: {settings.QA_FILE_PATH}"
                    )
                questions, answers = loader.load_csv_file(
                    settings.QA_FILE_PATH, encoding="utf-8"
                )
                source_name = settings.QA_FILE_PATH.name
            case "huggingface":
                print(f"Source type: Hugging Face - {settings.DATASET_NAME}")
                questions, answers = loader.load_huggingface_dataset(
                    settings.DATASET_NAME,
                    settings.DATASET_HASH,
                    settings.QUES_COLUMN_NAME,
                    settings.ANS_COLUMN_NAME,
                )
                source_name = settings.DATASET_NAME
            case _:
                raise ValueError(f"Unknown source type: {settings.SOURCE_TYPE}")
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset: {e}")

    documents = loader.build_documents(
        source_name,
        questions,
        answers,
        settings.QUES_PATTERN,
        settings.ANS_PATTERN,
    )

    if not documents:
        raise ValueError("Dataset is empty, cannot create index")
    print(f"Successfully built {len(documents)} documents from dataset")

    # Step 2: Text Chunking
    tokenizer_manager = TokenizerManager()
    if os.path.exists(settings.tokenizer_path):
        print("Loading existing tokenizer from local...")
        tokenizer = tokenizer_manager.load_from_local(settings.tokenizer_path)
    else:
        print("Downloading tokenizer from model...")
        tokenizer = tokenizer_manager.download_and_save(
            settings.tokenizer_path,
            settings.PRETRAINED_MODEL,
            settings.PRETRAINED_MODEL_HASH,
        )
        print(f"Saved tokenizer to {settings.tokenizer_path}")

    chunker = Chunker()
    chunked_documents = chunker.split_documents(
        documents, tokenizer, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    )
    print(f"Split into {len(chunked_documents)} chunks")

    # Step 3: Embeddings
    embedding_model = EmbeddingFactory.create_jina_embedding_model()

    # Step 4: Build FAISS Index
    if os.path.exists(settings.FAISS_INDEX_PATH):
        print("FAISS index already exists")
    else:
        print("Creating FAISS index...")
        vector_index = FAISSIndexManager.build(
            documents, embedding_model, settings.EMBEDDING_BATCH_SIZE
        )
        FAISSIndexManager.save(
            vector_index, settings.FAISS_INDEX_PATH, settings.FAISS_INDEX_NAME
        )
        print(f"Saved FAISS index to {settings.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
