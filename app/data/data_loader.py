from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset
from langchain.docstore.document import Document


class DataLoader:
    """Load and normalize Q&A data from multiple sources into LangChain Documents."""

    @staticmethod
    def load_csv_file(
        file_path: Path, encoding: str = "utf-8"
    ) -> tuple[list[str], list[str]]:
        """Load Q&A pairs from a CSV file.

        Args:
            file_path (Path): Path to the CSV file.
            encoding (str, optional): File encoding. Defaults to "utf-8".

        Returns:
            tuple[list[str], list[str]]: A tuple containing a list of questions and a list of answers.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            df = pd.read_csv(file_path, encoding=encoding).dropna()
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        if df.empty or len(df.columns) < 2:
            raise ValueError("CSV file is empty or lacks required columns.")

        questions = df.iloc[:, 0].tolist()
        answers = df.iloc[:, 1].tolist()

        return questions, answers

    @staticmethod
    def load_huggingface_dataset(
        dataset: str,
        dataset_hash: str,
        ques_column_name: str,
        ans_column_name: str,
    ) -> tuple[list[str], list[str]]:
        """Load Q&A pairs from a Hugging Face dataset.

        Args:
            dataset (str): The name of the Hugging Face dataset.
            dataset_hash (str): The revision hash or version of the dataset.
            ques_column_name (str): Column name for questions.
            ans_column_name (str): Column name for answers.

        Returns:
            tuple[list[str], list[str]]: A tuple containing a list of questions and a list of answers.
        """

        qa_dataset = load_dataset(
            dataset,
            revision=dataset_hash,
        )  # nosec: B615

        train_split = qa_dataset["train"]
        questions = list(train_split[ques_column_name])
        answers = list(train_split[ans_column_name])

        return questions, answers

    @staticmethod
    def build_documents(
        source_name: str,
        questions: list[str],
        answers: list[str],
        ques_pattern: str | None = None,
        ans_pattern: str | None = None,
    ) -> List[Document]:
        """Normalize question/answer text and build LangChain Documents.

        Args:
            source_name (str): Identifier for the data source.
            questions (list[str]): List of question strings.
            answers (list[str]): List of answer strings.
            ques_pattern (str | None, optional): Pattern to split questions. Defaults to None.
            ans_pattern (str | None, optional): Pattern to split answers. Defaults to None.

        Returns:
            List[Document]: A list of LangChain Document objects with normalized content and metadata.
        """

        documents: List[Document] = []
        for question, answer in zip(questions, answers):
            if ques_pattern and ques_pattern in question:
                question = question.split(ques_pattern)[1].strip()
            if ans_pattern and ans_pattern in answer:
                answer = answer.split(ans_pattern)[1].strip()

            doc = Document(
                page_content=question,
                metadata={"answer": answer, "source": source_name},
            )
            documents.append(doc)

        return documents
