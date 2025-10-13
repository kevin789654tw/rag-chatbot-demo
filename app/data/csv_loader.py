from pathlib import Path
from typing import List

import pandas as pd
from langchain.docstore.document import Document


class CSVLoader:
    """CSV Loader to parse Q&A CSV files into LangChain Documents."""

    def __init__(self, file_path: Path, ques_pattern: str, ans_pattern: str) -> None:
        """Initialize the CSVLoader.

        Args:
            file_path (str): Path to the CSV file.
            ques_pattern (str): Prefix pattern for questions.
            ans_pattern (str): Prefix pattern for answers.
        """
        self.file_path = file_path
        self.ques_pattern: str = ques_pattern
        self.ans_pattern: str = ans_pattern

    def load(self) -> List[Document]:
        """Load CSV file and convert rows into LangChain Documents.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        df = pd.read_csv(self.file_path).dropna()

        if df.empty or len(df.columns) < 2:
            raise ValueError("CSV data is empty or doesn't have at least 2 columns.")

        questions = df.iloc[:, 0].tolist()
        answers = df.iloc[:, 1].tolist()

        documents = []
        for question, answer in zip(questions, answers):
            if self.ques_pattern and self.ques_pattern in question:
                parts = question.split(self.ques_pattern, 1)
                question = parts[1].strip() if len(parts) > 1 else question.strip()

            if self.ans_pattern and self.ans_pattern in answer:
                parts = answer.split(self.ans_pattern, 1)
                answer = parts[1].strip() if len(parts) > 1 else answer.strip()

            doc = Document(
                page_content=question,
                metadata={"answer": answer, "source": str(self.file_path.name)},
            )
            documents.append(doc)

        return documents
