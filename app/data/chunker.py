from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizer


class Chunker:
    """Class for splitting documents into chunks using a tokenizer."""

    def __init__(self) -> None:
        """Initialize the Chunker.

        Returns:
            None
        """
        # fmt: off
        # To keep the list readable and preserve its line breaks
        self.separators: List[str] = [
            "\n\n", "\r\n", "\n",
            "。", "！", "？", "；", "，",
            ". ", "! ", "? ", "; ",
            ", ", ": ", "：", " - ", "—", "…", "...",
            "(", ")", "（", "）", "[", "]", "【", "】",
            "{", "}", "『", "』", "「", "」", "\"", "'", "“", "”", "‘", "’",
            " ", "\t", ""
        ]
        # fmt: on

    def split_documents(
        self,
        documents: List[Document],
        tokenizer: PreTrainedTokenizer,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Document]:
        """Split documents into chunks based on token count.

        Args:
            documents (List[Document]): List of LangChain Document objects.
            tokenizer: A tokenizer instance used to calculate token lengths.
            chunk_size (int): Maximum tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between chunks.

        Returns:
            List[Document]: Chunked Document objects.
        """

        # Combine character/punctuation splitting with token counting to create fine-grained chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(tokenizer.encode(text)),
            separators=self.separators,
        )
        chunked_documents = text_splitter.split_documents(documents)

        return chunked_documents
