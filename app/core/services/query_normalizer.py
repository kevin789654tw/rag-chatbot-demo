import json
from typing import Any, Dict, List

from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.prompts.query_normalization_prompt import QUERY_NORMALIZATION_PROMPT


class QueryNormalizer:
    """Service for query normalization and rewriting using LLM."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """Initialize QueryNormalizer.

        Args:
            llm (ChatOpenAI): The LLM instance used for query rewriting.

        Returns:
            None
        """
        self.llm = llm

    async def normalize_and_rewrite(
        self, query: str, history: List[HumanMessage]
    ) -> Dict[str, Any]:
        """Normalize and rewrite a user query based on conversation history.

        Args:
            query (str): The original user query.
            history (List[HumanMessage]): List of conversation history messages.

        Returns:
            Dict[str, Any]: Dictionary with keys:
                - 'needs_rewrite' (bool): True if the query was rewritten.
                - 'normalize_query' (List[str]): The rewritten or original query.
        """
        history_text = self._format_history(history)
        prompt = QUERY_NORMALIZATION_PROMPT.format(history=history_text, question=query)

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        try:
            print("Trying to parse JSON...")
            result: List[str] = json.loads(response.content)
            needs_rewrite: bool = result[0] != query if result else False
        except json.JSONDecodeError:
            print("JSON parsing failed.")
            result = [query]
            needs_rewrite = False

        return {"needs_rewrite": needs_rewrite, "normalize_query": result}

    @staticmethod
    def _format_history(history: List[HumanMessage]) -> str:
        """Format conversation history for prompts.

        Args:
            history (List[HumanMessage]): List of past messages.

        Returns:
            str: Formatted history string for LLM prompts.
        """
        return "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in history
        )
