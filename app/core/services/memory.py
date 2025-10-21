from typing import Dict

from langchain.memory import ConversationBufferWindowMemory


class SessionMemory:
    """Wrapper for multi-turn conversation memory using LangChain's ConversationBufferWindowMemory."""

    def __init__(self, memory: ConversationBufferWindowMemory) -> None:
        """Initialize SessionMemory.

        Args:
            memory (ConversationBufferWindowMemory): LangChain memory instance.

        Returns:
            None
        """
        self.memory = memory

    @classmethod
    def create(cls, memory_window: int) -> "SessionMemory":
        """Create a new SessionMemory instance with default settings.

        Args:
            memory_window (int): Number of recent conversation turns to keep in memory.

        Returns:
            SessionMemory: Initialized memory wrapper.
        """

        memory = ConversationBufferWindowMemory(
            k=memory_window, return_messages=True, memory_key="history"
        )
        return cls(memory)

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save a turn of conversation to memory.

        Args:
            inputs (Dict[str, Any]): User inputs.
            outputs (Dict[str, Any]): LLM outputs.

        Returns:
            None
        """
        self.memory.save_context(inputs, outputs)

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Load memory variables for current conversation context.

        Args:
            inputs (Dict[str, Any]): Input variables (can be empty).

        Returns:
            Dict[str, Any]: Memory variables including conversation history.
        """
        return self.memory.load_memory_variables(inputs)
