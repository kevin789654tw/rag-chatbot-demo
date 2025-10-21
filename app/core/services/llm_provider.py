from langchain_openai import ChatOpenAI


class LLMFactory:
    """Factory for creating or reusing ChatOpenAI instances.

    Attributes:
        _cache (dict[str, ChatOpenAI]): Class-level cache storing LLM instances.
    """

    _cache: dict[str, ChatOpenAI] = {}

    def get_llm(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        streaming: bool = True,
        use_cache: bool = True,
    ) -> ChatOpenAI:
        """Get OpenAI-compatible chat model instance from cache or create a new one.

        Note:
            OpenRouter simulates the OpenAI API, allowing ChatOpenAI to call non-OpenAI LLMs.

        Args:
            api_key (str): API key for OpenAI or OpenRouter.
            base_url (str): Base URL of the API endpoint.
            model (str): Model name.
            temperature (float):
                Sampling temperature controlling output diversity. Must be between 0 and 1.
            streaming (bool, optional):
                Whether to enable streaming responses. Defaults to True.
            use_cache (bool, optional):
                Whether to use the cached instance if available. Defaults to True.

        Returns:
            ChatOpenAI: An instance of ChatOpenAI configured with the given parameters.
        """
        cache_key = f"{api_key}-{base_url}-{model}-{temperature}-{streaming}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            streaming=streaming,
        )

        if use_cache:
            self._cache[cache_key] = llm

        return llm

    def list_cache(self) -> list[str]:
        """List all keys currently stored in the LLM cache.

        Returns:
            list[str]: List of cache keys.
        """
        return list(self._cache.keys())

    def clear_cache(self, key: str | None = None) -> None:
        """Clear cached LLM instances.

        Args:
            key (str | None, optional):
                Specific cache key to remove. If None, all cached instances
                will be cleared. Defaults to None.

        Returns:
            None
        """
        if key:
            removed = self._cache.pop(key, None)
            if removed:
                print(f"[LLMFactory] Removed cached: {key}")
            else:
                print(f"[LLMFactory] Key '{key}' not found.")
        else:
            self._cache.clear()
            print("[LLMFactory] Cleared all cached")
