from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer


class TokenizerManager:
    """Manager for downloading and saving tokenizer."""

    @staticmethod
    def download_and_save(
        path: Path, pretrained_model_name: str, pretrained_model_hash: str
    ) -> PreTrainedTokenizer:
        """Download the tokenizer from pretrained model and save it locally.

        Args:
            path (str): Path to save the tokenizer.
            pretrained_model_name (str): Pretrained model name or path.
            pretrained_model_hash (str): Specific model revision hash.

        Returns:
            PreTrainedTokenizer: The downloaded tokenizer instance.
        """
        # TODO: Add other tokenizer sources

        # Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, revision=pretrained_model_hash
        )  # nosec: B615
        tokenizer.save_pretrained(path)

        return tokenizer

    @staticmethod
    def load_from_local(path: Path) -> PreTrainedTokenizer:
        """Load tokenizer from a local path.

        Args:
            path (str): Path where tokenizer is saved.

        Returns:
            PreTrainedTokenizer: The loaded tokenizer instance.
        """
        return AutoTokenizer.from_pretrained(path)  # nosec: B615
