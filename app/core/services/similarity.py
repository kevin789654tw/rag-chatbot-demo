class SimilarityConverter:
    """Utility class for converting between FAISS distance score and cosine similarity."""

    @staticmethod
    def score_to_similarity(distance: float) -> float:
        """Convert FAISS L2 distance score to cosine similarity.

        Args:
            distance (float): FAISS distance score (L2 distance).

        Returns:
            float: Cosine similarity value in range [-1, 1].
        """
        return 1 - (distance**2) / 2

    @staticmethod
    def similarity_to_score(cosine_similarity: float) -> float:
        """Convert cosine similarity to FAISS L2 distance score.

        Args:
            cosine_similarity (float): Cosine similarity value in range [-1, 1].

        Returns:
            float: FAISS distance score (L2 distance).
        """
        return (2 * (1 - cosine_similarity)) ** 0.5
