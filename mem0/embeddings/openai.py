import os
import warnings
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        # Only pass `dimensions` to the API when the user set embedding_dims; non-matryoshka
        # OpenAI-compatible backends (vLLM, Voyage, etc.) reject the parameter
        self._pass_dimensions_to_api = self.config.embedding_dims is not None
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ").strip()
        if not text:
            raise ValueError("Cannot embed empty text")
        kwargs = {
            "input": [text],
            "model": self.config.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.config.embedding_dims
        return self.client.embeddings.create(**kwargs).data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        """Embed multiple texts in a single OpenAI API call.

        Automatically chunks into batches of 100 to stay within API limits.
        Filters out empty/whitespace-only texts to prevent 400 errors from
        backends like DashScope. Returns a list aligned with the *original*
        input; empty positions get a None placeholder.
        """
        MAX_BATCH = 100
        original_len = len(texts)
        cleaned = [text.replace("\n", " ").strip() for text in texts]

        # Build index of non-empty texts for actual embedding
        valid_indices = [i for i, t in enumerate(cleaned) if t]
        if not valid_indices:
            raise ValueError("Cannot embed batch: all texts are empty")

        valid_texts = [cleaned[i] for i in valid_indices]
        all_embeddings = []
        for i in range(0, len(valid_texts), MAX_BATCH):
            chunk = valid_texts[i : i + MAX_BATCH]
            kwargs = {
                "input": chunk,
                "model": self.config.model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self.config.embedding_dims
            response = self.client.embeddings.create(**kwargs)
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))

        # Map back to original positions; empty texts get None
        result = [None] * original_len
        for pos, embedding in zip(valid_indices, all_embeddings):
            result[pos] = embedding

        if len(all_embeddings) != len(valid_texts):
            raise ValueError(
                f"OpenAI embed_batch() returned {len(all_embeddings)} embeddings for {len(valid_texts)} non-empty texts"
                f" using model '{self.config.model}'"
            )
        return result
