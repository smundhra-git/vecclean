"""
Embedding generation for VecClean.

Provides embedding models and utilities for converting text chunks 
to vector representations with support for multiple backends,
quantization, similarity search, and quality evaluation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from vecclean.core.types import CleanedChunk, Embedding, EmbeddingModel
from vecclean.utils.io import ensure_directory_exists

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbedding(EmbeddingModel):
    """Local embedding model using sentence-transformers."""

    def __init__(
        self, 
        model_name: str, 
        device: str = "auto", 
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        **kwargs  # Accept any additional keyword arguments and ignore them
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.model = self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve the device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        logger.info(f"Loading model '{self.model_name}' on device '{self.device}'")
        model_path = self.model_name
        if self.cache_dir:
            model_path = str(Path(self.cache_dir) / self.model_name)
            ensure_directory_exists(model_path)
        return SentenceTransformer(model_path, device=self.device)

    async def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name


def create_embedding_model(config: "EmbeddingConfig") -> EmbeddingModel:
    """
    Create an embedding model with the specified configuration.
    
    Args:
        config: Embedding configuration
        
    Returns:
        Configured embedding model
    """
    # This can be extended to support other model types
    return LocalSentenceTransformerEmbedding(
        model_name=config.model_name,
        device=config.device,
        cache_dir=config.cache_dir,
    ) 