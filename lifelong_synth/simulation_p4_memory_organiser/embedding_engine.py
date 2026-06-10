"""
Embedding Engine — Local embedding model wrapper for STARE-E v2.

Zero API cost. Uses sentence-transformers with multilingual-e5-base.
Supports query/passage prefix convention for asymmetric retrieval.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Local embedding engine using multilingual-e5-base. Zero API cost.

    Uses query/passage prefix convention for asymmetric retrieval.
    The E5 model family requires 'query: ' prefix for retrieval queries
    and 'passage: ' prefix for documents/passages.
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ):
        self._model = None
        self._model_name = model_name
        self._device = device
        self._cache: Dict[int, np.ndarray] = {}
        self._dim: Optional[int] = None
        # Lazy loading: model is loaded on first use
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                f"Loading embedding model: {self._model_name} "
                f"(device={self._device})..."
            )
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
            # Determine embedding dimension
            test_emb = self._model.encode("test", normalize_embeddings=True)
            self._dim = test_emb.shape[0]
            self._loaded = True
            logger.info(
                f"Embedding model loaded: dim={self._dim}, "
                f"device={self._device}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load embedding model '{self._model_name}': {e}. "
                f"Embedding features will be disabled."
            )
            self._model = None
            self._loaded = True  # Mark as attempted

    @property
    def available(self) -> bool:
        """Whether the embedding model is loaded and ready."""
        self._ensure_loaded()
        return self._model is not None

    @property
    def dim(self) -> int:
        """Embedding dimension (768 for multilingual-e5-base)."""
        self._ensure_loaded()
        return self._dim or 768

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a query text (retrieval side). Uses 'query:' prefix."""
        self._ensure_loaded()
        if self._model is None:
            return np.zeros(self.dim, dtype=np.float32)

        prefixed = f"query: {text}"
        key = hash(prefixed)
        if key not in self._cache:
            self._cache[key] = self._model.encode(
                prefixed, normalize_embeddings=True,
            ).astype(np.float32)
        return self._cache[key]

    def encode_passage(self, text: str) -> np.ndarray:
        """Encode a passage/document text (write side). Uses 'passage:' prefix."""
        self._ensure_loaded()
        if self._model is None:
            return np.zeros(self.dim, dtype=np.float32)

        prefixed = f"passage: {text}"
        key = hash(prefixed)
        if key not in self._cache:
            self._cache[key] = self._model.encode(
                prefixed, normalize_embeddings=True,
            ).astype(np.float32)
        return self._cache[key]

    def encode_passages_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode passages for efficiency at write time."""
        self._ensure_loaded()
        if self._model is None:
            return np.zeros((len(texts), self.dim), dtype=np.float32)

        prefixed = [f"passage: {t}" for t in texts]
        return self._model.encode(
            prefixed, normalize_embeddings=True, batch_size=32,
        ).astype(np.float32)

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity (embeddings are already L2-normalized → dot product)."""
        return float(np.dot(emb_a, emb_b))

    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self._cache.clear()
