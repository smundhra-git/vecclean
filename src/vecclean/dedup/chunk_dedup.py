"""
Chunk-level deduplication for VecClean.

Provides comprehensive chunk deduplication using various algorithms including
exact matching, hash-based deduplication, and similarity-based deduplication.
"""

from __future__ import annotations

import logging
from typing import List, Set, Dict, Tuple, Optional
import hashlib
from collections import defaultdict
import numpy as np

from vecclean.core.types import CleanedChunk
from vecclean.core.config import DeduplicationConfig
from vecclean.utils.hashing import generate_content_hash

logger = logging.getLogger(__name__)


class ChunkDeduplicator:
    """
    Chunk-level deduplication with multiple strategies.
    
    Supports exact matching, fuzzy matching, and semantic similarity-based
    deduplication to remove duplicate content at the chunk level.
    """
    
    def __init__(self, config: DeduplicationConfig) -> None:
        """
        Initialize chunk deduplicator.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config
        self._exact_hashes: Set[str] = set()
        self._fuzzy_hashes: Dict[str, List[CleanedChunk]] = defaultdict(list)
        self._processed_count = 0
        self._duplicate_count = 0
        self._dedup_history: Dict[int, Tuple[bool, Optional[int]]] = {}

    async def deduplicate_chunks(self, chunks: List[CleanedChunk]) -> List[CleanedChunk]:
        """
        Remove duplicate chunks from a list of chunks.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            List of unique chunks
        """
        if not chunks:
            return chunks
        
        unique_chunks = []
        
        for chunk in chunks:
            self._processed_count += 1
            
            if not await self._is_duplicate(chunk):
                unique_chunks.append(chunk)
                await self._add_to_cache(chunk)
            else:
                self._duplicate_count += 1
                logger.debug(f"Removed duplicate chunk: {chunk.text[:50]}...")
        
        logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
        return unique_chunks
    
    async def _is_duplicate(self, chunk: CleanedChunk) -> bool:
        """Check if a chunk is a duplicate."""
        # Strategy 1: Exact hash matching
        if self.config.enable_exact_dedup:
            exact_hash = self._compute_exact_hash(chunk.text)
            if exact_hash in self._exact_hashes:
                return True
        
        # Strategy 2: Fuzzy hash matching for near-duplicates
        if self.config.enable_fuzzy_dedup:
            fuzzy_hash = self._compute_fuzzy_hash(chunk.text)
            if fuzzy_hash in self._fuzzy_hashes:
                # Check similarity with existing chunks
                for existing_chunk in self._fuzzy_hashes[fuzzy_hash]:
                    similarity = self._compute_similarity(chunk.text, existing_chunk.text)
                    if similarity >= self.config.similarity_threshold:
                        return True
        
        # Strategy 3: Semantic similarity (if enabled)
        if hasattr(self.config, "enable_semantic_dedup") and self.config.enable_semantic_dedup:
            # This would require embeddings - placeholder for now
            pass
        
        return False
    
    async def _add_to_cache(self, chunk: CleanedChunk) -> None:
        """Add a chunk to deduplication cache."""
        if self.config.enable_exact_dedup:
            exact_hash = self._compute_exact_hash(chunk.text)
            self._exact_hashes.add(exact_hash)
        
        if self.config.enable_fuzzy_dedup:
            fuzzy_hash = self._compute_fuzzy_hash(chunk.text)
            self._fuzzy_hashes[fuzzy_hash].append(chunk)
    
    def _compute_exact_hash(self, text: str) -> str:
        """Compute exact hash for precise deduplication."""
        # Normalize text before hashing
        normalized = text.lower().strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _compute_fuzzy_hash(self, text: str) -> str:
        """Compute fuzzy hash for near-duplicate detection."""
        # Simple fuzzy hash based on word count and length
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Create buckets based on length and word count
        length_bucket = char_count // 100  # Group by ~100 char buckets
        word_bucket = word_count // 10     # Group by ~10 word buckets
        
        return f"{length_bucket}_{word_bucket}"
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        if text1 == text2:
            return 1.0
        
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_exact_duplicates(self, chunks: List[CleanedChunk]) -> List[CleanedChunk]:
        """Find and remove exact duplicates based on content hash."""
        unique_chunks = []
        for chunk in chunks:
            content_hash = generate_content_hash(chunk.cleaned_text)
            if content_hash not in self._exact_hashes:
                self._exact_hashes.add(content_hash)
                unique_chunks.append(chunk)
        return unique_chunks

    def _find_semantic_duplicates(self, chunks: List[CleanedChunk]) -> List[CleanedChunk]:
        """
        Find semantic duplicates using embedding similarity.
        
        This is a placeholder and needs to be implemented.
        """
        if not all(chunk.embedding is not None for chunk in chunks):
            logger.warning("Embeddings not found for all chunks. Skipping semantic deduplication.")
            return []
            
        logger.warning("Semantic deduplication is not yet implemented.")
        return []

    def deduplicate(self, chunks: List[CleanedChunk]) -> List[CleanedChunk]:
        """Deduplicate a list of chunks based on the configured strategy."""
        if self.config.strategy == "exact":
            unique_chunks = self._find_exact_duplicates(chunks)
        elif self.config.strategy == "semantic":
            unique_chunks = self._find_semantic_duplicates(chunks)
        else:
            raise ValueError(f"Unknown deduplication strategy: {self.config.strategy}")
        
        num_deduplicated = len(chunks) - len(unique_chunks)
        if num_deduplicated > 0:
            logger.info(f"Removed {num_deduplicated} duplicate chunks.")
            
        return unique_chunks
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            "processed_chunks": self._processed_count,
            "duplicate_chunks_removed": self._duplicate_count,
            "unique_chunks": self._processed_count - self._duplicate_count,
            "exact_hashes_cached": len(self._exact_hashes),
            "fuzzy_hash_buckets": len(self._fuzzy_hashes)
        }
    
    def clear_cache(self) -> None:
        """Clear deduplication cache."""
        self._exact_hashes.clear()
        self._fuzzy_hashes.clear()
        self._processed_count = 0
        self._duplicate_count = 0
        logger.info("Deduplication cache cleared")


# Utility function for backwards compatibility
async def deduplicate_chunks(chunks: List[CleanedChunk], config: DeduplicationConfig) -> List[CleanedChunk]:
    """
    Deduplicate a list of chunks.
    
    Args:
        chunks: List of chunks to deduplicate
        config: Deduplication configuration
        
    Returns:
        List of unique chunks
    """
    deduplicator = ChunkDeduplicator(config)
    return await deduplicator.deduplicate_chunks(chunks) 