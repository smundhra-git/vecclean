"""
Sentence-level deduplication for VecClean.

Provides sentence-level deduplication using exact matching, fuzzy matching,
and semantic similarity to remove duplicate sentences within and across chunks.
"""

from __future__ import annotations

import logging
import re
from typing import List, Set, Dict, Tuple, Optional
import hashlib
from collections import defaultdict

from vecclean.core.config import DeduplicationConfig

logger = logging.getLogger(__name__)


class SentenceDeduplicator:
    """
    Sentence-level deduplication with multiple strategies.
    
    Removes duplicate sentences within text content using various
    matching strategies including exact, fuzzy, and semantic similarity.
    """
    
    def __init__(self, config: DeduplicationConfig) -> None:
        """
        Initialize sentence deduplicator.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config
        self._exact_hashes: Set[str] = set()
        self._fuzzy_hashes: Dict[str, List[str]] = defaultdict(list)
        self._processed_count = 0
        self._duplicate_count = 0
    
    async def deduplicate_sentences(self, text: str) -> str:
        """
        Remove duplicate sentences from text.
        
        Args:
            text: Input text containing sentences
            
        Returns:
            Text with duplicate sentences removed
        """
        if not text or not text.strip():
            return text
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return text
        
        unique_sentences = []
        
        for sentence in sentences:
            self._processed_count += 1
            
            if not await self._is_duplicate_sentence(sentence):
                unique_sentences.append(sentence)
                await self._add_sentence_to_cache(sentence)
            else:
                self._duplicate_count += 1
                logger.debug(f"Removed duplicate sentence: {sentence[:50]}...")
        
        result = ' '.join(unique_sentences)
        logger.debug(f"Deduplicated {len(sentences)} sentences to {len(unique_sentences)} unique sentences")
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using proper sentence boundary detection."""
        # Clean the text first
        text = text.strip()
        if not text:
            return []
        
        # Use regex to split on sentence boundaries
        # This handles common abbreviations and edge cases
        sentences = []
        
        # Split on sentence terminators followed by whitespace and capital letters
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(?=\s+[A-Z])'
        parts = re.split(pattern, text)
        
        for part in parts:
            part = part.strip()
            if part and len(part) >= self.config.min_sentence_length:
                sentences.append(part)
        
        # If no sentences found, return the whole text as one sentence
        if not sentences and len(text) >= self.config.min_sentence_length:
            sentences = [text]
        
        return sentences
    
    async def _is_duplicate_sentence(self, sentence: str) -> bool:
        """Check if a sentence is a duplicate."""
        # Strategy 1: Exact hash matching
        if self.config.enable_exact_dedup:
            exact_hash = self._compute_exact_sentence_hash(sentence)
            if exact_hash in self._exact_hashes:
                return True
        
        # Strategy 2: Fuzzy matching for near-duplicates
        if self.config.enable_fuzzy_dedup:
            fuzzy_hash = self._compute_fuzzy_sentence_hash(sentence)
            if fuzzy_hash in self._fuzzy_hashes:
                # Check similarity with existing sentences
                for existing_sentence in self._fuzzy_hashes[fuzzy_hash]:
                    similarity = self._compute_sentence_similarity(sentence, existing_sentence)
                    if similarity >= self.config.similarity_threshold:
                        return True
        
        return False
    
    async def _add_sentence_to_cache(self, sentence: str) -> None:
        """Add a sentence to deduplication cache."""
        if self.config.enable_exact_dedup:
            exact_hash = self._compute_exact_sentence_hash(sentence)
            self._exact_hashes.add(exact_hash)
        
        if self.config.enable_fuzzy_dedup:
            fuzzy_hash = self._compute_fuzzy_sentence_hash(sentence)
            self._fuzzy_hashes[fuzzy_hash].append(sentence)
    
    def _compute_exact_sentence_hash(self, sentence: str) -> str:
        """Compute exact hash for precise sentence deduplication."""
        # Normalize sentence: lowercase, remove extra spaces, strip punctuation at ends
        normalized = sentence.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation from start and end
        normalized = normalized.strip('.,!?;:')
        
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _compute_fuzzy_sentence_hash(self, sentence: str) -> str:
        """Compute fuzzy hash for near-duplicate sentence detection."""
        # Extract meaningful words (remove stopwords and short words)
        words = sentence.lower().split()
        meaningful_words = [w for w in words if len(w) > 2 and w not in self._get_stopwords()]
        
        # Create hash based on meaningful word count and first/last words
        word_count = len(meaningful_words)
        first_word = meaningful_words[0] if meaningful_words else ""
        last_word = meaningful_words[-1] if meaningful_words else ""
        
        return f"{word_count}_{first_word}_{last_word}"
    
    def _compute_sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """Compute similarity between two sentences."""
        if sentence1 == sentence2:
            return 1.0
        
        # Normalize both sentences
        norm1 = self._normalize_sentence_for_comparison(sentence1)
        norm2 = self._normalize_sentence_for_comparison(sentence2)
        
        if norm1 == norm2:
            return 1.0
        
        # Compute word-based Jaccard similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_sentence_for_comparison(self, sentence: str) -> str:
        """Normalize sentence for similarity comparison."""
        # Convert to lowercase
        normalized = sentence.lower()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _get_stopwords(self) -> Set[str]:
        """Get basic English stopwords for fuzzy hashing."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'could', 'should', 'have',
            'had', 'been', 'being', 'do', 'does', 'did', 'can', 'may', 'might'
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get sentence deduplication statistics."""
        return {
            "processed_sentences": self._processed_count,
            "duplicate_sentences_removed": self._duplicate_count,
            "unique_sentences": self._processed_count - self._duplicate_count,
            "exact_hashes_cached": len(self._exact_hashes),
            "fuzzy_hash_buckets": len(self._fuzzy_hashes)
        }
    
    def clear_cache(self) -> None:
        """Clear sentence deduplication cache."""
        self._exact_hashes.clear()
        self._fuzzy_hashes.clear()
        self._processed_count = 0
        self._duplicate_count = 0
        logger.info("Sentence deduplication cache cleared")


# Utility function for backwards compatibility
async def deduplicate_sentences(text: str, config: DeduplicationConfig) -> str:
    """
    Remove duplicate sentences from text.
    
    Args:
        text: Input text
        config: Deduplication configuration
        
    Returns:
        Text with duplicate sentences removed
    """
    deduplicator = SentenceDeduplicator(config)
    return await deduplicator.deduplicate_sentences(text)


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences with proper boundary detection.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Enhanced sentence splitting with better boundary detection
    text = text.strip()
    
    # Handle common abbreviations that shouldn't trigger sentence breaks
    # This is a simplified list - a full implementation would be more comprehensive
    abbreviations = {'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'etc.', 'vs.', 'i.e.', 'e.g.'}
    
    sentences = []
    current_sentence = ""
    
    # Split on potential sentence boundaries
    parts = re.split(r'([.!?]+)', text)
    
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
            
        text_part = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        
        current_sentence += text_part + punct
        
        # Check if this is actually a sentence boundary
        if punct and not any(current_sentence.strip().endswith(abbr) for abbr in abbreviations):
            sentence = current_sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                sentences.append(sentence)
            current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        sentence = current_sentence.strip()
        if len(sentence) > 10:
            sentences.append(sentence)
    
    return sentences 