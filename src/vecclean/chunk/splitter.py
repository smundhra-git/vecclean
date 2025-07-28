"""
Text chunking for VecClean.

Splits text into semantically meaningful chunks with configurable overlap,
sentence boundary preservation, and tokenizer awareness for optimal embedding quality.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from vecclean.core.config import ChunkingConfig


logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Different chunking strategies."""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TOKEN = "token"
    SEMANTIC = "semantic"
    FIXED = "fixed"


class TextChunker:
    """
    Advanced text chunker with multiple strategies and tokenizer awareness.
    
    Supports sentence-aware chunking, semantic chunking, and various strategies
    optimized for different use cases and embedding models.
    """
    
    def __init__(self, config: ChunkingConfig) -> None:
        """
        Initialize chunker with configuration.
        
        Args:
            config: Chunking configuration
        """
        self.config = config
        self.tokenizer = None
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """Initialize tokenizer for token-aware chunking."""
        try:
            # Try to use tiktoken for OpenAI-compatible tokenization
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Initialized tiktoken tokenizer")
        except ImportError:
            try:
                # Fallback to transformers tokenizer
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Initialized transformers tokenizer")
            except ImportError:
                logger.warning("No tokenizer available, using word-based approximation")
                self.tokenizer = None
    
    async def chunk_text(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks using the configured strategy.
        
        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        if not text.strip():
            return []
        
        strategy = ChunkingStrategy(self.config.strategy)
        
        if strategy == ChunkingStrategy.SENTENCE:
            return await self._chunk_by_sentences(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return await self._chunk_by_paragraphs(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.TOKEN:
            return await self._chunk_by_tokens(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return await self._chunk_semantically(text, chunk_size, overlap)
        else:
            # Default to sentence-based chunking
            return await self._chunk_by_sentences(text, chunk_size, overlap)
    
    async def _chunk_by_sentences(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """Chunk text by sentences with token awareness."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_char_pos = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                end_char_pos = start_char_pos + len(chunk_text)
                chunks.append((chunk_text, start_char_pos, end_char_pos))
                
                # Handle overlap
                if overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = []
                    overlap_tokens = 0
                    
                    # Take sentences from the end for overlap
                    for i in range(len(current_chunk) - 1, -1, -1):
                        sent_tokens = self._count_tokens(current_chunk[i])
                        if overlap_tokens + sent_tokens <= overlap:
                            overlap_sentences.insert(0, current_chunk[i])
                            overlap_tokens += sent_tokens
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                    start_char_pos = end_char_pos - len(' '.join(overlap_sentences))
                else:
                    current_chunk = []
                    current_tokens = 0
                    start_char_pos = end_char_pos
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end_char_pos = start_char_pos + len(chunk_text)
            chunks.append((chunk_text, start_char_pos, end_char_pos))
        
        logger.debug(f"Sentence-based chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
        return chunks
    
    async def _chunk_by_paragraphs(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """Chunk text by paragraphs with sentence boundary preservation."""
        paragraphs = self._split_into_paragraphs(text)
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_char_pos = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self._count_tokens(paragraph)
            
            # If paragraph is too large, split it further
            if paragraph_tokens > chunk_size:
                # Split large paragraph by sentences
                para_chunks = await self._chunk_by_sentences(paragraph, chunk_size, overlap)
                chunks.extend(para_chunks)
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_tokens + paragraph_tokens > chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                end_char_pos = start_char_pos + len(chunk_text)
                chunks.append((chunk_text, start_char_pos, end_char_pos))
                
                # Handle overlap
                if overlap > 0 and len(current_chunk) > 1:
                    # Take the last paragraph for overlap if it fits
                    last_para_tokens = self._count_tokens(current_chunk[-1])
                    if last_para_tokens <= overlap:
                        current_chunk = [current_chunk[-1]]
                        current_tokens = last_para_tokens
                        start_char_pos = end_char_pos - len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_tokens = 0
                        start_char_pos = end_char_pos
                else:
                    current_chunk = []
                    current_tokens = 0
                    start_char_pos = end_char_pos
            
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens
        
        # Add remaining paragraphs as final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            end_char_pos = start_char_pos + len(chunk_text)
            chunks.append((chunk_text, start_char_pos, end_char_pos))
        
        logger.debug(f"Paragraph-based chunking: {len(paragraphs)} paragraphs -> {len(chunks)} chunks")
        return chunks
    
    async def _chunk_by_tokens(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """Chunk text by exact token count."""
        if not self.tokenizer:
            # Fallback to word-based chunking
            return await self._chunk_by_words(text, chunk_size, overlap)
        
        # Tokenize the entire text
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text)
            decode_func = self.tokenizer.decode
        else:
            tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
            decode_func = lambda x: self.tokenizer.decode(x, skip_special_tokens=True)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = decode_func(chunk_tokens)
            
            # Calculate character positions (approximate)
            start_char = int((start_idx / len(tokens)) * len(text))
            end_char = int((end_idx / len(tokens)) * len(text))
            
            chunks.append((chunk_text, start_char, end_char))
            
            # Move start position with overlap
            start_idx = end_idx - overlap
            if start_idx >= end_idx:
                break
        
        logger.debug(f"Token-based chunking: {len(tokens)} tokens -> {len(chunks)} chunks")
        return chunks
    
    async def _chunk_semantically(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """Chunk text using semantic similarity (requires embeddings)."""
        # For now, use enhanced sentence-based chunking with semantic hints
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Group sentences that are semantically related
        semantic_groups = self._group_sentences_semantically(sentences)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_char_pos = 0
        
        for group in semantic_groups:
            group_text = ' '.join(group)
            group_tokens = self._count_tokens(group_text)
            
            # If group is too large, split it
            if group_tokens > chunk_size:
                group_chunks = await self._chunk_by_sentences(group_text, chunk_size, overlap)
                chunks.extend(group_chunks)
                continue
            
            # Check if adding this group would exceed chunk size
            if current_tokens + group_tokens > chunk_size and current_chunk:
                # Create chunk from current content
                chunk_text = ' '.join(current_chunk)
                end_char_pos = start_char_pos + len(chunk_text)
                chunks.append((chunk_text, start_char_pos, end_char_pos))
                
                current_chunk = group
                current_tokens = group_tokens
                start_char_pos = end_char_pos
            else:
                current_chunk.extend(group)
                current_tokens += group_tokens
        
        # Add remaining content as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end_char_pos = start_char_pos + len(chunk_text)
            chunks.append((chunk_text, start_char_pos, end_char_pos))
        
        logger.debug(f"Semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
        return chunks
    
    async def _chunk_by_words(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """Fallback word-based chunking when no tokenizer is available."""
        words = text.split()
        if not words:
            return []
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions (approximate)
            start_char = len(' '.join(words[:start_idx]))
            end_char = start_char + len(chunk_text)
            
            chunks.append((chunk_text, start_char, end_char))
            
            # Move start position with overlap
            start_idx = end_idx - overlap
            if start_idx >= end_idx:
                break
        
        logger.debug(f"Word-based chunking: {len(words)} words -> {len(chunks)} chunks")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved boundary detection."""
        # Enhanced sentence splitting with abbreviation handling
        abbreviations = {
            'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'vs.', 'etc.', 'i.e.', 'e.g.',
            'inc.', 'ltd.', 'corp.', 'co.', 'jr.', 'sr.', 'ph.d.', 'm.d.',
            'b.a.', 'm.a.', 'b.s.', 'm.s.', 'u.s.', 'u.k.', 'no.', 'vol.'
        }
        
        sentences = []
        current_sentence = ""
        
        # Split by potential sentence endings
        parts = re.split(r'([.!?]+)', text)
        
        for i in range(0, len(parts), 2):
            if i >= len(parts):
                break
                
            text_part = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            
            current_sentence += text_part + punct
            
            # Check if this is actually a sentence boundary
            if punct and not self._is_abbreviation(current_sentence, abbreviations):
                sentence = current_sentence.strip()
                if sentence and len(sentence) >= self.config.min_chunk_size:
                    sentences.append(sentence)
                current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentence = current_sentence.strip()
            if len(sentence) >= self.config.min_chunk_size:
                sentences.append(sentence)
        
        logger.debug(f"Split text into {len(sentences)} sentences.")

        # If no sentences found, return the whole text as one sentence
        if not sentences and len(text) >= self.config.min_sentence_length:
            sentences = [text]
        print(f"[DEBUG] Sentences found for chunking: {sentences}")
        return sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _group_sentences_semantically(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences that are semantically related."""
        # Simple heuristic-based grouping (can be enhanced with embeddings)
        groups = []
        current_group = []
        
        for sentence in sentences:
            if not current_group:
                current_group.append(sentence)
                continue
            
            # Heuristic: sentences with similar keywords are related
            if self._sentences_are_related(current_group[-1], sentence):
                current_group.append(sentence)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _sentences_are_related(self, sent1: str, sent2: str) -> bool:
        """Simple heuristic to check if sentences are semantically related."""
        # Extract key words (excluding common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words1 = {word.lower() for word in re.findall(r'\b\w+\b', sent1) if word.lower() not in stop_words}
        words2 = {word.lower() for word in re.findall(r'\b\w+\b', sent2) if word.lower() not in stop_words}
        
        # Check for word overlap
        overlap = len(words1 & words2)
        total_unique = len(words1 | words2)
        
        if total_unique == 0:
            return False
        
        similarity = overlap / total_unique
        return similarity > 0.2  # Threshold for considering sentences related
    
    def _is_abbreviation(self, text: str, abbreviations: set) -> bool:
        """Check if text ends with a known abbreviation."""
        words = text.strip().split()
        if not words:
            return False
        
        last_word = words[-1].lower()
        return last_word in abbreviations
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using available tokenizer."""
        if not text:
            return 0
        
        if self.tokenizer:
            if hasattr(self.tokenizer, 'encode'):
                return len(self.tokenizer.encode(text))
            else:
                return len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
        else:
            # Fallback to word count approximation
            return len(text.split()) 