"""
Text normalization and cleaning utilities.

This module provides comprehensive text cleaning and normalization functionality
with both C++ (high-performance) and Python (fallback) implementations.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import TYPE_CHECKING

from vecclean.core.config import CleaningConfig

if TYPE_CHECKING:
    pass

# Check if C++ backend is available
try:
    import vecclean_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Text normalization and cleaning with C++ acceleration support.
    
    Provides comprehensive text cleaning including Unicode normalization,
    whitespace standardization, punctuation cleaning, and stopword removal.
    """
    
    def __init__(self, config: CleaningConfig) -> None:
        """
        Initialize normalizer.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config
        self.use_cpp = _CPP_AVAILABLE and getattr(config, 'use_cpp_backend', True)
        
        # Load stopwords if needed
        if config.remove_stopwords:
            self._stopwords = self._load_stopwords()
        else:
            self._stopwords = set()
        
        if self.use_cpp:
            try:
                # Import and initialize C++ normalizer if available
                import vecclean_cpp
                
                # Convert Python config to C++ config
                cpp_config = vecclean_cpp.CleaningConfig()
                cpp_config.normalize_unicode = True
                cpp_config.unicode_form = "NFC"
                cpp_config.normalize_whitespace = self.config.normalize_whitespace
                cpp_config.standardize_punctuation = self.config.standardize_punctuation
                cpp_config.min_text_length = self.config.min_text_length
                cpp_config.use_simd = True
                cpp_config.parallel_processing = True
                cpp_config.thread_count = 0  # Auto-detect
                
                self._cpp_processor = vecclean_cpp.TextProcessor(cpp_config)
                logger.info("Using C++ backend for text normalization")
            except Exception as e:
                logger.warning(f"Failed to initialize C++ backend: {e}, falling back to Python")
                self.use_cpp = False
                self._cpp_processor = None
        else:
            logger.info("Using Python backend for text normalization")
            self._cpp_processor = None
    
    async def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if self.use_cpp:
            return await self._clean_text_cpp(text)
        else:
            return await self._clean_text_python(text)
    
    async def _clean_text_cpp(self, text: str) -> str:
        """Clean text using C++ backend."""
        if self._cpp_processor is not None:
            try:
                # Use the C++ processor for high-performance cleaning
                return self._cpp_processor.clean_text(text)
            except Exception as e:
                logger.warning(f"C++ text cleaning failed: {e}, falling back to Python")
                return await self._clean_text_python(text)
        else:
            # Fallback to Python implementation
            return await self._clean_text_python(text)
    
    async def _clean_text_python(self, text: str) -> str:
        """Clean text using Python implementation."""
        import re
        
        original_length = len(text)
        result = text
        
        # Step 1: Unicode normalization
        if self.config.normalize_unicode:
            result = unicodedata.normalize('NFC', result)
            
            # Replace smart quotes and special characters
            replacements = {
                ''': "'", ''': "'", '"': '"', '"': '"',  # Smart quotes
                '–': '-', '—': '-',  # Em/en dashes
                '…': '...',  # Ellipsis
                '\u00a0': ' ',  # Non-breaking space
                '\u2019': "'",  # Right single quotation mark
                '\u201c': '"', '\u201d': '"',  # Left/right double quotation marks
            }
            for old, new in replacements.items():
                result = result.replace(old, new)
        
        # Step 2: Remove control characters (keep tabs, newlines, carriage returns)
        result = ''.join(c for c in result if ord(c) >= 32 or c in '\t\n\r')
        
        # Step 3: Standardize punctuation
        if self.config.standardize_punctuation:
            # Multiple periods to single periods
            result = re.sub(r'\.{2,}', '.', result)
            # Multiple question marks to single
            result = re.sub(r'\?{2,}', '?', result)
            # Multiple exclamation marks to single
            result = re.sub(r'!{2,}', '!', result)
            # Add space after punctuation if missing
            result = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', result)
        
        # Step 4: Normalize whitespace
        if self.config.normalize_whitespace:
            # Replace multiple whitespace with single space
            result = re.sub(r'\s+', ' ', result)
            result = result.strip()
        
        # Step 5: Remove boilerplate (empty lines and very short lines)
        if self.config.remove_boilerplate:
            lines = result.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) >= self.config.min_text_length:
                    cleaned_lines.append(line)
            result = '\n'.join(cleaned_lines)
        
        # Step 6: Remove stopwords if configured
        if self.config.remove_stopwords and hasattr(self, '_stopwords'):
            words = result.split()
            filtered_words = [word for word in words if word.lower() not in self._stopwords]
            result = ' '.join(filtered_words)
        
        return result

    async def normalize_text(self, text: str) -> str:
        """
        Normalize text by applying various cleaning operations.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        return await self.clean_text(text)
    
    def _load_stopwords(self) -> set:
        """Load stopwords for text cleaning."""
        # Basic English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'with', 'will', 'would', 'could', 'should', 'have',
            'had', 'been', 'being', 'do', 'does', 'did', 'doing', 'can', 'may',
            'might', 'must', 'shall', 'this', 'these', 'those', 'they', 'them',
            'their', 'there', 'where', 'when', 'who', 'what', 'which', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'now'
        }
        return stopwords 