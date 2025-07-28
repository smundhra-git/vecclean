"""
Core type definitions for VecClean.

This module defines the main data structures, protocols, and type annotations
used throughout the VecClean pipeline.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
from numpy.typing import NDArray


# Type aliases for better readability
Embedding = NDArray[np.float32]
FilePath = Union[str, Path]
JsonDict = Dict[str, Any]


class FileType(enum.Enum):
    """Supported file types for processing."""
    
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    TXT = "txt"
    EMAIL = "email"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "md"
    UNKNOWN = "unknown"


class ProcessingStatus(enum.Enum):
    """Status of processing operation."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    
    # Basic file information
    filename: str
    file_type: FileType
    file_size: int
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    
    # Document-specific metadata
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    
    # Processing metadata
    processing_timestamp: float = 0.0
    processing_duration: float = 0.0
    
    # Additional custom metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary for serialization."""
        return {
            "filename": self.filename,
            "file_type": self.file_type.value,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "language": self.language,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "processing_timestamp": self.processing_timestamp,
            "processing_duration": self.processing_duration,
            "extra": self.extra,
        }


@dataclass
class CleanedChunk:
    """A cleaned and processed text chunk with embeddings."""
    
    # Unique identifier for this chunk
    chunk_id: str
    
    # Core content
    text: str
    text_hash: str  # Hash of the cleaned text for deduplication
    
    # Embeddings
    embedding: Optional[Embedding] = None
    embedding_model: Optional[str] = None
    
    # Chunk positioning and metadata
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    
    # Text statistics
    char_count: int = 0
    word_count: int = 0
    token_count: int = 0
    
    # Source information
    source_document: Optional[DocumentMetadata] = None
    
    # Processing metadata
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Calculate derived fields after initialization."""
        if self.char_count == 0:
            self.char_count = len(self.text)
        if self.word_count == 0:
            self.word_count = len(self.text.split())
    
    def to_dict(self, include_embedding: bool = True) -> JsonDict:
        """Convert to dictionary for serialization."""
        result = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "text_hash": self.text_hash,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "token_count": self.token_count,
            "embedding_model": self.embedding_model,
            "source_document": self.source_document.to_dict() if self.source_document else None,
            "processing_stats": self.processing_stats,
        }
        
        if include_embedding and self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
        
        return result


@dataclass 
class ProcessingStats:
    """Statistics from processing operations."""
    
    # Input statistics
    total_files: int = 0
    total_size_bytes: int = 0
    
    # Processing results
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    # Content statistics
    total_chunks: int = 0
    total_text_length: int = 0
    total_word_count: int = 0
    
    # Deduplication statistics
    duplicate_sentences: int = 0
    duplicate_chunks: int = 0
    compression_ratio: float = 0.0
    
    # Performance metrics
    total_processing_time: float = 0.0
    avg_processing_time_per_file: float = 0.0
    avg_processing_time_per_mb: float = 0.0
    
    # C++ backend usage
    cpp_backend_used: bool = False
    python_fallback_used: bool = False
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "total_chunks": self.total_chunks,
            "total_text_length": self.total_text_length,
            "total_word_count": self.total_word_count,
            "duplicate_sentences": self.duplicate_sentences,
            "duplicate_chunks": self.duplicate_chunks,
            "compression_ratio": self.compression_ratio,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time_per_file": self.avg_processing_time_per_file,
            "avg_processing_time_per_mb": self.avg_processing_time_per_mb,
            "cpp_backend_used": self.cpp_backend_used,
            "python_fallback_used": self.python_fallback_used,
        }


@dataclass
class ProcessingResult:
    """Result of processing one or more documents."""
    
    # Processing status
    status: ProcessingStatus
    
    # Processed chunks
    chunks: List[CleanedChunk] = field(default_factory=list)
    
    # Processing statistics
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_timestamp: float = 0.0
    config_used: Optional[JsonDict] = None
    
    def to_dict(self, include_embeddings: bool = True) -> JsonDict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "chunks": [chunk.to_dict(include_embeddings) for chunk in self.chunks],
            "stats": self.stats.to_dict(),
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_timestamp": self.processing_timestamp,
            "config_used": self.config_used,
        }


# Protocol definitions for extensibility

class FileIngester(Protocol):
    """Protocol for file ingestion implementations."""
    
    def can_handle(self, file_path: FilePath) -> bool:
        """Check if this ingester can handle the given file."""
        ...
    
    async def extract_text(self, file_path: FilePath) -> tuple[str, DocumentMetadata]:
        """Extract text and metadata from the file."""
        ...


class TextCleaner(Protocol):
    """Protocol for text cleaning implementations."""
    
    async def clean_text(self, text: str, config: JsonDict) -> str:
        """Clean and normalize text."""
        ...
    
    async def remove_stopwords(self, text: str, stopwords: set[str]) -> str:
        """Remove stopwords from text."""
        ...


class TextDeduplicator(Protocol):
    """Protocol for text deduplication implementations."""
    
    async def deduplicate_sentences(self, sentences: List[str], threshold: float) -> List[str]:
        """Remove duplicate sentences."""
        ...
    
    async def deduplicate_chunks(self, chunks: List[str], threshold: float) -> List[str]:
        """Remove duplicate chunks."""
        ...


class TextChunker(Protocol):
    """Protocol for text chunking implementations."""
    
    async def chunk_text(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int,
        respect_boundaries: bool = True
    ) -> List[tuple[str, int, int]]:
        """Split text into chunks with metadata."""
        ...


class EmbeddingModel(Protocol):
    """Protocol for embedding model implementations."""
    
    async def encode(self, texts: List[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts."""
        ...
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        ...
    
    def get_model_name(self) -> str:
        """Get the model name/identifier."""
        ...


class OutputWriter(Protocol):
    """Protocol for output writing implementations."""
    
    async def write_results(
        self, 
        results: ProcessingResult, 
        output_path: FilePath,
        format_options: Optional[JsonDict] = None
    ) -> None:
        """Write processing results to file."""
        ...


# Exception types for error handling

class VecCleanError(Exception):
    """Base exception for VecClean errors."""
    pass


class ConfigurationError(VecCleanError):
    """Error in configuration."""
    pass


class ProcessingError(VecCleanError):
    """Error during document processing."""
    pass


class IngestionError(ProcessingError):
    """Error during file ingestion."""
    pass


class CleaningError(ProcessingError):
    """Error during text cleaning."""
    pass


class ChunkingError(ProcessingError):
    """Error during text chunking."""
    pass


class EmbeddingError(ProcessingError):
    """Error during embedding generation."""
    pass


class OutputError(VecCleanError):
    """Error during output writing."""
    pass 