"""
Pydantic schemas for VecClean API.

Defines request and response models for all API endpoints with proper
validation, serialization, and documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: Optional[float] = Field(None, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracing")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Health status", regex="^(healthy|unhealthy|degraded)$")
    version: str = Field(..., description="Application version")
    cpp_backend: str = Field(..., description="C++ backend availability")
    timestamp: float = Field(..., description="Response timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Server uptime in seconds")
    checks: Optional[Dict[str, Any]] = Field(None, description="Detailed health checks")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class VersionResponse(BaseModel):
    """Version information response."""
    
    version: str = Field(..., description="Application version")
    cpp_backend_version: str = Field(..., description="C++ backend version")
    python_version: str = Field(..., description="Python backend version")
    build_info: Dict[str, Any] = Field(..., description="Build information")


class ChunkSchema(BaseModel):
    """Schema for a processed text chunk."""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Cleaned chunk text")
    text_hash: str = Field(..., description="Hash of the chunk text")
    chunk_index: int = Field(..., description="Index of chunk within document")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    start_page: Optional[int] = Field(None, description="Start page number")
    end_page: Optional[int] = Field(None, description="End page number")
    char_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")
    token_count: int = Field(..., description="Token count")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    source_document: Optional[Dict[str, Any]] = Field(None, description="Source document metadata")
    processing_stats: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")


class ProcessingStatsSchema(BaseModel):
    """Schema for processing statistics."""
    
    total_files: int = Field(..., description="Total files processed")
    total_size_bytes: int = Field(..., description="Total input size in bytes")
    successful_files: int = Field(..., description="Successfully processed files")
    failed_files: int = Field(..., description="Failed files")
    skipped_files: int = Field(..., description="Skipped files")
    total_chunks: int = Field(..., description="Total chunks generated")
    total_text_length: int = Field(..., description="Total text length")
    total_word_count: int = Field(..., description="Total word count")
    duplicate_sentences: int = Field(..., description="Duplicate sentences removed")
    duplicate_chunks: int = Field(..., description="Duplicate chunks removed")
    compression_ratio: float = Field(..., description="Text compression ratio")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    avg_processing_time_per_file: float = Field(..., description="Average time per file")
    avg_processing_time_per_mb: float = Field(..., description="Average time per MB")
    cpp_backend_used: bool = Field(..., description="Whether C++ backend was used")
    python_fallback_used: bool = Field(..., description="Whether Python fallback was used")


# Webhook payload schemas
class WebhookPayload(BaseModel):
    """Base webhook payload."""
    
    event_type: str = Field(..., description="Type of event that triggered the webhook")
    timestamp: float = Field(..., description="Event timestamp")
    request_id: str = Field(..., description="Original request ID")
    user_id: Optional[str] = Field(None, description="User ID if authenticated")


class ProcessingCompleteWebhook(WebhookPayload):
    """Webhook payload for processing completion."""
    
    event_type: str = Field("processing.completed", const=True)
    job_id: str = Field(..., description="Completed job ID")
    status: str = Field(..., description="Final processing status")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    processing_time: float = Field(..., description="Total processing time in seconds")
    download_url: Optional[str] = Field(None, description="URL to download results")


class ProcessingFailedWebhook(WebhookPayload):
    """Webhook payload for processing failure."""
    
    event_type: str = Field("processing.failed", const=True)
    job_id: str = Field(..., description="Failed job ID")
    error_message: str = Field(..., description="Error description")
    retry_count: int = Field(..., description="Number of retry attempts")


# Enhanced request schemas with custom validators
class CleanAndEmbedRequest(BaseModel):
    """Request schema for clean-and-embed endpoint with enhanced validation."""
    
    config_override: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configuration overrides for processing"
    )
    include_embeddings: bool = Field(
        True, 
        description="Whether to include embedding vectors in response"
    )
    include_metadata: bool = Field(
        True, 
        description="Whether to include metadata in response"
    )
    output_format: str = Field(
        "json", 
        description="Preferred output format",
        pattern="^(json|jsonl|parquet)$"
    )
    chunk_size: Optional[int] = Field(
        None, 
        description="Override chunk size",
        ge=100,
        le=2048
    )
    chunk_overlap: Optional[int] = Field(
        None, 
        description="Override chunk overlap",
        ge=0,
        le=500
    )
    embedding_model: Optional[str] = Field(
        None, 
        description="Override embedding model",
        pattern="^[a-zA-Z0-9/_-]+$"
    )
    deduplicate: Optional[bool] = Field(
        None, 
        description="Override deduplication setting"
    )
    priority: int = Field(
        0, 
        description="Processing priority (0=normal, 1=high, -1=low)",
        ge=-1,
        le=1
    )
    
    @validator("chunk_overlap")
    def overlap_less_than_size(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """Ensure overlap is less than chunk size."""
        if v is not None and "chunk_size" in values and values["chunk_size"] is not None:
            if v >= values["chunk_size"]:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    @validator("config_override")
    def validate_config_override(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate configuration override structure."""
        if v is None:
            return v
        
        allowed_keys = {
            "chunking.chunk_size", "chunking.overlap", "chunking.strategy",
            "embedding.model_name", "embedding.batch_size", "embedding.device",
            "cleaning.normalize_unicode", "cleaning.remove_stopwords",
            "dedup.enable_exact_dedup", "dedup.enable_fuzzy_dedup"
        }
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid config override key: {key}")
        
        return v
    
    class Config:
        """Enhanced schema configuration."""
        schema_extra = {
            "example": {
                "include_embeddings": True,
                "include_metadata": True,
                "output_format": "jsonl",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_model": "all-MiniLM-L6-v2",
                "deduplicate": True,
                "priority": 0,
                "config_override": {
                    "chunking.strategy": "sentence",
                    "cleaning.normalize_unicode": True
                }
            }
        }


class CleanAndEmbedResponse(BaseModel):
    """Response schema for clean-and-embed endpoint."""
    
    status: str = Field(..., description="Processing status")
    chunks: List[Dict[str, Any]] = Field(..., description="Processed chunks")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    warnings: Optional[List[str]] = Field(None, description="Processing warnings")
    errors: Optional[List[str]] = Field(None, description="Processing errors")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "status": "completed",
                "chunks": [
                    {
                        "chunk_id": "document_0",
                        "text": "This is a sample cleaned text chunk.",
                        "text_hash": "abc123...",
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": 37,
                        "char_count": 37,
                        "word_count": 7,
                        "token_count": 8,
                        "embedding": [0.1, 0.2, 0.3],
                        "embedding_model": "all-MiniLM-L6-v2"
                    }
                ],
                "statistics": {
                    "total_files": 1,
                    "successful_files": 1,
                    "total_chunks": 1,
                    "total_processing_time": 0.5,
                    "cpp_backend_used": True
                },
                "metadata": {
                    "processing_time_seconds": 0.5,
                    "files_processed": 1,
                    "backend_used": "cpp"
                }
            }
        }


class TextCleaningRequest(BaseModel):
    """Request schema for text cleaning endpoint."""
    
    text: str = Field(..., description="Text to clean", max_length=1000000)
    config_override: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configuration overrides"
    )
    normalize_unicode: Optional[bool] = Field(None, description="Normalize Unicode")
    normalize_whitespace: Optional[bool] = Field(None, description="Normalize whitespace")
    remove_stopwords: Optional[bool] = Field(None, description="Remove stopwords")
    language: Optional[str] = Field(None, description="Text language")


class TextCleaningResponse(BaseModel):
    """Response schema for text cleaning endpoint."""
    
    original_text: str = Field(..., description="Original input text")
    cleaned_text: str = Field(..., description="Cleaned output text")
    statistics: Dict[str, Any] = Field(..., description="Cleaning statistics")
    warnings: Optional[List[str]] = Field(None, description="Processing warnings")


class BatchProcessingRequest(BaseModel):
    """Request schema for batch processing."""
    
    job_id: Optional[str] = Field(None, description="Optional job identifier")
    files: List[str] = Field(..., description="List of file URLs or paths")
    config: Optional[Dict[str, Any]] = Field(None, description="Processing configuration")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion")
    priority: int = Field(0, description="Job priority", ge=0, le=10)
    
    @validator("files")
    def validate_files(cls, v: List[str]) -> List[str]:
        """Validate file list."""
        if len(v) == 0:
            raise ValueError("At least one file must be provided")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 files per batch")
        return v


class BatchProcessingResponse(BaseModel):
    """Response schema for batch processing."""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    submitted_at: float = Field(..., description="Submission timestamp")
    estimated_completion: Optional[float] = Field(None, description="Estimated completion time")
    files_queued: int = Field(..., description="Number of files queued")
    progress_url: str = Field(..., description="URL to check progress")


class BatchStatusResponse(BaseModel):
    """Response schema for batch status."""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    files_processed: int = Field(..., description="Files processed")
    files_total: int = Field(..., description="Total files")
    files_failed: int = Field(..., description="Files failed")
    started_at: Optional[float] = Field(None, description="Start timestamp")
    completed_at: Optional[float] = Field(None, description="Completion timestamp")
    results_url: Optional[str] = Field(None, description="URL to download results")
    error: Optional[str] = Field(None, description="Error message if failed")


class ConfigSchema(BaseModel):
    """Schema for configuration display."""
    
    processing: Dict[str, Any] = Field(..., description="Processing configuration")
    chunking: Dict[str, Any] = Field(..., description="Chunking configuration")
    cleaning: Dict[str, Any] = Field(..., description="Cleaning configuration")
    dedup: Dict[str, Any] = Field(..., description="Deduplication configuration")
    embedding: Dict[str, Any] = Field(..., description="Embedding configuration")
    output: Dict[str, Any] = Field(..., description="Output configuration")


class CapabilitiesResponse(BaseModel):
    """Response schema for capabilities endpoint."""
    
    version_info: Dict[str, str] = Field(..., description="Version information")
    cpp_available: bool = Field(..., description="C++ backend availability")
    cpp_capabilities: Optional[Dict[str, Any]] = Field(None, description="C++ capabilities")
    supported_formats: List[str] = Field(..., description="Supported file formats")
    embedding_models: List[str] = Field(..., description="Available embedding models")
    max_file_size: int = Field(..., description="Maximum file size in bytes")
    max_batch_size: int = Field(..., description="Maximum batch size")


# Request/Response unions for OpenAPI documentation
RequestSchema = Union[
    CleanAndEmbedRequest,
    TextCleaningRequest,
    BatchProcessingRequest,
]

ResponseSchema = Union[
    CleanAndEmbedResponse,
    TextCleaningResponse,
    BatchProcessingResponse,
    BatchStatusResponse,
    HealthResponse,
    VersionResponse,
    CapabilitiesResponse,
    ErrorResponse,
]


# Schema inheritance for common patterns
class BaseResponseSchema(BaseModel):
    """Base response schema with common fields."""
    
    status: str = Field(..., description="Response status")
    timestamp: float = Field(..., description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    
    class Config:
        """Base configuration for all responses."""
        allow_population_by_field_name = True


class BaseProcessingResponse(BaseResponseSchema):
    """Base processing response with common processing fields."""
    
    processing_time: float = Field(..., description="Processing time in seconds")
    backend_used: str = Field(..., description="Backend used for processing")
    warnings: Optional[List[str]] = Field(None, description="Processing warnings")
    errors: Optional[List[str]] = Field(None, description="Processing errors")


# Version-aware schemas
class V1CleanAndEmbedResponse(BaseProcessingResponse):
    """Version 1 API response for clean-and-embed endpoint."""
    
    api_version: str = Field("v1", const=True)
    chunks: List[Dict[str, Any]] = Field(..., description="Processed chunks")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")


class V2CleanAndEmbedResponse(BaseProcessingResponse):
    """Version 2 API response with enhanced features."""
    
    api_version: str = Field("v2", const=True)
    chunks: List[Dict[str, Any]] = Field(..., description="Processed chunks")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    quality_metrics: Optional[Dict[str, float]] = Field(None, description="Quality assessment metrics")
    optimization_suggestions: Optional[List[str]] = Field(None, description="Optimization suggestions")


# Enhanced validation with custom business rules
class FileUploadValidator:
    """Custom validator for file uploads."""
    
    @staticmethod
    def validate_file_batch(files: List[Any], max_files: int = 100, max_total_size: int = 1024**3) -> bool:
        """Validate a batch of uploaded files."""
        if len(files) > max_files:
            raise ValueError(f"Too many files: {len(files)} > {max_files}")
        
        total_size = sum(getattr(f, 'size', 0) for f in files)
        if total_size > max_total_size:
            raise ValueError(f"Total size too large: {total_size} bytes")
        
        return True
    
    @staticmethod
    def validate_file_types(files: List[Any], allowed_types: set) -> bool:
        """Validate file types in batch."""
        for file in files:
            if hasattr(file, 'filename') and file.filename:
                ext = file.filename.split('.')[-1].lower()
                if ext not in allowed_types:
                    raise ValueError(f"Unsupported file type: .{ext}")
        
        return True


# Dynamic schema generation helper
class SchemaGenerator:
    """Generate schemas dynamically based on configuration."""
    
    @staticmethod
    def generate_config_schema(config_dict: Dict[str, Any]) -> BaseModel:
        """Generate a Pydantic schema from configuration."""
        # This would dynamically create schemas based on config structure
        # Implementation would inspect config and create appropriate Field definitions
        pass
    
    @staticmethod
    def generate_response_schema(
        include_embeddings: bool = True,
        include_metadata: bool = True,
        api_version: str = "v1"
    ) -> BaseModel:
        """Generate response schema based on request parameters."""
        # Dynamic schema generation based on request parameters
        pass


# Comprehensive examples for all schemas
SCHEMA_EXAMPLES = {
    "health_response": {
        "status": "healthy",
        "version": "0.1.0",
        "cpp_backend": "available", 
        "timestamp": 1672531200.0,
        "uptime_seconds": 3600.0,
        "checks": {
            "database": "connected",
            "embedding_model": "loaded",
            "memory_usage": "normal"
        }
    },
    "version_response": {
        "version": "0.1.0",
        "cpp_backend_version": "0.1.0",
        "python_version": "3.9.0",
        "build_info": {
            "cpp_available": True,
            "build_timestamp": "2024-01-15T10:30:00Z",
            "git_commit": "abc123"
        }
    },
    "processing_response": {
        "status": "completed",
        "chunks": [
            {
                "chunk_id": "doc1_chunk_0",
                "text": "This is a sample text chunk.",
                "embedding": [0.1, 0.2, -0.3],
                "metadata": {"confidence": 0.95}
            }
        ],
        "statistics": {
            "total_files": 1,
            "processing_time": 2.5,
            "chunks_generated": 1
        }
    }
}


# Future enhancement placeholders (v2 features)
class MLQualityMetrics(BaseModel):
    """ML-based quality assessment metrics."""
    
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Text coherence score")
    readability_score: float = Field(..., ge=0.0, le=1.0, description="Text readability score")  
    information_density: float = Field(..., ge=0.0, le=1.0, description="Information density score")
    embedding_quality: float = Field(..., ge=0.0, le=1.0, description="Embedding quality score")


class AdvancedProcessingOptions(BaseModel):
    """Advanced processing options for v2 API."""
    
    enable_ml_quality_assessment: bool = Field(False, description="Enable ML quality assessment")
    custom_preprocessing_pipeline: Optional[List[str]] = Field(None, description="Custom preprocessing steps")
    target_embedding_dimensions: Optional[int] = Field(None, ge=128, le=2048, description="Target embedding dimensions")
    enable_adaptive_chunking: bool = Field(False, description="Enable adaptive chunking based on content")


# Export all schemas for OpenAPI generation
__all__ = [
    "ErrorResponse", "HealthResponse", "VersionResponse", "ChunkSchema",
    "ProcessingStatsSchema", "CleanAndEmbedRequest", "CleanAndEmbedResponse",
    "TextCleaningRequest", "TextCleaningResponse", "BatchProcessingRequest",
    "BatchProcessingResponse", "BatchStatusResponse", "ConfigSchema",
    "CapabilitiesResponse", "WebhookPayload", "ProcessingCompleteWebhook",
    "ProcessingFailedWebhook", "V1CleanAndEmbedResponse", "V2CleanAndEmbedResponse",
    "FileUploadValidator", "SchemaGenerator", "SCHEMA_EXAMPLES"
] 