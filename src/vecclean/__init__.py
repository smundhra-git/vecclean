"""
VecClean - Ultra-low latency text cleaning, deduplication, and vectorization pipeline.

A production-ready Python + C++ system for processing documents at scale.
Combines the flexibility of Python with the performance of C++ to deliver
maximum throughput for RAG pipelines.

Key Features:
- Ultra-low latency with C++ acceleration
- Multi-format document support (PDF, DOCX, HTML, TXT, PPTX, emails)
- Intelligent text cleaning and deduplication
- Token-aware chunking with configurable overlap
- Built-in embedding generation with model swapping
- No storage - returns processed chunks + vectors
- Production-ready with full test coverage

Example usage:

    # CLI
    $ vecclean document.pdf --output results.jsonl
    
    # API
    from vecclean.api.main import app
    
    # Direct pipeline usage
    from vecclean.core.pipeline import Pipeline
    pipeline = Pipeline()
    result = await pipeline.process_files(['document.pdf'])
"""

__version__ = "0.1.0" #TODO: Update this to the actual version
__author__ = "Shlok Mundhra"
__email__ = "shlokmundhra1111@gmail.com"
__license__ = "MIT"

# Core exports for public API
from .core.types import (
    CleanedChunk,
    ConfigurationError,
    Embedding,
    FileType,
    ProcessingError,
    ProcessingResult,
    ProcessingStatus,
    ProcessingStats,
)
from .core.pipeline import Pipeline
from .version import __version__

# Optional C++ backend
try:
    import vecclean_cpp
    
    _cpp_available = True
    __cpp_version__ = getattr(vecclean_cpp, "__version__", "unknown")
except ImportError:
    _cpp_available = False
    __cpp_version__ = None
    
    # C++ extension not available - this is expected for Python-only builds
    pass

# Public API
__all__ = [
    "clean_text",
    "normalize_whitespace",
    "standardize_punctuation",
    "get_capabilities",
    "version",
    # Core classes
    "Pipeline",
    "CleanedChunk",
    "Embedding",
    "FileType",
    "ProcessingStatus",
    "ProcessingResult",
    "ProcessingStats",
    # Exceptions
    "ConfigurationError",
    "ProcessingError",
    # Version
    "__version__",
]

def get_cpp_capabilities() -> dict[str, any] | None:
    """Get capabilities from the C++ extension, if available."""
    if is_cpp_available():
        return vecclean_cpp.get_capabilities()
    return None

def is_cpp_available() -> bool:
    """Check if C++ acceleration is available."""
    return _cpp_available


def get_version_info() -> dict[str, str]:
    """Get comprehensive version information."""
    import sys
    import platform
    from datetime import datetime
    
    version_info = {
        "vecclean": __version__,
        "cpp_backend": __cpp_version__ if _cpp_available else "not available",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    
    # Add build timestamp (fallback to current time if not available)
    try:
        # In a real build system, this would be injected during build
        version_info["build_timestamp"] = "2024-01-15T10:30:00Z"  # Placeholder
    except Exception:
        version_info["build_timestamp"] = datetime.now().isoformat()
    
    # Add git commit info (placeholder - would be injected during build)
    try:
        version_info["git_commit"] = "main"  # Placeholder - would be actual commit hash
    except Exception:
        version_info["git_commit"] = "unknown"
    
    # Add key dependencies
    dependencies = {}
    try:
        import fastapi
        dependencies["fastapi"] = fastapi.__version__
    except ImportError:
        pass
        
    try:
        import torch
        dependencies["torch"] = torch.__version__
    except ImportError:
        pass
        
    try:
        import numpy
        dependencies["numpy"] = numpy.__version__
    except ImportError:
        pass
        
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        pass
    
    version_info["dependencies"] = dependencies
    
    return version_info 