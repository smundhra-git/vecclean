"""
I/O utilities for VecClean.

Provides utilities for reading and writing various file formats
including JSONL, Parquet, and JSON with compression support,
streaming capabilities, and cloud storage integration.
"""

from __future__ import annotations

import asyncio
import brotli
import gzip
import json
import lz4.frame
import zstandard as zstd
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Union, Iterator, AsyncIterator, Optional, BinaryIO, TextIO
from urllib.parse import urlparse
import aiofiles
import aiofiles.os

import pandas as pd


def ensure_directory_exists(path: Union[str, Path]) -> None:
    """Ensure that the directory for a given path exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """Read file as bytes."""
        pass
    
    @abstractmethod
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to file."""
        pass
    
    @abstractmethod
    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file as text."""
        pass
    
    @abstractmethod
    async def write_text(self, path: str, text: str, encoding: str = 'utf-8') -> None:
        """Write text to file."""
        pass
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete file."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    async def read_bytes(self, path: str) -> bytes:
        """Read file as bytes."""
        async with aiofiles.open(path, 'rb') as f:
            return await f.read()
    
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, 'wb') as f:
            await f.write(data)
    
    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file as text."""
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            return await f.read()
    
    async def write_text(self, path: str, text: str, encoding: str = 'utf-8') -> None:
        """Write text to file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, 'w', encoding=encoding) as f:
            await f.write(text)
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return await aiofiles.os.path.exists(path)
    
    async def delete(self, path: str) -> None:
        """Delete file."""
        await aiofiles.os.remove(path)


class S3StorageBackend(StorageBackend):
    """Amazon S3 storage backend."""
    
    def __init__(self, bucket: str, region: str = 'us-east-1', 
                 access_key: Optional[str] = None, secret_key: Optional[str] = None):
        """Initialize S3 backend."""
        self.bucket = bucket
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self._client = None
    
    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import aioboto3
                session = aioboto3.Session()
                self._client = session.client(
                    's3',
                    region_name=self.region,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key
                )
            except ImportError:
                raise ImportError("aioboto3 is required for S3 support. Install with: pip install aioboto3")
        return self._client
    
    async def read_bytes(self, path: str) -> bytes:
        """Read file as bytes from S3."""
        client = await self._get_client()
        async with client as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=path)
            return await response['Body'].read()
    
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to S3."""
        client = await self._get_client()
        async with client as s3:
            await s3.put_object(Bucket=self.bucket, Key=path, Body=data)
    
    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file as text from S3."""
        data = await self.read_bytes(path)
        return data.decode(encoding)
    
    async def write_text(self, path: str, text: str, encoding: str = 'utf-8') -> None:
        """Write text to S3."""
        data = text.encode(encoding)
        await self.write_bytes(path, data)
    
    async def exists(self, path: str) -> bool:
        """Check if file exists in S3."""
        client = await self._get_client()
        async with client as s3:
            try:
                await s3.head_object(Bucket=self.bucket, Key=path)
                return True
            except s3.exceptions.NoSuchKey:
                return False
    
    async def delete(self, path: str) -> None:
        """Delete file from S3."""
        client = await self._get_client()
        async with client as s3:
            await s3.delete_object(Bucket=self.bucket, Key=path)


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(self, bucket: str, credentials_path: Optional[str] = None):
        """Initialize GCS backend."""
        self.bucket = bucket
        self.credentials_path = credentials_path
        self._client = None
    
    async def _get_client(self):
        """Get or create GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage
                if self.credentials_path:
                    self._client = storage.Client.from_service_account_json(self.credentials_path)
                else:
                    self._client = storage.Client()
            except ImportError:
                raise ImportError("google-cloud-storage is required for GCS support. Install with: pip install google-cloud-storage")
        return self._client
    
    async def read_bytes(self, path: str) -> bytes:
        """Read file as bytes from GCS."""
        client = await self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(path)
        return blob.download_as_bytes()
    
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to GCS."""
        client = await self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(path)
        blob.upload_from_string(data)
    
    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file as text from GCS."""
        data = await self.read_bytes(path)
        return data.decode(encoding)
    
    async def write_text(self, path: str, text: str, encoding: str = 'utf-8') -> None:
        """Write text to GCS."""
        data = text.encode(encoding)
        await self.write_bytes(path, data)
    
    async def exists(self, path: str) -> bool:
        """Check if file exists in GCS."""
        client = await self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(path)
        return blob.exists()
    
    async def delete(self, path: str) -> None:
        """Delete file from GCS."""
        client = await self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(path)
        blob.delete()


class AzureStorageBackend(StorageBackend):
    """Azure Blob Storage backend."""
    
    def __init__(self, container: str, account_name: str, account_key: Optional[str] = None):
        """Initialize Azure backend."""
        self.container = container
        self.account_name = account_name
        self.account_key = account_key
        self._client = None
    
    async def _get_client(self):
        """Get or create Azure client."""
        if self._client is None:
            try:
                from azure.storage.blob.aio import BlobServiceClient
                if self.account_key:
                    account_url = f"https://{self.account_name}.blob.core.windows.net"
                    self._client = BlobServiceClient(account_url=account_url, credential=self.account_key)
                else:
                    # Use default credential
                    from azure.identity.aio import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    account_url = f"https://{self.account_name}.blob.core.windows.net"
                    self._client = BlobServiceClient(account_url=account_url, credential=credential)
            except ImportError:
                raise ImportError("azure-storage-blob is required for Azure support. Install with: pip install azure-storage-blob azure-identity")
        return self._client
    
    async def read_bytes(self, path: str) -> bytes:
        """Read file as bytes from Azure."""
        client = await self._get_client()
        blob_client = client.get_blob_client(container=self.container, blob=path)
        stream = await blob_client.download_blob()
        return await stream.readall()
    
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to Azure."""
        client = await self._get_client()
        blob_client = client.get_blob_client(container=self.container, blob=path)
        await blob_client.upload_blob(data, overwrite=True)
    
    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file as text from Azure."""
        data = await self.read_bytes(path)
        return data.decode(encoding)
    
    async def write_text(self, path: str, text: str, encoding: str = 'utf-8') -> None:
        """Write text to Azure."""
        data = text.encode(encoding)
        await self.write_bytes(path, data)
    
    async def exists(self, path: str) -> bool:
        """Check if file exists in Azure."""
        client = await self._get_client()
        blob_client = client.get_blob_client(container=self.container, blob=path)
        return await blob_client.exists()
    
    async def delete(self, path: str) -> None:
        """Delete file from Azure."""
        client = await self._get_client()
        blob_client = client.get_blob_client(container=self.container, blob=path)
        await blob_client.delete_blob()


class CompressionHandler:
    """Handles various compression formats."""
    
    @staticmethod
    def detect_compression(path: str) -> Optional[str]:
        """Detect compression format from file extension."""
        path_lower = path.lower()
        if path_lower.endswith('.gz'):
            return 'gzip'
        elif path_lower.endswith('.bz2'):
            return 'bzip2'
        elif path_lower.endswith('.br'):
            return 'brotli'
        elif path_lower.endswith('.lz4'):
            return 'lz4'
        elif path_lower.endswith('.zst'):
            return 'zstd'
        return None
    
    @staticmethod
    def compress(data: bytes, compression: str) -> bytes:
        """Compress data using specified format."""
        if compression == 'gzip':
            return gzip.compress(data)
        elif compression == 'bzip2':
            import bz2
            return bz2.compress(data)
        elif compression == 'brotli':
            return brotli.compress(data)
        elif compression == 'lz4':
            return lz4.frame.compress(data)
        elif compression == 'zstd':
            return zstd.compress(data)
        else:
            raise ValueError(f"Unsupported compression format: {compression}")
    
    @staticmethod
    def decompress(data: bytes, compression: str) -> bytes:
        """Decompress data using specified format."""
        if compression == 'gzip':
            return gzip.decompress(data)
        elif compression == 'bzip2':
            import bz2
            return bz2.decompress(data)
        elif compression == 'brotli':
            return brotli.decompress(data)
        elif compression == 'lz4':
            return lz4.frame.decompress(data)
        elif compression == 'zstd':
            return zstd.decompress(data)
        else:
            raise ValueError(f"Unsupported compression format: {compression}")


class FileFormatDetector:
    """Detects file format from content or extension."""
    
    @staticmethod
    def detect_format(path: str, content: Optional[bytes] = None) -> str:
        """Detect file format from path and optionally content."""
        path_lower = path.lower()
        
        # Remove compression extension to get actual format
        for ext in ['.gz', '.bz2', '.br', '.lz4', '.zst']:
            if path_lower.endswith(ext):
                path_lower = path_lower[:-len(ext)]
                break
        
        # Detect by extension
        if path_lower.endswith('.json'):
            return 'json'
        elif path_lower.endswith('.jsonl') or path_lower.endswith('.ndjson'):
            return 'jsonl'
        elif path_lower.endswith('.parquet'):
            return 'parquet'
        elif path_lower.endswith('.csv'):
            return 'csv'
        elif path_lower.endswith('.tsv'):
            return 'tsv'
        
        # Try content-based detection if available
        if content:
            try:
                # Try to parse as JSON
                json.loads(content.decode('utf-8'))
                return 'json'
            except:
                pass
            
            # Check for JSONL (multiple JSON objects)
            try:
                lines = content.decode('utf-8').strip().split('\n')
                if len(lines) > 1:
                    for line in lines[:3]:  # Check first few lines
                        if line.strip():
                            json.loads(line)
                    return 'jsonl'
            except:
                pass
        
        return 'unknown'


class StreamingReader:
    """Streaming reader for large files."""
    
    def __init__(self, storage: StorageBackend, chunk_size: int = 64 * 1024):
        """Initialize streaming reader."""
        self.storage = storage
        self.chunk_size = chunk_size
    
    async def read_text_chunks(self, path: str, encoding: str = 'utf-8') -> AsyncIterator[str]:
        """Read file in text chunks."""
        # For now, we'll implement a basic version
        # In production, this would use proper streaming
        content = await self.storage.read_text(path, encoding)
        
        # Split into chunks
        for i in range(0, len(content), self.chunk_size):
            yield content[i:i + self.chunk_size]
    
    async def read_jsonl_chunks(self, path: str, chunk_lines: int = 1000) -> AsyncIterator[List[Dict[str, Any]]]:
        """Read JSONL file in chunks of lines."""
        # For basic implementation, read entire file
        # In production, this would use line-by-line streaming
        try:
            content = await self.storage.read_text(path)
            lines = content.strip().split('\n')
            
            current_chunk = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        current_chunk.append(json.loads(line))
                        if len(current_chunk) >= chunk_lines:
                            yield current_chunk
                            current_chunk = []
                    except json.JSONDecodeError:
                        continue
            
            # Yield remaining items
            if current_chunk:
                yield current_chunk
                
        except Exception:
            # Return empty generator if file can't be read
            return
            yield  # Make this an async generator


class UniversalIOManager:
    """Universal I/O manager with support for multiple storage backends and formats."""
    
    def __init__(self):
        """Initialize I/O manager."""
        self.backends: Dict[str, StorageBackend] = {
            'local': LocalStorageBackend(),
        }
        self.compression_handler = CompressionHandler()
        self.format_detector = FileFormatDetector()
    
    def register_backend(self, scheme: str, backend: StorageBackend) -> None:
        """Register a storage backend for a URL scheme."""
        self.backends[scheme] = backend
    
    def _get_backend_and_path(self, path: Union[str, Path]) -> tuple[StorageBackend, str]:
        """Get appropriate backend and normalized path."""
        path_str = str(path)
        
        # Parse URL to determine backend
        if '://' in path_str:
            parsed = urlparse(path_str)
            scheme = parsed.scheme
            
            if scheme == 's3':
                # s3://bucket/key format
                if 's3' not in self.backends:
                    bucket = parsed.netloc
                    self.backends['s3'] = S3StorageBackend(bucket)
                return self.backends['s3'], parsed.path.lstrip('/')
            elif scheme == 'gs':
                # gs://bucket/key format
                if 'gs' not in self.backends:
                    bucket = parsed.netloc
                    self.backends['gs'] = GCSStorageBackend(bucket)
                return self.backends['gs'], parsed.path.lstrip('/')
            elif scheme == 'azure':
                # azure://container/blob format
                if 'azure' not in self.backends:
                    container = parsed.netloc
                    account_name = 'default'  # Should be configured
                    self.backends['azure'] = AzureStorageBackend(container, account_name)
                return self.backends['azure'], parsed.path.lstrip('/')
            elif scheme in self.backends:
                return self.backends[scheme], parsed.path
        
        # Default to local filesystem
        return self.backends['local'], path_str
    
    async def read_with_compression(self, path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Read file with automatic compression detection."""
        backend, normalized_path = self._get_backend_and_path(path)
        
        # Detect compression
        compression = self.compression_handler.detect_compression(normalized_path)
        
        if compression:
            # Read as bytes and decompress
            data = await backend.read_bytes(normalized_path)
            decompressed = self.compression_handler.decompress(data, compression)
            return decompressed.decode(encoding)
        else:
            # Read as text directly
            return await backend.read_text(normalized_path, encoding)
    
    async def write_with_compression(self, path: Union[str, Path], content: str, 
                                   compression: Optional[str] = None, encoding: str = 'utf-8') -> None:
        """Write file with optional compression."""
        backend, normalized_path = self._get_backend_and_path(path)
        
        # Auto-detect compression if not specified
        if compression is None:
            compression = self.compression_handler.detect_compression(normalized_path)
        
        if compression:
            # Compress and write as bytes
            data = content.encode(encoding)
            compressed = self.compression_handler.compress(data, compression)
            await backend.write_bytes(normalized_path, compressed)
        else:
            # Write as text directly
            await backend.write_text(normalized_path, content, encoding)


# Global I/O manager instance
io_manager = UniversalIOManager()


# Enhanced file I/O functions
async def write_json(data: Dict[str, Any], file_path: Union[str, Path], 
                    compression: Optional[str] = None) -> None:
    """
    Write data to JSON file with optional compression.
    
    Args:
        data: Data to write
        file_path: Output file path (supports local, S3, GCS, Azure)
        compression: Optional compression format (gzip, brotli, lz4, zstd)
    """
    content = json.dumps(data, indent=2, ensure_ascii=False)
    await io_manager.write_with_compression(file_path, content, compression)


async def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path], 
                     compression: Optional[str] = None) -> None:
    """
    Write data to JSONL file with optional compression.
    
    Args:
        data: List of dictionaries to write
        file_path: Output file path (supports local, S3, GCS, Azure)
        compression: Optional compression format
    """
    lines = [json.dumps(item, ensure_ascii=False) for item in data]
    content = '\n'.join(lines) + '\n'
    await io_manager.write_with_compression(file_path, content, compression)


async def write_parquet(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Write data to Parquet file.
    
    Args:
        data: List of dictionaries to write
        file_path: Output file path
    """
    if not data:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(data)
    
    backend, normalized_path = io_manager._get_backend_and_path(file_path)
    
    if isinstance(backend, LocalStorageBackend):
        # For local files, use pandas directly
        Path(normalized_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(normalized_path, index=False)
    else:
        # For cloud storage, write to bytes first
        import io as iolib
        buffer = iolib.BytesIO()
        df.to_parquet(buffer, index=False)
        await backend.write_bytes(normalized_path, buffer.getvalue())


async def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read data from JSON file with automatic compression detection.
    
    Args:
        file_path: Input file path (supports local, S3, GCS, Azure)
        
    Returns:
        Loaded data
    """
    content = await io_manager.read_with_compression(file_path)
    return json.loads(content)


async def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read data from JSONL file with automatic compression detection.
    
    Args:
        file_path: Input file path (supports local, S3, GCS, Azure)
        
    Returns:
        List of loaded dictionaries
    """
    content = await io_manager.read_with_compression(file_path)
    data = []
    for line in content.strip().split('\n'):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


async def read_jsonl_stream(file_path: Union[str, Path], 
                           chunk_lines: int = 1000) -> AsyncIterator[List[Dict[str, Any]]]:
    """
    Stream read JSONL file in chunks.
    
    Args:
        file_path: Input file path
        chunk_lines: Number of lines per chunk
        
    Yields:
        Chunks of loaded dictionaries
    """
    backend, normalized_path = io_manager._get_backend_and_path(file_path)
    reader = StreamingReader(backend)
    
    async for chunk in reader.read_jsonl_chunks(normalized_path, chunk_lines):
        yield chunk


def detect_file_format(file_path: Union[str, Path]) -> str:
    """
    Detect file format from path.
    
    Args:
        file_path: File path to analyze
        
    Returns:
        Detected format (json, jsonl, parquet, csv, tsv, unknown)
    """
    return io_manager.format_detector.detect_format(str(file_path))


# Legacy sync functions for backward compatibility
def write_json_sync(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Write data to JSON file (synchronous)."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl_sync(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Write data to JSONL file (synchronous)."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_parquet_sync(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Write data to Parquet file (synchronous)."""
    if not data:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(data)
    
    df.to_parquet(file_path, index=False)


def read_json_sync(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read data from JSON file (synchronous)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl_sync(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read data from JSONL file (synchronous)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data 