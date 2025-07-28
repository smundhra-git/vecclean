"""
File type detection for VecClean.

Detects file types and routes to appropriate processors.
Implements content-based detection, magic numbers, and email support.
"""

from __future__ import annotations

import email
import mimetypes
import struct
import zipfile
import tarfile
import gzip
import bz2
from pathlib import Path
from typing import Optional, Dict, Tuple, BinaryIO

from vecclean.core.types import FileType, FilePath


# Magic number signatures for binary file detection
MAGIC_NUMBERS = {
    # PDF files
    b'%PDF': FileType.PDF,
    
    # Microsoft Office formats (ZIP-based)
    b'PK\x03\x04': 'zip_based',  # Will need further inspection
    
    # Images
    b'\xFF\xD8\xFF': FileType.UNKNOWN,  # JPEG
    b'\x89PNG\r\n\x1a\n': FileType.UNKNOWN,  # PNG
    b'GIF87a': FileType.UNKNOWN,  # GIF87a
    b'GIF89a': FileType.UNKNOWN,  # GIF89a
    
    # Archives
    b'\x1f\x8b': 'gzip',  # GZIP
    b'BZ': 'bzip2',  # BZIP2
    b'7z\xbc\xaf\x27\x1c': 'seven_zip',  # 7-Zip
    b'Rar!\x1a\x07\x00': 'rar',  # RAR
    
    # Text files (BOM markers)
    b'\xef\xbb\xbf': 'utf8_bom',  # UTF-8 BOM
    b'\xff\xfe': 'utf16_le',  # UTF-16 LE BOM
    b'\xfe\xff': 'utf16_be',  # UTF-16 BE BOM
}

# ZIP-based Office formats central directory signatures
OFFICE_SIGNATURES = {
    b'word/': FileType.DOCX,
    b'ppt/': FileType.PPTX,
    b'xl/': FileType.UNKNOWN,  # Excel - not currently supported
}

# Email header patterns
EMAIL_HEADERS = {
    'from:', 'to:', 'subject:', 'date:', 'message-id:', 'received:',
    'return-path:', 'delivered-to:', 'x-original-to:', 'content-type:'
}


class FileTypeDetector:
    """
    Comprehensive file type detector for routing to appropriate processors.
    
    Supports:
    - Extension-based detection
    - MIME type detection
    - Magic number detection for binary files
    - Content-based detection for text files
    - Email message detection
    - Archive file detection
    """
    
    def __init__(self) -> None:
        """Initialize detector with magic number cache."""
        self._magic_cache: Dict[str, FileType] = {}
        
        # Initialize mimetypes
        mimetypes.init()
    
    async def detect_file_type(self, file_path: FilePath) -> FileType:
        """
        Detect file type using multiple detection methods.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected file type
        """
        file_path = Path(file_path)
        
        # Check cache first
        cache_key = f"{file_path.stat().st_mtime}_{file_path.stat().st_size}_{file_path}"
        if cache_key in self._magic_cache:
            return self._magic_cache[cache_key]
        
        # Try extension-based detection first (fastest)
        file_type = self._detect_by_extension(file_path)
        if file_type != FileType.UNKNOWN:
            self._magic_cache[cache_key] = file_type
            return file_type
        
        # Try magic number detection for binary files
        file_type = await self._detect_by_magic_numbers(file_path)
        if file_type != FileType.UNKNOWN:
            self._magic_cache[cache_key] = file_type
            return file_type
        
        # Try content-based detection for text files
        file_type = await self._detect_by_content(file_path)
        if file_type != FileType.UNKNOWN:
            self._magic_cache[cache_key] = file_type
            return file_type
        
        # Try MIME type as fallback
        file_type = self._detect_by_mime_type(file_path)
        self._magic_cache[cache_key] = file_type
        return file_type
    
    def _detect_by_extension(self, file_path: Path) -> FileType:
        """Detect file type by extension."""
        suffix = file_path.suffix.lower()
        
        extension_map = {
            '.pdf': FileType.PDF,
            '.docx': FileType.DOCX,
            '.pptx': FileType.PPTX,
            '.html': FileType.HTML,
            '.htm': FileType.HTML,
            '.txt': FileType.TXT,
            '.csv': FileType.CSV,
            '.json': FileType.JSON,
            '.xml': FileType.XML,
            '.md': FileType.MARKDOWN,
            '.markdown': FileType.MARKDOWN,
            '.eml': FileType.TXT,  # Email files as text for now
            '.msg': FileType.TXT,  # Outlook messages
        }
        
        return extension_map.get(suffix, FileType.UNKNOWN)
    
    async def _detect_by_magic_numbers(self, file_path: Path) -> FileType:
        """Detect file type by reading magic numbers from file header."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 512 bytes for magic number detection
                header = f.read(512)
                
                if not header:
                    return FileType.UNKNOWN
                
                # Check for known magic numbers
                for magic_bytes, file_type in MAGIC_NUMBERS.items():
                    if header.startswith(magic_bytes):
                        if file_type == 'zip_based':
                            # Need to inspect ZIP contents for Office formats
                            return await self._detect_office_format(file_path)
                        elif isinstance(file_type, str):
                            # Archive formats - treat as unknown for now
                            return FileType.UNKNOWN
                        else:
                            return file_type
                
                # Check if it's a ZIP file that might be an Office document
                if header.startswith(b'PK'):
                    return await self._detect_office_format(file_path)
                
        except (IOError, OSError):
            pass
        
        return FileType.UNKNOWN
    
    async def _detect_office_format(self, file_path: Path) -> FileType:
        """Detect specific Office format by inspecting ZIP contents."""
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                file_list = zf.namelist()
                
                # Check for Office-specific directory structures
                for file_name in file_list:
                    file_bytes = file_name.encode('utf-8')
                    for signature, file_type in OFFICE_SIGNATURES.items():
                        if signature in file_bytes:
                            return file_type
                
                # Check for content types
                try:
                    content_types = zf.read('[Content_Types].xml')
                    if b'wordprocessingml' in content_types:
                        return FileType.DOCX
                    elif b'presentationml' in content_types:
                        return FileType.PPTX
                except KeyError:
                    pass
        
        except (zipfile.BadZipFile, OSError):
            pass
        
        return FileType.UNKNOWN
    
    async def _detect_by_content(self, file_path: Path) -> FileType:
        """Detect file type by analyzing text content."""
        try:
            # Try to read as text with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        # Read first few KB to detect content type
                        content = f.read(8192)
                        
                        if not content.strip():
                            continue
                        
                        # Check for email patterns
                        if self._is_email_content(content):
                            return FileType.TXT  # Treat emails as text for processing
                        
                        # Check for HTML patterns
                        if self._is_html_content(content):
                            return FileType.HTML
                        
                        # Check for XML patterns
                        if self._is_xml_content(content):
                            return FileType.XML
                        
                        # Check for JSON patterns
                        if self._is_json_content(content):
                            return FileType.JSON
                        
                        # Check for CSV patterns
                        if self._is_csv_content(content):
                            return FileType.CSV
                        
                        # Check for Markdown patterns
                        if self._is_markdown_content(content):
                            return FileType.MARKDOWN
                        
                        # If we can read it as text, assume it's a text file
                        return FileType.TXT
                
                except UnicodeDecodeError:
                    continue
        
        except (IOError, OSError):
            pass
        
        return FileType.UNKNOWN
    
    def _is_email_content(self, content: str) -> bool:
        """Check if content appears to be an email message."""
        content_lower = content.lower()
        
        # Count email headers
        header_count = 0
        for header in EMAIL_HEADERS:
            if header in content_lower:
                header_count += 1
        
        # If we find multiple email headers, it's likely an email
        if header_count >= 3:
            return True
        
        # Try parsing as email
        try:
            msg = email.message_from_string(content)
            # Check if it has standard email headers
            required_headers = ['from', 'to', 'subject']
            present_headers = sum(1 for h in required_headers if msg.get(h))
            return present_headers >= 2
        except:
            pass
        
        return False
    
    def _is_html_content(self, content: str) -> bool:
        """Check if content appears to be HTML."""
        content_lower = content.lower()
        html_indicators = ['<html', '<head', '<body', '<div', '<p>', '<a href', '<!doctype html']
        return any(indicator in content_lower for indicator in html_indicators)
    
    def _is_xml_content(self, content: str) -> bool:
        """Check if content appears to be XML."""
        content_stripped = content.strip()
        return (content_stripped.startswith('<?xml') or 
                (content_stripped.startswith('<') and 
                 content_stripped.endswith('>') and
                 '</' in content_stripped))
    
    def _is_json_content(self, content: str) -> bool:
        """Check if content appears to be JSON."""
        content_stripped = content.strip()
        if not content_stripped:
            return False
        
        try:
            import json
            json.loads(content_stripped)
            return True
        except:
            return False
    
    def _is_csv_content(self, content: str) -> bool:
        """Check if content appears to be CSV."""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check if multiple lines have similar comma patterns
        comma_counts = [line.count(',') for line in lines[:5]]
        # If most lines have commas and similar counts, likely CSV
        if len([c for c in comma_counts if c > 0]) >= len(comma_counts) * 0.8:
            # Check consistency
            avg_commas = sum(comma_counts) / len(comma_counts)
            return all(abs(c - avg_commas) <= avg_commas * 0.5 for c in comma_counts if c > 0)
        
        return False
    
    def _is_markdown_content(self, content: str) -> bool:
        """Check if content appears to be Markdown."""
        markdown_indicators = ['# ', '## ', '### ', '**', '__', '`', '[', '](', '* ', '- ', '1. ']
        indicator_count = sum(1 for indicator in markdown_indicators if indicator in content)
        return indicator_count >= 2
    
    def _detect_by_mime_type(self, file_path: Path) -> FileType:
        """Detect file type using MIME type detection as fallback."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if not mime_type:
            return FileType.UNKNOWN
        
        mime_map = {
            'application/pdf': FileType.PDF,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCX,
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileType.PPTX,
            'text/html': FileType.HTML,
            'text/plain': FileType.TXT,
            'text/csv': FileType.CSV,
            'application/json': FileType.JSON,
            'text/xml': FileType.XML,
            'application/xml': FileType.XML,
            'text/markdown': FileType.MARKDOWN,
        }
        
        if mime_type in mime_map:
            return mime_map[mime_type]
        
        # Generic text types
        if mime_type.startswith('text/'):
            return FileType.TXT
        
        return FileType.UNKNOWN
    
    def is_archive_file(self, file_path: FilePath) -> bool:
        """
        Check if file is an archive that we can extract.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is a supported archive format
        """
        file_path = Path(file_path)
        
        # Check by extension
        archive_extensions = {'.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.gz', '.bz2'}
        if file_path.suffix.lower() in archive_extensions:
            return True
        
        # Check by magic numbers
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
                # ZIP files
                if header.startswith(b'PK'):
                    return True
                
                # GZIP files
                if header.startswith(b'\x1f\x8b'):
                    return True
                
                # BZIP2 files
                if header.startswith(b'BZ'):
                    return True
                
                # TAR files (may not have magic numbers at start)
                try:
                    with tarfile.open(file_path, 'r') as tar:
                        return True
                except:
                    pass
        
        except (IOError, OSError):
            pass
        
        return False
    
    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions."""
        return {
            '.pdf', '.docx', '.pptx', '.html', '.htm', '.txt', '.csv', 
            '.json', '.xml', '.md', '.markdown', '.eml', '.msg'
        } 