# VecClean Documentation

VecClean is an ultra-low latency text cleaning, deduplication, and vectorization pipeline designed for production-ready document processing at scale. It combines the flexibility of Python with the performance of C++ to deliver maximum throughput for RAG pipelines.

## Installation


### From PyPI (Development Version)
```bash
pip install vecclean
```


## Quick Start

### Basic Text Processing

```python
import asyncio
from vecclean import Pipeline

async def main():
    # Initialize the pipeline with default configuration
    pipeline = Pipeline()
    
    # Process a simple text
    sample_text = """
    This is a sample document for testing.
    It contains multiple sentences that need processing.
    
    Some sentences might have    extra   whitespace.
    Others might contain HTML tags like <b>bold text</b>.
    """
    
    # Process the text
    result = await pipeline.process_text(sample_text)
    
    # Access the processed chunks
    for i, chunk in enumerate(result.chunks):
        print(f"Chunk {i}: {chunk.text}")
        print(f"Hash: {chunk.text_hash}")
        print(f"Word count: {chunk.word_count}")
        print(f"Embedding shape: {chunk.embedding.shape if chunk.embedding is not None else 'None'}")
        print("---")

# Run the async function
asyncio.run(main())
```

### Processing Files

```python
import asyncio
from vecclean import Pipeline
from pathlib import Path

async def process_files():
    pipeline = Pipeline()
    
    # Process multiple files
    files = [
        "document1.pdf",
        "document2.docx", 
        "document3.txt"
    ]
    
    result = await pipeline.process_files(files)
    
    print(f"Processed {len(result.chunks)} chunks from {len(files)} files")
    print(f"Processing time: {result.stats.total_processing_time:.2f} seconds")
    
    # Save results to JSON
    import json
    with open("processed_chunks.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

asyncio.run(process_files())
```

## Configuration

### Custom Configuration

```python
from vecclean import Pipeline, Config

# Create custom configuration
config = Config(
    chunking={
        "chunk_size": 512,
        "chunk_overlap": 50,
        "strategy": "sentence"
    },
    cleaning={
        "normalize_whitespace": True,
        "strip_html_tags": True,
        "remove_stopwords": True
    },
    dedup={
        "sentence_dedup": True,
        "chunk_dedup": True,
        "similarity_threshold": 0.85
    },
    embedding={
        "model_name": "all-MiniLM-L6-v2",
        "device": "auto"
    }
)

# Initialize pipeline with custom config
pipeline = Pipeline(config)
```

### Configuration Options

#### Chunking Configuration
- `chunk_size`: Maximum size of each chunk (default: 512)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `strategy`: Chunking strategy - "sentence", "token", "recursive" (default: "sentence")
- `min_chunk_size`: Minimum chunk size (default: 100)
- `max_chunk_size`: Maximum chunk size (default: 1000)

#### Cleaning Configuration
- `normalize_unicode`: Unicode normalization form (default: "NFC")
- `normalize_whitespace`: Normalize whitespace (default: True)
- `standardize_punctuation`: Standardize punctuation (default: True)
- `strip_html_tags`: Remove HTML tags (default: True)
- `remove_stopwords`: Remove stopwords (default: True)
- `min_text_length`: Minimum text length (default: 10)

#### Deduplication Configuration
- `sentence_dedup`: Enable sentence-level deduplication (default: True)
- `chunk_dedup`: Enable chunk-level deduplication (default: True)
- `similarity_threshold`: Similarity threshold for deduplication (default: 0.85)
- `hash_algorithm`: Hash algorithm for deduplication (default: "xxhash")

#### Embedding Configuration
- `model_name`: Embedding model name (default: "all-MiniLM-L6-v2")
- `device`: Device for embedding generation (default: "auto")
- `batch_size`: Batch size for embedding generation (default: 32)
- `cache_embeddings`: Cache embeddings (default: True)

## API Reference

### Pipeline Class

The main entry point for text processing.

#### Methods

##### `process_text(text: str) -> ProcessingResult`
Process a single text string.

```python
result = await pipeline.process_text("Your text here")
```

##### `process_files(files: List[str]) -> ProcessingResult`
Process multiple files.

```python
result = await pipeline.process_files(["file1.pdf", "file2.txt"])
```

##### `process_documents(documents: List[Document]) -> ProcessingResult`
Process a list of document objects.

```python
from vecclean.core.types import Document

documents = [
    Document(content="Text 1", metadata={"source": "file1.txt"}),
    Document(content="Text 2", metadata={"source": "file2.txt"})
]
result = await pipeline.process_documents(documents)
```

### ProcessingResult

The result object containing processed chunks and metadata.

#### Attributes
- `chunks`: List of `CleanedChunk` objects
- `stats`: Processing statistics
- `status`: Processing status (COMPLETED, FAILED, etc.)
- `errors`: List of error messages
- `warnings`: List of warning messages

#### Methods
- `to_dict()`: Convert to dictionary for serialization
- `to_json()`: Convert to JSON string

### CleanedChunk

Represents a processed text chunk.

#### Attributes
- `text`: The cleaned text content
- `text_hash`: Hash of the text for deduplication
- `embedding`: Vector embedding (numpy array)
- `chunk_index`: Index of the chunk in the document
- `start_char`: Starting character position
- `end_char`: Ending character position
- `word_count`: Number of words in the chunk
- `char_count`: Number of characters in the chunk

## Supported File Formats

VecClean supports processing various file formats:

- **PDF** (.pdf) - Text extraction with metadata
- **Word Documents** (.docx) - Text and formatting
- **PowerPoint** (.pptx) - Text from slides
- **Text Files** (.txt) - Plain text
- **HTML** (.html, .htm) - Web content
- **Email** (.eml) - Email messages

## Performance Features

### C++ Acceleration
VecClean includes C++ optimizations for high-performance text processing:
- SIMD-optimized text cleaning
- Parallel processing with work-stealing thread pools
- Memory-efficient streaming for large files

### Async Processing
All operations are asynchronous for better performance:
```python
# Process multiple texts concurrently
tasks = [
    pipeline.process_text(text1),
    pipeline.process_text(text2),
    pipeline.process_text(text3)
]
results = await asyncio.gather(*tasks)
```

## Error Handling

```python
try:
    result = await pipeline.process_text(text)
    if result.status == ProcessingStatus.COMPLETED:
        print(f"Successfully processed {len(result.chunks)} chunks")
    else:
        print(f"Processing failed: {result.errors}")
except Exception as e:
    print(f"Error during processing: {e}")
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI, UploadFile, File
from vecclean import Pipeline

app = FastAPI()
pipeline = Pipeline()

@app.post("/process-text")
async def process_text(text: str):
    result = await pipeline.process_text(text)
    return {
        "chunks": [chunk.to_dict() for chunk in result.chunks],
        "stats": result.stats.to_dict()
    }

@app.post("/process-file")
async def process_file(file: UploadFile = File(...)):
    content = await file.read()
    result = await pipeline.process_text(content.decode())
    return {"chunks": len(result.chunks)}
```

### With LangChain

```python
from langchain.text_splitter import TextSplitter
from vecclean import Pipeline

class VecCleanTextSplitter(TextSplitter):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
    
    async def split_text(self, text: str):
        result = await self.pipeline.process_text(text)
        return [chunk.text for chunk in result.chunks]

# Usage
pipeline = Pipeline()
splitter = VecCleanTextSplitter(pipeline)
chunks = await splitter.split_text("Your long text here")
```

## Best Practices

1. **Batch Processing**: Process multiple documents together for better performance
2. **Memory Management**: Use streaming for very large files
3. **Configuration**: Tune chunking parameters based on your use case
4. **Error Handling**: Always check the processing status and handle errors
5. **Caching**: Enable embedding caching for repeated processing

## Troubleshooting

### Common Issues

1. **C++ Backend Not Available**: VecClean will fall back to Python implementation
2. **Memory Issues**: Reduce batch size or use streaming for large files
3. **Slow Processing**: Check if C++ backend is enabled and consider using GPU for embeddings

### Debug Mode

Enable debug logging to see detailed processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = Pipeline()
# Processing will now show detailed logs
```

## License

VecClean is licensed under the MIT License. See the LICENSE file for details.
