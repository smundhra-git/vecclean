import asyncio
from vecclean import Pipeline

async def main():
    # Initialize the pipeline with default configuration
    print("Initializing pipeline...")
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