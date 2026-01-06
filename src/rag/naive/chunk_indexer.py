from typing import List, Optional
from pathlib import Path
import pickle
from src.core.models import Chunk
from src.core.embedding_service import EmbeddingService


class ChunkIndexer:
    """
    Embeds chunks for similarity search with smart caching support.
    Handles both embedding and persistence of indexed chunks.
    """
    
    def __init__(self, embedding_service: EmbeddingService, cache_dir: str = "cache"):
        """
        Initialize indexer with embedding service.
        
        Args:
            embedding_service: Service for generating embeddings
            cache_dir: Directory to store cached chunks (default: "cache")
        """
        self.embedding_service = embedding_service
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def index_and_cache(
        self, 
        chunks: List[Chunk], 
        cache_name: str,
        force_reindex: bool = False
    ) -> List[Chunk]:
        """
        Embed chunks with smart caching. Loads from cache if available.
        
        Args:
            chunks: List of chunks to embed (only used if cache miss)
            cache_name: Name for the cache file (e.g., "story_chunks")
            force_reindex: If True, ignore cache and re-embed
            
        Returns:
            List of chunks with embeddings populated
            
        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        # Try loading from cache (unless force_reindex)
        if not force_reindex:
            cached = self.load_from_cache(cache_name)
            if cached is not None:
                return cached
        
        # Cache miss or forced reindex - embed and save
        indexed = self.index(chunks)
        self.save_to_cache(indexed, cache_name)
        return indexed
    
    def index(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Embed chunks without caching.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            Same chunks list with embeddings populated
            
        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        # Extract text from all chunks
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Batch embed all texts
        embeddings = self.embedding_service.embed_batch(texts=chunk_texts)
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def load_from_cache(self, cache_name: str) -> Optional[List[Chunk]]:
        """
        Load indexed chunks from cache with validation.
        
        Args:
            cache_name: Name of the cache file
            
        Returns:
            List of chunks with embeddings, or None if cache doesn't exist or is invalid
        """
        cache_path = self._get_cache_path(cache_name)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                chunks = pickle.load(f)
            
            # Validate chunks is a list
            if not isinstance(chunks, list):
                return None
            
            # Validate all chunks have embeddings
            if chunks and not all(hasattr(c, 'embedding') and c.embedding is not None for c in chunks):
                return None
            
            return chunks
            
        except Exception:
            # Any error loading/deserializing - treat as cache miss
            return None
    
    def save_to_cache(self, chunks: List[Chunk], cache_name: str) -> None:
        """
        Save indexed chunks to cache.
        
        Args:
            chunks: List of chunks with embeddings
            cache_name: Name for the cache file
            
        Raises:
            ValueError: If any chunk is missing embedding
        """
        # Validate all chunks have embeddings
        for chunk in chunks:
            if not hasattr(chunk, 'embedding') or chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.id}' is missing embedding. "
                    "Call index() before saving to cache."
                )
        
        cache_path = self._get_cache_path(cache_name)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(chunks, f)
    
    def clear_cache(self, cache_name: Optional[str] = None) -> None:
        """
        Clear cache files.
        
        Args:
            cache_name: Specific cache to clear, or None to clear all
        """
        if cache_name is not None:
            # Clear specific cache
            cache_path = self._get_cache_path(cache_name)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def _get_cache_path(self, cache_name: str) -> Path:
        """
        Get full path to cache file.
        
        Args:
            cache_name: Name of the cache
            
        Returns:
            Path object for the cache file
        """
        return self.cache_dir / f"{cache_name}.pkl"


if __name__ == "__main__":
    from src.core.models import Message
    from src.rag.naive.chunker import MessageChunker
    
    print("=" * 70)
    print("ChunkIndexer - Simple Manual Test")
    print("=" * 70)
    
    # Create sample messages and chunks
    messages = [
        Message(id="m1", sender="marcus", receiver="alex", body="Bring the USB"),
        Message(id="m2", sender="alex", receiver="marcus", body="What time?"),
        Message(id="m3", sender="marcus", receiver="alex", body="8pm at cafe")
    ]
    
    chunker = MessageChunker()
    chunks = chunker.chunk_messages(messages)
    
    print(f"\n✓ Created {len(chunks)} chunks")
    
    # Initialize indexer
    embedding_service = EmbeddingService()
    indexer = ChunkIndexer(embedding_service, cache_dir="test_cache")
    
    # Test 1: First indexing (cache miss)
    print("\n" + "=" * 70)
    print("[TEST 1] First indexing - should embed and cache")
    print("-" * 70)
    indexed = indexer.index_and_cache(chunks, cache_name="test_story")
    print(f"✓ Indexed {len(indexed)} chunks")
    print(f"✓ First chunk has embedding: {indexed[0].embedding is not None}")
    print(f"✓ Embedding shape: {indexed[0].embedding.shape}")
    
    # Test 2: Second indexing (cache hit)
    print("\n" + "=" * 70)
    print("[TEST 2] Second indexing - should load from cache")
    print("-" * 70)
    indexed2 = indexer.index_and_cache(chunks, cache_name="test_story")
    print(f"✓ Loaded {len(indexed2)} chunks from cache")
    print(f"✓ Embeddings match: {(indexed[0].embedding == indexed2[0].embedding).all()}")
    
    # Test 3: Force reindex
    print("\n" + "=" * 70)
    print("[TEST 3] Force reindex - should ignore cache")
    print("-" * 70)
    indexed3 = indexer.index_and_cache(
        chunks, 
        cache_name="test_story",
        force_reindex=True
    )
    print(f"✓ Re-indexed {len(indexed3)} chunks")
    
    # Test 4: Manual cache operations
    print("\n" + "=" * 70)
    print("[TEST 4] Manual cache operations")
    print("-" * 70)
    
    # Load from cache
    cached = indexer.load_from_cache("test_story")
    print(f"✓ Loaded from cache: {len(cached)} chunks")
    
    # Clear cache
    indexer.clear_cache("test_story")
    print(f"✓ Cleared cache")
    
    # Try loading again (should be None)
    cached2 = indexer.load_from_cache("test_story")
    print(f"✓ Cache cleared: {cached2 is None}")
    
    print("\n" + "=" * 70)
    print("All manual tests completed!")
    print("=" * 70)