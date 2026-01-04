from typing import List, Optional
from pathlib import Path
import pickle
from src.core.models import Chunk
from src.rag.naive.embedding_service import EmbeddingService


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
        pass
    
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
        pass
    
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
        pass
    
    def load_from_cache(self, cache_name: str) -> Optional[List[Chunk]]:
        """
        Load indexed chunks from cache with validation.
        
        Args:
            cache_name: Name of the cache file
            
        Returns:
            List of chunks with embeddings, or None if cache doesn't exist or is invalid
        """
        pass
    
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
        pass
    
    def clear_cache(self, cache_name: Optional[str] = None) -> None:
        """
        Clear cache files.
        
        Args:
            cache_name: Specific cache to clear, or None to clear all
        """
        pass
    
    def _get_cache_path(self, cache_name: str) -> Path:
        pass