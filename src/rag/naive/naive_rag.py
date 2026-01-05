"""
Naive RAG implementation for story investigation.
Receives pre-parsed messages and returns search results.
"""
import logging
from typing import List, Optional
from src.rag.naive.chunker import MessageChunker
from src.rag.naive.chunk_indexer import ChunkIndexer
from src.rag.naive.similarity import SimilaritySearch
from src.rag.naive.embedding_service import EmbeddingService
from src.core.models import Message, SearchResult

logger = logging.getLogger(__name__)


class NaiveRAG:
    """
    Naive RAG retrieval engine.
    Receives messages, chunks them, indexes, and searches.
    """
    
    def __init__(
        self,
        messages: List[Message],
        chunker: MessageChunker,
        embedding_service: EmbeddingService,
        indexer: ChunkIndexer,
        cache_name: str = "story_chunks",
        force_reindex: bool = False
    ):
        """
        Initialize the naive RAG system.
        
        Args:
            messages: Pre-parsed story messages
            chunker: Chunker instance for breaking messages into chunks
            embedding_service: Service for generating embeddings
            indexer: Indexer for embedding chunks with caching
            cache_name: Name for cache file
            force_reindex: If True, ignore cache and re-index
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        self.messages = messages
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.indexer = indexer
        self.cache_name = cache_name
        
        # Initialize by chunking and indexing
        logger.info(f"Initializing NaiveRAG with {len(messages)} messages")
        self._initialize(force_reindex)
    
    def _initialize(self, force_reindex: bool) -> None:
        """
        Chunk and index messages.
        
        Args:
            force_reindex: If True, bypass cache
        """
        # Step 1: Chunk messages
        logger.info("Chunking messages...")
        chunks = self.chunker.chunk_messages(self.messages)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Index chunks (with caching)
        logger.info("Indexing chunks...")
        self.indexed_chunks = self.indexer.index_and_cache(
            chunks,
            cache_name=self.cache_name,
            force_reindex=force_reindex
        )
        logger.info(f"Indexed {len(self.indexed_chunks)} chunks")
        
        # Step 3: Initialize search
        logger.info("Initializing similarity search...")
        self.search = SimilaritySearch(self.indexed_chunks)
        logger.info("NaiveRAG initialized successfully")
    
    
    def retrieve(
        self,
        question: str,
        threshold: float = 0.7,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a question.
        
        Args:
            question: The user's question
            threshold: Minimum similarity score (0.0-1.0)
            max_results: Max chunks to return (None = all above threshold)
            
        Returns:
            List of SearchResult objects, sorted by score descending
            
        Raises:
            ValueError: If question is empty or invalid parameters
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        logger.info(f"Retrieving results for: '{question}'")
        
        # Embed the query
        logger.debug("Embedding query...")
        query_embedding = self.embedding_service.embed_query(question)
        
        # Search for relevant chunks
        logger.debug(f"Searching (threshold={threshold}, max_results={max_results})...")
        results = self.search.search(
            query_embedding=query_embedding,
            threshold=threshold,
            max_results=max_results
        )
        
        logger.info(f"Found {len(results)} relevant chunks")
        for i, result in enumerate(results, 1):
            logger.debug(f"  {i}. [{result.score:.3f}] {result.chunk.id}")
        
        return results
