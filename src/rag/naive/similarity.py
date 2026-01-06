from typing import List
import numpy as np
from src.core.models import Chunk, SearchResult

    
class SimilaritySearch:
    """
    Semantic similarity search using pre-embedded chunks.
    Finds chunks relevant to a query above a similarity threshold.
    
    Note: This class expects chunks to already have embeddings.
    Use ChunkIndexer to embed chunks before passing them here.
    """
    
    def __init__(self, chunks: List[Chunk]):
        """
        Initialize search with pre-indexed chunks.
        
        Args:
            chunks: List of chunks with embeddings already set
            
        Raises:
            ValueError: If chunks list is empty
            ValueError: If any chunk is missing embedding
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        # Validate all chunks have embeddings
        for chunk in chunks:
            if not hasattr(chunk, 'embedding') or chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.id}' is missing embedding. "
                    "Use ChunkIndexer to embed chunks before searching."
                )
        
        self.chunks = chunks
    
    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.7,
        max_results: int = None
    ) -> List[SearchResult]:
        """
        Find chunks above similarity threshold.
        
        Args:
            query_embedding: Embedded query vector (from EmbeddingService.embed_query)
            threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with score >= threshold,
            sorted by score descending, limited to max_results
            
        Raises:
            ValueError: If threshold not in [0, 1]
            ValueError: If max_results <= 0
            ValueError: If query_embedding is invalid
        """
        # Validate threshold
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        # Validate max_results if provided
        if max_results is not None and max_results <= 0:
            raise ValueError(f"max_results must be positive, got {max_results}")
        
        # Validate query_embedding
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("query_embedding must be a numpy array")
        
        if query_embedding.size == 0:
            raise ValueError("query_embedding cannot be empty")
        
        # Calculate similarity for all chunks
        results = []
        for chunk in self.chunks:
            score = self._cosine_similarity(query_embedding, chunk.embedding)
            
            if score >= threshold:
                results.append(SearchResult(chunk=chunk, score=float(score)))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        if max_results is None:
            return results
        else:
            return results[:max_results]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        Since embeddings are normalized, this is just the dot product.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        return float(np.dot(a, b))


if __name__ == "__main__":
    from src.core.models import Message
    from src.rag.naive.chunker import MessageChunker
    from src.rag.naive.chunk_indexer import ChunkIndexer
    from src.core.embedding_service import EmbeddingService
    
    print("=" * 70)
    print("SimilaritySearch - Simple Manual Test")
    print("=" * 70)
    
    # Create sample messages and chunks
    messages = [
        Message(id="m1", sender="marcus", receiver="alex", body="Bring the USB drive tonight"),
        Message(id="m2", sender="alex", receiver="marcus", body="What time should I arrive?"),
        Message(id="m3", sender="marcus", receiver="alex", body="Meeting is at 8pm at the cafe"),
        Message(id="m4", sender="alice", receiver="bob", body="Did you see the weather forecast?"),
        Message(id="m5", sender="bob", receiver="alice", body="Yes, it will rain tomorrow")
    ]
    
    print(f"\n✓ Created {len(messages)} sample messages")
    
    # Chunk messages
    chunker = MessageChunker()
    chunks = chunker.chunk_messages(messages)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Index chunks
    embedding_service = EmbeddingService()
    indexer = ChunkIndexer(embedding_service, cache_dir="test_cache")
    indexed_chunks = indexer.index_and_cache(chunks, "similarity_test")
    print(f"✓ Indexed {len(indexed_chunks)} chunks")
    
    # Initialize search
    search = SimilaritySearch(indexed_chunks)
    print(f"✓ Initialized SimilaritySearch")
    
    # Test 1: Search for USB-related messages
    print("\n" + "=" * 70)
    print("[TEST 1] Search: 'Who should bring the USB?'")
    print("-" * 70)
    query = "Who should bring the USB?"
    query_emb = embedding_service.embed_query(query)
    
    results = search.search(query_emb, threshold=0.5, max_results=3)
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Text: {result.chunk.text}")
        print(f"   ID: {result.chunk.id}\n")
    
    # Test 2: Search for time-related messages
    print("=" * 70)
    print("[TEST 2] Search: 'What time is the meeting?'")
    print("-" * 70)
    query2 = "What time is the meeting?"
    query_emb2 = embedding_service.embed_query(query2)
    
    results2 = search.search(query_emb2, threshold=0.5, max_results=3)
    print(f"Found {len(results2)} results:\n")
    for i, result in enumerate(results2, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Text: {result.chunk.text}")
        print(f"   ID: {result.chunk.id}\n")
    
    # Test 3: Search with high threshold (strict)
    print("=" * 70)
    print("[TEST 3] Search with high threshold (0.8)")
    print("-" * 70)
    query3 = "USB drive"
    query_emb3 = embedding_service.embed_query(query3)
    
    results3 = search.search(query_emb3, threshold=0.8, max_results=5)
    print(f"Found {len(results3)} results with threshold >= 0.8\n")
    for i, result in enumerate(results3, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Text: {result.chunk.text}\n")
    
    # Test 4: Search with low threshold (permissive)
    print("=" * 70)
    print("[TEST 4] Search with low threshold (0.3)")
    print("-" * 70)
    results4 = search.search(query_emb3, threshold=0.3, max_results=5)
    print(f"Found {len(results4)} results with threshold >= 0.3")
    
    # Test 5: Max results limiting
    print("\n" + "=" * 70)
    print("[TEST 5] Max results limiting (max_results=2)")
    print("-" * 70)
    results5 = search.search(query_emb, threshold=0.0, max_results=2)
    print(f"Requested max 2 results, got {len(results5)} results")
    
    # Cleanup
    indexer.clear_cache("similarity_test")
    
    print("\n" + "=" * 70)
    print("All manual tests completed!")
    print("=" * 70)