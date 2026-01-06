"""
Embedding service using local sentence-transformers models.
Generates vector embeddings for text chunks to enable semantic search.
"""
from typing import List
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using local sentence-transformers models.
    Uses all-mpnet-base-v2 model for high-quality text-to-vector transformation.
    No API keys required - runs completely offline after initial model download.
    """
    
    def __init__(
        self,
        model: str = "all-mpnet-base-v2",
        output_dimensionality: int = 768,
        device: str = None  # None = auto-detect (cuda/cpu)
    ):
        """
        Initialize the embedding service with a local model.
        
        Args:
            model: Sentence-transformers model name (default: all-mpnet-base-v2)
                   Options: 
                   - all-mpnet-base-v2: 768 dims, best quality (recommended)
                   - all-MiniLM-L6-v2: 384 dims, faster but lower quality
            output_dimensionality: Expected output dimensions (for validation)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model
        self.output_dimensionality = output_dimensionality
        
        logger.info(f"Loading sentence-transformer model: {model}")
        logger.info("First-time download may take a few minutes...")
        
        # Load model (will download on first use, ~420MB for all-mpnet-base-v2)
        self.model = SentenceTransformer(model, device=device)
        
        # Validate output dimensions match expected
        actual_dims = self.model.get_sentence_embedding_dimension()
        if actual_dims != output_dimensionality:
            logger.warning(
                f"Model {model} produces {actual_dims}D embeddings, "
                f"but config expects {output_dimensionality}D. "
                f"Using actual model dimensions: {actual_dims}D"
            )
            self.output_dimensionality = actual_dims
        
        logger.info(f"Model loaded successfully ({self.output_dimensionality}D embeddings)")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector as numpy array
        """
        # sentence-transformers returns normalized embeddings by default
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching for efficiency.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default: 32 for optimal GPU/CPU usage)
            
        Returns:
            List of normalized embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}...")
        
        # sentence-transformers handles batching internally and shows progress
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Convert to list of arrays
        return [emb for emb in embeddings]
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding optimized for search queries.
        For sentence-transformers, this is the same as embed_text.
        
        Args:
            query: Search query text
            
        Returns:
            Normalized query embedding
        """
        return self.embed_text(query)


if __name__ == "__main__":
    print("Sentence-Transformers EmbeddingService quick test\n" + "-"*40)
    service = EmbeddingService()
    query = "Who brought the USB?"
    sentences = [
        "Marcus asked Alex to bring the USB.",
        "The meeting is at 8pm tonight.",
        "Alex forgot the USB at home."
    ]
    print(f"Query: {query}")
    print("Sentences:")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")

    # Embed query and sentences
    query_emb = service.embed_query(query)
    sent_embs = service.embed_batch(sentences)

    # Compute cosine distances
    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nCosine distances (lower = more similar):")
    for i, emb in enumerate(sent_embs):
        dist = cosine_distance(query_emb, emb)
        print(f"Query <-> Sentence {i+1}: {dist:.4f}")
