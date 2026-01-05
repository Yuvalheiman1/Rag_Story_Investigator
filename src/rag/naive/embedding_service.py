"""
Embedding service using Google Gemini API.
Generates vector embeddings for text chunks to enable semantic search.
"""
from os import getenv
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / '.env')
from typing import List
import time
import logging
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using Google Gemini API.
    Uses gemini-embedding-001 model for text-to-vector transformation.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = 768,
        requests_per_minute: int = 90  # Free tier is 100, use 90 to be safe
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Embedding model to use (default: gemini-embedding-001)
            task_type: Task type for optimization (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            output_dimensionality: Output embedding dimensions (128-3072, recommended: 768, 1536, 3072)
            requests_per_minute: Max requests per minute (default: 90 for free tier safety)
        """
        self.api_key = api_key or getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it as environment variable or pass to constructor."
            )
        
        self.model = model
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        self.requests_per_minute = requests_per_minute
        self.min_delay_between_requests = 60.0 / requests_per_minute  # Seconds between requests
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
    
    def _embed_with_retry(self, contents, task_type: str, max_retries: int = 3) -> any:
        """
        Make an embed request with automatic retry on rate limit errors.
        
        Args:
            contents: Text or list of texts to embed
            task_type: Task type for embedding
            max_retries: Maximum retry attempts
            
        Returns:
            Embed result from API
            
        Raises:
            ClientError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.output_dimensionality
                    )
                )
                return result
                
            except ClientError as e:
                # Check if it's a rate limit error (429)
                if e.code == 429 and attempt < max_retries - 1:
                    # Extract retry delay from error if available
                    retry_delay = 60  # Default to 60 seconds
                    
                    # Try to parse retry delay from error message
                    if hasattr(e, 'message') and 'retry in' in str(e.message).lower():
                        import re
                        match = re.search(r'retry in ([\d.]+)s', str(e.message))
                        if match:
                            retry_delay = float(match.group(1))
                    
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {retry_delay:.1f}s before retry..."
                    )
                    time.sleep(retry_delay)
                else:
                    # Not a rate limit error, or out of retries
                    raise
        
        raise ClientError("Max retries exceeded")
    
    def embed_text(self, text: str, task_type: str = None) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            task_type: Override default task type for this embedding
            
        Returns:
            Normalized embedding vector as numpy array
        """
        result = self._embed_with_retry(
            contents=text,
            task_type=task_type or self.task_type
        )
        
        # Extract embedding values
        embedding = np.array(result.embeddings[0].values)
        
        # Normalize for dimensions < 3072 (as per Gemini docs)
        if self.output_dimensionality < 3072:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str], task_type: str = None, batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts, automatically batching if needed.
        Gemini API allows max 100 texts per batch, with rate limiting.
        
        Args:
            texts: List of texts to embed
            task_type: Override default task type for this batch
            batch_size: Maximum texts per API call (default: 100, Gemini's limit)
            
        Returns:
            List of normalized embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches of batch_size
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            
            logger.info(f"Embedding batch {batch_idx + 1}/{total_batches} ({len(batch)} texts)...")
            
            result = self._embed_with_retry(
                contents=batch,
                task_type=task_type or self.task_type
            )
            
            # Extract and normalize embeddings
            for embedding_obj in result.embeddings:
                embedding = np.array(embedding_obj.values)
                
                # Normalize for dimensions < 3072
                if self.output_dimensionality < 3072:
                    embedding = embedding / np.linalg.norm(embedding)
                
                all_embeddings.append(embedding)
            
            # Add delay between batches to avoid rate limits (except for last batch)
            if batch_idx < total_batches - 1:
                logger.debug(f"Waiting {self.min_delay_between_requests:.2f}s before next batch...")
                time.sleep(self.min_delay_between_requests)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding optimized for search queries.
        
        Args:
            query: Search query text
            
        Returns:
            Normalized query embedding
        """
        return self.embed_text(query, task_type="RETRIEVAL_QUERY")


if __name__ == "__main__":
    print("Gemini EmbeddingService quick test\n" + "-"*40)
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