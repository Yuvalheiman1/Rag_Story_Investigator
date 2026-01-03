"""
Embedding service using Google Gemini API.
Generates vector embeddings for text chunks to enable semantic search.
"""
from os import getenv
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / '.env')
from typing import List
import numpy as np
from google import genai
from google.genai import types


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
        output_dimensionality: int = 768
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Embedding model to use (default: gemini-embedding-001)
            task_type: Task type for optimization (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            output_dimensionality: Output embedding dimensions (128-3072, recommended: 768, 1536, 3072)
        """
        self.api_key = api_key or getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it as environment variable or pass to constructor."
            )
        
        self.model = model
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
    
    def embed_text(self, text: str, task_type: str = None) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            task_type: Override default task type for this embedding
            
        Returns:
            Normalized embedding vector as numpy array
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type or self.task_type,
                output_dimensionality=self.output_dimensionality
            )
        )
        
        # Extract embedding values
        embedding = np.array(result.embeddings[0].values)
        
        # Normalize for dimensions < 3072 (as per Gemini docs)
        if self.output_dimensionality < 3072:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str], task_type: str = None) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in one API call.
        
        Args:
            texts: List of texts to embed
            task_type: Override default task type for this batch
            
        Returns:
            List of normalized embedding vectors
        """
        if not texts:
            return []
        
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type or self.task_type,
                output_dimensionality=self.output_dimensionality
            )
        )
        
        # Extract and normalize embeddings
        embeddings = []
        for embedding_obj in result.embeddings:
            embedding = np.array(embedding_obj.values)
            
            # Normalize for dimensions < 3072
            if self.output_dimensionality < 3072:
                embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding)
        
        return embeddings
    
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