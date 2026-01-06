"""
LightRAG implementation for story investigation.
Uses the LightRAG library for advanced graph-based RAG.
"""
import logging
import asyncio
from typing import List, Optional
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from src.core.models import Message, SearchResult, Chunk
from src.core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class LightRAGEngine:
    """
    LightRAG-based retrieval engine.
    Uses graph-based retrieval with entity extraction.
    """
    
    def __init__(
        self,
        messages: List[Message],
        embedding_service: EmbeddingService,
        llm_model_func,
        working_dir: str = "cache/lightrag",
        mode: str = "hybrid",
        force_reindex: bool = False
    ):
        """
        Initialize LightRAG engine.
        
        Args:
            messages: Pre-parsed story messages
            embedding_service: Service for generating embeddings
            llm_model_func: LLM function for entity extraction and queries
            working_dir: Directory for LightRAG storage
            mode: Query mode (naive, local, global, hybrid)
            force_reindex: If True, rebuild the index
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        self.messages = messages
        self.embedding_service = embedding_service
        self.llm_model_func = llm_model_func
        self.working_dir = Path(working_dir)
        self.mode = mode
        self.force_reindex = force_reindex
        
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing LightRAG with {len(messages)} messages")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize LightRAG instance and index messages."""
        try:
            # Create async embedding function wrapper for LightRAG
            async def async_embedding_func(texts: List[str]):
                """Async wrapper for embedding service that returns numpy arrays."""
                import numpy as np
                embeddings = self.embedding_service.embed_batch(texts)
                # Convert list of numpy arrays to single 2D numpy array
                return np.array(embeddings)
            
            embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_service.output_dimensionality,
                max_token_size=8192,
                func=async_embedding_func
            )
            
            # Create async LLM wrapper for LightRAG
            async def async_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                """Async wrapper for LLM function - runs sync LLM in executor."""
                import asyncio
                loop = asyncio.get_event_loop()
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                # Run sync LLM function in thread pool to make it truly async
                return await loop.run_in_executor(None, self.llm_model_func, full_prompt)
            
            # Initialize LightRAG
            logger.info("Creating LightRAG instance...")
            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=async_llm_func,
                embedding_func=embedding_func,
            )
            
            # Initialize storages (required before use)
            logger.info("Initializing LightRAG storages...")
            asyncio.run(self.rag.initialize_storages())
            
            # Check if we need to reindex
            if self.force_reindex or self._should_index():
                logger.info("Indexing messages into LightRAG...")
                self._index_messages()
            else:
                logger.info("Using existing LightRAG index")
            
            logger.info("LightRAG initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            raise
    
    def _should_index(self) -> bool:
        """Check if indexing is needed."""
        # Check if working directory has any data
        kv_store = self.working_dir / "kv_store_full_docs.json"
        return not kv_store.exists()
    
    def _index_messages(self) -> None:
        """Index all messages into LightRAG."""
        # Combine all messages into text documents
        # Each message as a separate document with metadata
        docs = []
        for msg in self.messages:
            timestamp_str = f" at {msg.timestamp}" if msg.timestamp else ""
            doc = (
                f"Message ID: {msg.id}{timestamp_str}\n"
                f"From: {msg.sender}\n"
                f"To: {msg.receiver}\n"
                f"Content: {msg.body}\n"
            )
            docs.append(doc)
        
        # Insert all documents
        combined_text = "\n\n".join(docs)
        logger.info(f"Inserting {len(docs)} messages into LightRAG...")
        self.rag.insert(combined_text)
        logger.info("Indexing complete")
    
    def retrieve(
        self,
        question: str,
        threshold: float = 0.7,  # Not used by LightRAG but kept for interface consistency
        max_results: Optional[int] = None  # Not used by LightRAG but kept for interface consistency
    ) -> List[SearchResult]:
        """
        Retrieve relevant information for a question using LightRAG.
        
        Args:
            question: The user's question
            threshold: Not used by LightRAG (kept for interface compatibility)
            max_results: Not used by LightRAG (kept for interface compatibility)
            
        Returns:
            List of SearchResult objects
            
        Raises:
            ValueError: If question is empty
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        logger.info(f"Querying LightRAG ({self.mode} mode): '{question}'")
        
        try:
            # Query LightRAG
            response = self.rag.query(
                question,
                param=QueryParam(mode=self.mode)
            )
            
            # Handle None or empty response (e.g., due to rate limiting or errors)
            if response is None or not response.strip():
                logger.warning("LightRAG returned empty response, likely due to API errors")
                return []
            
            # LightRAG returns text response, not chunks
            # We need to wrap it in our SearchResult format
            # Create a single "chunk" with the entire response
            chunk = Chunk(
                id="lightrag_response",
                text=response,
                metadata={
                    "mode": self.mode,
                    "source": "lightrag"
                },
                source_message_ids=[]  # LightRAG doesn't provide source tracking in this format
            )
            
            # Return as a single high-confidence result
            result = SearchResult(chunk=chunk, score=1.0)
            
            logger.info(f"LightRAG returned response ({len(response)} chars)")
            return [result]
            
        except Exception as e:
            logger.error(f"LightRAG query failed: {e}")
            raise
