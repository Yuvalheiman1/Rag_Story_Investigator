"""
GraphRAG engine using Neo4j and neo4j-graphrag-python.
Implements vector-based retrieval over story message chunks stored in Neo4j.
"""
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever

from src.core.models import Message, Chunk, SearchResult, RagEngine
from src.core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class GraphRAGEngine(RagEngine):
    """
    GraphRAG engine that stores message chunks in Neo4j and uses vector retrieval.
    
    Architecture:
    - Each Message becomes a :Chunk node in Neo4j
    - Vector index enables semantic search
    - Returns SearchResult objects compatible with existing prompt builder
    """
    
    def __init__(
        self,
        *,
        messages: List[Message],
        embedding_service: EmbeddingService,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        vector_index_name: str = "chunk_embeddings",
        node_label: str = "Chunk",
        embedding_property: str = "embedding",
        working_dir: str = "cache/graphrag",
        force_reindex: bool = False,
        top_k: int = 10,
    ):
        """
        Initialize GraphRAG engine.
        
        Args:
            messages: Story messages to index
            embedding_service: Service for generating embeddings
            neo4j_uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            vector_index_name: Name of the vector index
            node_label: Label for chunk nodes
            embedding_property: Property name for embeddings
            working_dir: Directory for caching/markers
            force_reindex: Force re-indexing even if already indexed
            top_k: Number of results to retrieve
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        self._messages = messages
        self._embedding_service = embedding_service
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._neo4j_database = neo4j_database
        self._vector_index_name = vector_index_name
        self._node_label = node_label
        self._embedding_property = embedding_property
        self._working_dir = Path(working_dir)
        self._force_reindex = force_reindex
        self._top_k = top_k
        
        self._driver = None
        self._retriever = None
        self._indexed_marker = self._working_dir / ".indexed"
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Neo4j connection and ensure data is indexed."""
        if self._force_reindex and self._working_dir.exists():
            logger.info(f"GraphRAG force reindex: removing {self._working_dir}")
            shutil.rmtree(self._working_dir, ignore_errors=True)
        
        self._working_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to Neo4j
        logger.info(f"Connecting to Neo4j at {self._neo4j_uri}")
        self._driver = GraphDatabase.driver(
            self._neo4j_uri,
            auth=(self._neo4j_user, self._neo4j_password)
        )
        
        # Verify connection
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        # Check if indexing is needed
        if self._needs_indexing():
            logger.info("Indexing story into Neo4j...")
            self._ensure_vector_index()
            self._ingest_messages()
            try:
                self._indexed_marker.write_text("ok\n", encoding="utf-8")
            except Exception:
                logger.debug("Failed to write GraphRAG index marker", exc_info=True)
            logger.info("GraphRAG indexing complete")
        else:
            logger.info("GraphRAG storage already populated; skipping indexing")
        
        # Initialize retriever
        self._retriever = VectorRetriever(
            driver=self._driver,
            index_name=self._vector_index_name,
            embedder=None,  # We'll provide embeddings directly
        )
    
    def _needs_indexing(self) -> bool:
        """Check if indexing is needed."""
        if self._force_reindex:
            return True
        if not self._working_dir.exists():
            return True
        return not self._indexed_marker.exists()
    
    def _ensure_vector_index(self):
        """Create vector index if it doesn't exist."""
        embedding_dim = self._embedding_service.output_dimensionality
        
        with self._driver.session(database=self._neo4j_database) as session:
            # Check if index exists
            result = session.run(
                "SHOW INDEXES YIELD name WHERE name = $index_name RETURN count(*) as count",
                index_name=self._vector_index_name
            )
            index_exists = result.single()["count"] > 0
            
            if index_exists:
                logger.info(f"Vector index '{self._vector_index_name}' already exists")
            else:
                logger.info(f"Creating vector index '{self._vector_index_name}' with dimension {embedding_dim}")
                session.run(f"""
                    CREATE VECTOR INDEX {self._vector_index_name}
                    FOR (n:{self._node_label})
                    ON (n.{self._embedding_property})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info("Vector index created successfully")
    
    def _ingest_messages(self):
        """Ingest all messages as chunk nodes with embeddings."""
        logger.info(f"Ingesting {len(self._messages)} messages into Neo4j...")
        
        # Prepare chunks and generate embeddings
        chunks_data = []
        texts = []
        
        for msg in self._messages:
            chunk_id = f"chunk_{msg.id}"
            # Format text like naive RAG does
            text = self._format_message_text(msg)
            texts.append(text)
            chunks_data.append({
                "chunk_id": chunk_id,
                "message_id": msg.id,
                "sender": msg.sender,
                "receiver": msg.receiver,
                "timestamp": msg.timestamp,
                "text": text,
            })
        
        # Generate embeddings in batch
        logger.info("Generating embeddings for all messages...")
        embeddings = self._embedding_service.embed_batch(texts)
        
        # Add embeddings to chunks_data
        for chunk_data, embedding in zip(chunks_data, embeddings):
            chunk_data["embedding"] = embedding.tolist()
        
        # Ingest into Neo4j in batches
        batch_size = 100
        with self._driver.session(database=self._neo4j_database) as session:
            for i in range(0, len(chunks_data), batch_size):
                batch = chunks_data[i:i+batch_size]
                session.run(f"""
                    UNWIND $chunks AS chunk
                    CREATE (n:{self._node_label})
                    SET n.chunk_id = chunk.chunk_id,
                        n.message_id = chunk.message_id,
                        n.sender = chunk.sender,
                        n.receiver = chunk.receiver,
                        n.timestamp = chunk.timestamp,
                        n.text = chunk.text,
                        n.{self._embedding_property} = chunk.embedding
                """, chunks=batch)
                logger.info(f"Ingested batch {i//batch_size + 1}/{(len(chunks_data) + batch_size - 1)//batch_size}")
        
        logger.info(f"Successfully ingested {len(chunks_data)} chunks")
    
    def _format_message_text(self, msg: Message) -> str:
        """Format message into text for embedding and retrieval."""
        parts = []
        if msg.timestamp:
            parts.append(f"Message {msg.id} at {msg.timestamp}")
        else:
            parts.append(f"Message {msg.id}")
        parts.append(f"From: {msg.sender}")
        parts.append(f"To: {msg.receiver}")
        parts.append(f"Body: {msg.body}")
        return "\n".join(parts)
    
    def retrieve(
        self,
        question: str,
        threshold: float = 0.7,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a question using Neo4j vector search.
        
        Args:
            question: User's question
            threshold: Minimum similarity score (not used for Neo4j, but kept for interface)
            max_results: Maximum results to return (overrides top_k if provided)
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Embed the question
        logger.info(f"Retrieving chunks for question: '{question}'")
        query_embedding = self._embedding_service.embed_query(question)
        
        # Determine top_k
        top_k = max_results if max_results is not None else self._top_k
        
        # Retrieve using Neo4j vector search
        search_results = self._retriever.search(
            query_vector=query_embedding.tolist(),
            top_k=top_k
        )
        
        # Convert to our SearchResult format
        results = []
        for item in search_results.items:
            # item is a RetrieverResultItem with content and metadata
            # Parse the content (it's a string representation of the node)
            import ast
            node_dict = ast.literal_eval(item.content) if isinstance(item.content, str) and item.content.startswith('{') else {}
            score = item.metadata.get("score", 0.0) if item.metadata else 0.0
            
            # Create Chunk object
            chunk = Chunk(
                id=node_dict.get("chunk_id", ""),
                text=node_dict.get("text", ""),
                metadata={
                    "sender": node_dict.get("sender", ""),
                    "receiver": node_dict.get("receiver", ""),
                    "message_id": node_dict.get("message_id", ""),
                    "timestamp": node_dict.get("timestamp", ""),
                },
                source_message_ids=[node_dict.get("message_id", "")],
                embedding=None  # Don't need to load embedding back
            )
            
            results.append(SearchResult(chunk=chunk, score=float(score)))
        
        logger.info(f"Retrieved {len(results)} chunks from Neo4j")
        return results
    
    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            logger.info("Closing Neo4j connection")
            self._driver.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
