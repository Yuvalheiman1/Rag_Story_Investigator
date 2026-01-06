"""
Configuration loader with dependency injection for RAG components.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import functools
import yaml

from src.rag.naive.chunker import MessageChunker
from src.core.embedding_service import EmbeddingService
from src.rag.naive.chunk_indexer import ChunkIndexer
from src.rag.naive.naive_rag import NaiveRAG
from src.core.prompt_builder import PromptBuilder
from src.core.llm_client import LLMClient
from src.core.models import Message

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads configuration and creates component instances via dependency injection.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML config file
        """
        # Load .env file if it exists
        from dotenv import load_dotenv
        load_dotenv()
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated key.
        
        Args:
            key: Dot-separated config key (e.g., 'naive.chunker.max_chunk_size')
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_default_rag_system(self) -> str:
        """Get the default RAG system from config."""
        return self.get('default_rag_system', 'naive')

    def is_lightrag_enabled(self) -> bool:
        """Whether LightRAG is enabled in config."""
        return bool(self.get('lightrag.enabled', False))

    def get_lightrag_working_dir(self) -> str:
        """Get LightRAG working dir (storage) path."""
        return self.get('lightrag.working_dir', str(Path(self.get_cache_dir()) / 'lightrag'))

    def get_lightrag_query_mode(self) -> str:
        return self.get('lightrag.query.mode', 'naive')

    def get_lightrag_only_need_context(self) -> bool:
        return bool(self.get('lightrag.query.only_need_context', True))
    
    def get_lightrag_user_prompt(self) -> Optional[str]:
        """Get custom user prompt for LightRAG queries."""
        return self.get('lightrag.query.user_prompt', None)

    def get_lightrag_embedding_batch_size(self) -> int:
        return int(self.get('lightrag.embedding.batch_size', 32))

    def get_lightrag_force_reindex(self) -> bool:
        return bool(self.get('lightrag.index.force_reindex', False))

    def get_lightrag_llm_provider(self) -> str:
        return str(self.get('lightrag.llm.provider', 'ollama')).lower()

    def get_lightrag_llm_model(self) -> str:
        return str(self.get('lightrag.llm.model', 'gemma3:1b'))

    def get_lightrag_llm_api_url(self) -> str:
        return str(self.get('lightrag.llm.api_url', 'http://localhost:11434/api/generate'))

    def get_lightrag_llm_max_tokens(self) -> int:
        return int(self.get('lightrag.llm.max_tokens', 4000))

    def get_lightrag_ollama_host(self) -> Optional[str]:
        host = self.get('lightrag.llm.host', None)
        return host if host not in ('', None) else None

    def get_lightrag_ollama_timeout(self) -> Optional[float]:
        timeout = self.get('lightrag.llm.timeout', None)
        return timeout if timeout not in ('', None) else None
    
    def is_graphrag_enabled(self) -> bool:
        """Whether GraphRAG is enabled in config."""
        return bool(self.get('graphrag.enabled', False))
    
    def get_graphrag_working_dir(self) -> str:
        """Get GraphRAG working dir (storage) path."""
        return self.get('graphrag.working_dir', str(Path(self.get_cache_dir()) / 'graphrag'))
    
    def get_graphrag_neo4j_uri(self) -> str:
        """Get Neo4j URI from environment or config."""
        return os.getenv('NEO4J_URI', self.get('graphrag.neo4j.uri', 'bolt://localhost:7687'))
    
    def get_graphrag_neo4j_username(self) -> str:
        """Get Neo4j username from environment or config."""
        return os.getenv('NEO4J_USERNAME', self.get('graphrag.neo4j.username', 'neo4j'))
    
    def get_graphrag_neo4j_password(self) -> str:
        """Get Neo4j password from environment or config."""
        return os.getenv('NEO4J_PASSWORD', self.get('graphrag.neo4j.password', 'password'))
    
    def get_graphrag_neo4j_database(self) -> str:
        """Get Neo4j database from environment or config."""
        return os.getenv('NEO4J_DATABASE', self.get('graphrag.neo4j.database', 'neo4j'))
    
    def get_graphrag_vector_index_name(self) -> str:
        return self.get('graphrag.index.vector_index_name', 'chunk_embeddings')
    
    def get_graphrag_node_label(self) -> str:
        return self.get('graphrag.index.node_label', 'Chunk')
    
    def get_graphrag_embedding_property(self) -> str:
        return self.get('graphrag.index.embedding_property', 'embedding')
    
    def get_graphrag_force_reindex(self) -> bool:
        return bool(self.get('graphrag.index.force_reindex', False))
    
    def get_graphrag_top_k(self) -> int:
        return int(self.get('graphrag.retrieval.top_k', 10))
    
    def get_graphrag_similarity_threshold(self) -> float:
        return float(self.get('graphrag.retrieval.similarity_threshold', 0.4))
    
    def get_story_path(self) -> str:
        """Get the story file path from config."""
        return self.get('story.path', 'data/story.xml')
    
    def get_cache_dir(self) -> str:
        """Get the cache directory from config."""
        return self.get('story.cache_dir', 'cache')
    
    def create_chunker(self) -> MessageChunker:
        """
        Create a MessageChunker instance from config.
        
        Returns:
            Configured MessageChunker
        """
        max_chunk_size = self.get('naive.chunker.max_chunk_size', 500)
        overlap = self.get('naive.chunker.overlap', 50)
        
        # logger.debug(f"Creating chunker (max_size={max_chunk_size}, overlap={overlap})")
        # return MessageChunker(max_chunk_size=max_chunk_size, overlap=overlap)
        logger.debug(f"Creating chunker (max_size={max_chunk_size})")
        return MessageChunker(max_chunk_size=max_chunk_size)
    
    def create_embedding_service(self) -> EmbeddingService:
        """
        Create an EmbeddingService instance from config.
        
        Returns:
            Configured EmbeddingService
        """
        model = self.get('naive.embedding.model', 'all-mpnet-base-v2')
        output_dim = self.get('naive.embedding.output_dimensionality', 768)
        device = self.get('naive.embedding.device', None)
        
        logger.debug(f"Creating local embedding service (model={model}, dim={output_dim})")
        return EmbeddingService(
            model=model,
            output_dimensionality=output_dim,
            device=device
        )
    
    def create_indexer(self, embedding_service: EmbeddingService) -> ChunkIndexer:
        """
        Create a ChunkIndexer instance from config.
        
        Args:
            embedding_service: The embedding service to use
            
        Returns:
            Configured ChunkIndexer
        """
        cache_dir = self.get_cache_dir()
        
        logger.debug(f"Creating indexer (cache_dir={cache_dir})")
        return ChunkIndexer(embedding_service, cache_dir=cache_dir)
    
    def create_prompt_builder(self) -> PromptBuilder:
        """
        Create a PromptBuilder instance from config.
        
        Returns:
            Configured PromptBuilder
        """
        max_length = self.get('prompt_builder.max_length', 4000)
        system_instructions = self.get('prompt_builder.system_instructions')
        
        logger.debug(f"Creating prompt builder (max_length={max_length})")
        
        if system_instructions:
            return PromptBuilder(
                system_instructions=system_instructions,
                max_length=max_length
            )
        else:
            return PromptBuilder(max_length=max_length)
    
    def create_llm_client(self) -> LLMClient:
        """
        Create an LLMClient instance from config with OpenAI fallback.
        
        Returns:
            Configured LLMClient
        """
        model = self.get('llm.model', 'gpt-5-nano')
        fallback_model = self.get('llm.fallback_model', 'gpt-4o-mini')
        temperature = self.get('llm.temperature', 0.7)
        max_tokens = self.get('llm.max_tokens', 1024)
        enable_fallback = self.get('llm.enable_fallback', True)
        
        logger.debug(f"Creating LLM client (model={model}, fallback={fallback_model})")
        return LLMClient(
            model=model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_fallback=enable_fallback
        )

    def create_lightrag_llm_model_func(self):
        """Create a LightRAG-compatible llm_model_func from config.

        Supports `openai` (recommended) and `ollama`.
        """
        provider = self.get_lightrag_llm_provider()

        if provider == 'openai':
            model = self.get_lightrag_llm_model()
            from lightrag.llm.openai import (
                gpt_4o_complete,
                gpt_4o_mini_complete,
                openai_complete,
            )

            if model == 'gpt-4o-mini':
                return gpt_4o_mini_complete
            if model == 'gpt-4o':
                return gpt_4o_complete
            return functools.partial(openai_complete, model=model)

        if provider == 'ollama':
            from src.rag.lightrag.lightrag_rag import ollama_http_complete

            model = self.get_lightrag_llm_model()

            # Bind config_path so the underlying HTTP client reads api_url/max_tokens
            # from the same config.yaml file that created this ConfigLoader.
            return functools.partial(
                ollama_http_complete,
                config_path=str(self.config_path),
                model=model,
            )

        raise ValueError(f"Unsupported LightRAG llm.provider: {provider}")

    def create_lightrag_embedding_func(self):
        """Create a LightRAG-compatible embedding_func from config.

        For simplicity (and to match upstream examples), this defaults to OpenAI embeddings.
        """
        provider = str(self.get('lightrag.embedding.provider', 'openai')).lower()

        if provider == 'openai':
            from lightrag.llm.openai import openai_embed

            # NOTE: openai_embed is already a wrapped EmbeddingFunc with attrs.
            return openai_embed

        if provider == 'local':
            # Local sentence-transformers embedding adapter.
            embedding_service = self.create_embedding_service()
            from src.rag.lightrag.lightrag_rag import make_lightrag_embedding_func

            return make_lightrag_embedding_func(embedding_service)

        raise ValueError(f"Unsupported LightRAG embedding.provider: {provider}")

    def create_lightrag_engine(self, messages: List[Message]):
        """Create a LightRAG engine that answers questions end-to-end."""
        if not self.is_lightrag_enabled():
            raise ValueError("LightRAG is disabled in config (lightrag.enabled: false)")

        from lightrag import QueryParam
        from src.rag.lightrag.lightrag_engine import LightRAGEngine, LightRAGEngineConfig

        llm_model_func = self.create_lightrag_llm_model_func()
        embedding_func = self.create_lightrag_embedding_func()

        query_param = QueryParam(
            mode=self.get_lightrag_query_mode(),
            only_need_context=self.get_lightrag_only_need_context(),
            stream=False,
            user_prompt=self.get_lightrag_user_prompt(),
        )

        engine_config = LightRAGEngineConfig(
            working_dir=self.get_lightrag_working_dir(),
            query_param=query_param,
            force_reindex=self.get_lightrag_force_reindex(),
        )

        return LightRAGEngine(
            messages=messages,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            config=engine_config,
        )
    
    def create_graphrag_engine(self, messages: List[Message]):
        """Create a GraphRAG engine with Neo4j vector retrieval."""
        if not self.is_graphrag_enabled():
            raise ValueError("GraphRAG is disabled in config (graphrag.enabled: false)")
        
        from src.rag.graphrag.graphrag_engine import GraphRAGEngine
        
        # Create embedding service for GraphRAG
        embedding_service = self.create_embedding_service()
        
        return GraphRAGEngine(
            messages=messages,
            embedding_service=embedding_service,
            neo4j_uri=self.get_graphrag_neo4j_uri(),
            neo4j_user=self.get_graphrag_neo4j_username(),
            neo4j_password=self.get_graphrag_neo4j_password(),
            neo4j_database=self.get_graphrag_neo4j_database(),
            vector_index_name=self.get_graphrag_vector_index_name(),
            node_label=self.get_graphrag_node_label(),
            embedding_property=self.get_graphrag_embedding_property(),
            working_dir=self.get_graphrag_working_dir(),
            force_reindex=self.get_graphrag_force_reindex(),
            top_k=self.get_graphrag_top_k(),
        )
    
    def create_naive_rag(self, messages: List[Message]) -> NaiveRAG:
        """
        Create a NaiveRAG instance with all dependencies from config.
        
        Args:
            messages: Pre-parsed story messages
            
        Returns:
            Fully configured NaiveRAG instance
        """
        logger.info("Creating Naive RAG system from config...")
        
        # Create all dependencies
        chunker = self.create_chunker()
        embedding_service = self.create_embedding_service()
        indexer = self.create_indexer(embedding_service)
        
        # Get additional config
        cache_name = self.get('naive.indexer.cache_name', 'story_naive')
        force_reindex = self.get('naive.indexer.force_reindex', False)
        
        # Create and return RAG engine
        return NaiveRAG(
            messages=messages,
            chunker=chunker,
            embedding_service=embedding_service,
            indexer=indexer,
            cache_name=cache_name,
            force_reindex=force_reindex
        )
    
    def get_similarity_threshold(self) -> float:
        """Get default similarity threshold from config."""
        return self.get('naive.similarity.default_threshold', 0.7)
    
    def get_max_results(self) -> int:
        """Get default max results from config."""
        return self.get('naive.similarity.default_max_results', 5)
    
    def configure_logging(self) -> None:
        """Configure logging from config settings."""
        level_str = self.get('logging.level', 'INFO')
        format_str = self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        level = getattr(logging, level_str.upper(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            format=format_str,
            force=True  # Override any existing config
        )
        
        logger.info(f"Logging configured: level={level_str}")
