"""
Configuration loader with dependency injection for RAG components.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
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
