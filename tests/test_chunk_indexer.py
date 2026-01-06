import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock
from src.rag.naive.chunk_indexer import ChunkIndexer
from src.core.models import Chunk


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service that returns predictable embeddings."""
    service = Mock()
    
    def mock_embed_batch(texts, task_type=None):
        """Return normalized random embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Deterministic embedding based on text
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return embeddings
    
    service.embed_batch = Mock(side_effect=mock_embed_batch)
    return service


@pytest.fixture
def sample_chunks():
    """Create sample chunks without embeddings."""
    return [
        Chunk(
            id="chunk_1",
            text="marcus to alex: Bring USB",
            metadata={"sender": "marcus", "receiver": "alex", "message_id": "m1"},
            source_message_ids=["m1"]
        ),
        Chunk(
            id="chunk_2",
            text="alex to marcus: What time?",
            metadata={"sender": "alex", "receiver": "marcus", "message_id": "m2"},
            source_message_ids=["m2"]
        ),
        Chunk(
            id="chunk_3",
            text="marcus to alex: 8pm",
            metadata={"sender": "marcus", "receiver": "alex", "message_id": "m3"},
            source_message_ids=["m3"]
        )
    ]


@pytest.fixture
def indexer(mock_embedding_service, tmp_path):
    """Create indexer with mock service and temp cache dir."""
    return ChunkIndexer(mock_embedding_service, cache_dir=str(tmp_path / "cache"))


# Initialization Tests
def test_init_creates_cache_directory(mock_embedding_service, tmp_path):
    cache_dir = tmp_path / "test_cache"
    indexer = ChunkIndexer(mock_embedding_service, cache_dir=str(cache_dir))
    assert cache_dir.exists()


def test_init_stores_embedding_service(mock_embedding_service, tmp_path):
    indexer = ChunkIndexer(mock_embedding_service, cache_dir=str(tmp_path))
    assert indexer.embedding_service is mock_embedding_service


# Index Tests
def test_index_returns_chunks_with_embeddings(indexer, sample_chunks):
    result = indexer.index(sample_chunks)
    assert result[0].embedding is not None


def test_index_calls_embed_batch(indexer, sample_chunks):
    indexer.index(sample_chunks)
    assert indexer.embedding_service.embed_batch.called


def test_index_embeds_all_chunks(indexer, sample_chunks):
    result = indexer.index(sample_chunks)
    assert all(chunk.embedding is not None for chunk in result)


def test_index_returns_same_chunks_list(indexer, sample_chunks):
    result = indexer.index(sample_chunks)
    assert result is sample_chunks


def test_index_with_empty_chunks_raises_error(indexer):
    with pytest.raises(ValueError, match="cannot be empty"):
        indexer.index([])


# Cache Save/Load Tests
def test_save_to_cache_creates_file(indexer, sample_chunks):
    indexed = indexer.index(sample_chunks)
    indexer.save_to_cache(indexed, "test")
    cache_path = indexer._get_cache_path("test")
    assert cache_path.exists()


def test_save_to_cache_without_embeddings_raises_error(indexer, sample_chunks):
    with pytest.raises(ValueError, match="missing embedding"):
        indexer.save_to_cache(sample_chunks, "test")


def test_load_from_cache_returns_chunks(indexer, sample_chunks):
    indexed = indexer.index(sample_chunks)
    indexer.save_to_cache(indexed, "test")
    loaded = indexer.load_from_cache("test")
    assert len(loaded) == 3


def test_load_from_cache_preserves_embeddings(indexer, sample_chunks):
    indexed = indexer.index(sample_chunks)
    indexer.save_to_cache(indexed, "test")
    loaded = indexer.load_from_cache("test")
    assert np.allclose(loaded[0].embedding, indexed[0].embedding)


def test_load_from_cache_nonexistent_returns_none(indexer):
    result = indexer.load_from_cache("nonexistent")
    assert result is None


def test_load_from_cache_corrupted_returns_none(indexer, tmp_path):
    cache_path = indexer._get_cache_path("corrupted")
    cache_path.write_text("invalid data")
    result = indexer.load_from_cache("corrupted")
    assert result is None


# Index and Cache Tests
def test_index_and_cache_first_call_embeds(indexer, sample_chunks):
    result = indexer.index_and_cache(sample_chunks, "test")
    assert result[0].embedding is not None


def test_index_and_cache_first_call_creates_cache(indexer, sample_chunks):
    indexer.index_and_cache(sample_chunks, "test")
    cache_path = indexer._get_cache_path("test")
    assert cache_path.exists()


def test_index_and_cache_second_call_loads_from_cache(indexer, sample_chunks):
    indexer.index_and_cache(sample_chunks, "test")
    indexer.embedding_service.embed_batch.reset_mock()
    
    result = indexer.index_and_cache(sample_chunks, "test")
    assert not indexer.embedding_service.embed_batch.called


def test_index_and_cache_with_force_reindex_ignores_cache(indexer, sample_chunks):
    indexer.index_and_cache(sample_chunks, "test")
    indexer.embedding_service.embed_batch.reset_mock()
    
    indexer.index_and_cache(sample_chunks, "test", force_reindex=True)
    assert indexer.embedding_service.embed_batch.called


def test_index_and_cache_with_empty_chunks_raises_error(indexer):
    with pytest.raises(ValueError, match="cannot be empty"):
        indexer.index_and_cache([], "test")


# Clear Cache Tests
def test_clear_cache_removes_specific_file(indexer, sample_chunks):
    indexed = indexer.index(sample_chunks)
    indexer.save_to_cache(indexed, "test")
    
    indexer.clear_cache("test")
    cache_path = indexer._get_cache_path("test")
    assert not cache_path.exists()


def test_clear_cache_all_removes_all_files(indexer, sample_chunks):
    indexed = indexer.index(sample_chunks)
    indexer.save_to_cache(indexed, "test1")
    indexer.save_to_cache(indexed, "test2")
    
    indexer.clear_cache()
    assert not indexer._get_cache_path("test1").exists()
    assert not indexer._get_cache_path("test2").exists()


def test_clear_cache_nonexistent_does_not_raise_error(indexer):
    indexer.clear_cache("nonexistent")
    # Should not raise error


# Edge Cases
def test_index_single_chunk(indexer):
    chunk = Chunk(
        id="chunk_1",
        text="test",
        metadata={},
        source_message_ids=["m1"]
    )
    result = indexer.index([chunk])
    assert result[0].embedding is not None


def test_cache_path_format(indexer):
    path = indexer._get_cache_path("my_cache")
    assert path.name == "my_cache.pkl"
    assert path.parent == indexer.cache_dir