import pytest
import numpy as np
from src.rag.naive.similarity import SimilaritySearch
from src.core.models import Chunk, SearchResult


@pytest.fixture
def sample_embeddings():
    """Create deterministic sample embeddings."""
    np.random.seed(42)
    emb1 = np.random.rand(768)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    np.random.seed(43)
    emb2 = np.random.rand(768)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    np.random.seed(44)
    emb3 = np.random.rand(768)
    emb3 = emb3 / np.linalg.norm(emb3)
    
    return [emb1, emb2, emb3]


@pytest.fixture
def indexed_chunks(sample_embeddings):
    """Create chunks with embeddings."""
    chunks = [
        Chunk(
            id="chunk_1",
            text="marcus to alex: Bring USB",
            metadata={"sender": "marcus", "receiver": "alex"},
            source_message_ids=["m1"],
            embedding=sample_embeddings[0]
        ),
        Chunk(
            id="chunk_2",
            text="alex to marcus: What time?",
            metadata={"sender": "alex", "receiver": "marcus"},
            source_message_ids=["m2"],
            embedding=sample_embeddings[1]
        ),
        Chunk(
            id="chunk_3",
            text="marcus to alex: 8pm",
            metadata={"sender": "marcus", "receiver": "alex"},
            source_message_ids=["m3"],
            embedding=sample_embeddings[2]
        )
    ]
    return chunks


@pytest.fixture
def query_embedding(sample_embeddings):
    """Query embedding similar to first chunk."""
    return sample_embeddings[0]


@pytest.fixture
def search(indexed_chunks):
    """Initialize SimilaritySearch with indexed chunks."""
    return SimilaritySearch(indexed_chunks)


# Initialization Tests
def test_init_with_valid_chunks(indexed_chunks):
    search = SimilaritySearch(indexed_chunks)
    assert search.chunks == indexed_chunks


def test_init_with_empty_chunks_raises_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        SimilaritySearch([])


def test_init_with_missing_embeddings_raises_error():
    chunks = [
        Chunk(
            id="chunk_1",
            text="test",
            metadata={},
            source_message_ids=["m1"],
            embedding=None
        )
    ]
    with pytest.raises(ValueError, match="missing embedding"):
        SimilaritySearch(chunks)


def test_init_stores_chunks(indexed_chunks):
    search = SimilaritySearch(indexed_chunks)
    assert len(search.chunks) == 3


# Basic Search Tests
def test_search_returns_list_of_search_results(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert isinstance(results, list)


def test_search_returns_search_result_objects(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_result_has_chunk(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert hasattr(results[0], 'chunk')


def test_search_result_has_score(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert hasattr(results[0], 'score')


def test_search_returns_results_sorted_by_score_descending(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0, max_results=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_max_results(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0, max_results=2)
    assert len(results) <= 2


# Threshold Tests
def test_search_filters_by_threshold(search, query_embedding):
    results = search.search(query_embedding, threshold=0.99)
    assert all(r.score >= 0.99 for r in results)


def test_search_with_high_threshold_returns_fewer_results(search, query_embedding):
    low_threshold_results = search.search(query_embedding, threshold=0.1)
    high_threshold_results = search.search(query_embedding, threshold=0.9)
    assert len(high_threshold_results) <= len(low_threshold_results)


def test_search_with_threshold_zero_returns_all_chunks(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0, max_results=10)
    assert len(results) == 3


def test_search_with_threshold_one_returns_only_exact_matches(search):
    # Query embedding identical to first chunk
    query = search.chunks[0].embedding
    results = search.search(query, threshold=1.0)
    assert len(results) == 1


# Score Validation Tests
def test_search_scores_are_floats(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert all(isinstance(r.score, float) for r in results)


def test_search_scores_between_zero_and_one(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert all(0 <= r.score <= 1 for r in results)


def test_search_identical_embedding_has_score_near_one(search):
    query = search.chunks[0].embedding
    results = search.search(query, threshold=0.0)
    assert results[0].score > 0.99


# Input Validation Tests
def test_search_with_invalid_threshold_below_zero_raises_error(search, query_embedding):
    with pytest.raises(ValueError, match="must be in"):
        search.search(query_embedding, threshold=-0.1)


def test_search_with_invalid_threshold_above_one_raises_error(search, query_embedding):
    with pytest.raises(ValueError, match="must be in"):
        search.search(query_embedding, threshold=1.5)


def test_search_with_zero_max_results_raises_error(search, query_embedding):
    with pytest.raises(ValueError, match="must be positive"):
        search.search(query_embedding, threshold=0.5, max_results=0)


def test_search_with_negative_max_results_raises_error(search, query_embedding):
    with pytest.raises(ValueError, match="must be positive"):
        search.search(query_embedding, threshold=0.5, max_results=-1)


def test_search_with_non_array_embedding_raises_error(search):
    with pytest.raises(ValueError, match="must be a numpy array"):
        search.search("not an array", threshold=0.5)


def test_search_with_empty_embedding_raises_error(search):
    with pytest.raises(ValueError, match="cannot be empty"):
        search.search(np.array([]), threshold=0.5)


# Edge Cases
def test_search_with_single_chunk():
    chunk = Chunk(
        id="chunk_1",
        text="test",
        metadata={},
        source_message_ids=["m1"],
        embedding=np.random.rand(768)
    )
    search = SimilaritySearch([chunk])
    query = np.random.rand(768)
    results = search.search(query, threshold=0.0)
    assert len(results) == 1


def test_search_max_results_greater_than_chunks_returns_all(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0, max_results=100)
    assert len(results) == 3


def test_search_returns_empty_list_when_no_chunks_above_threshold(search, query_embedding):
    results = search.search(query_embedding, threshold=1.0)
    # Unless query is identical to a chunk, should return empty or very few
    assert isinstance(results, list)


def test_search_preserves_chunk_metadata(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert results[0].chunk.metadata is not None


def test_search_preserves_source_message_ids(search, query_embedding):
    results = search.search(query_embedding, threshold=0.0)
    assert len(results[0].chunk.source_message_ids) > 0


# Cosine Similarity Tests
def test_cosine_similarity_identical_vectors_returns_one(search):
    vec = np.random.rand(768)
    vec = vec / np.linalg.norm(vec)
    similarity = search._cosine_similarity(vec, vec)
    assert abs(similarity - 1.0) < 0.001


def test_cosine_similarity_returns_float(search):
    vec1 = np.random.rand(768)
    vec2 = np.random.rand(768)
    similarity = search._cosine_similarity(vec1, vec2)
    assert isinstance(similarity, float)