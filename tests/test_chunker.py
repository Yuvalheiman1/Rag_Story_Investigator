import pytest
from src.rag.naive.chunker import MessageChunker
from src.core.models import Message, Chunk


@pytest.fixture
def chunker():
    """Default MessageChunker instance."""
    return MessageChunker()


@pytest.fixture
def single_message():
    """A simple message."""
    return Message(
        id="m1",
        sender="marcus",
        receiver="alex",
        body="Bring the USB drive tonight"
    )


@pytest.fixture
def multiple_messages():
    """Three messages in sequence."""
    return [
        Message(id="m1", sender="marcus", receiver="alex", body="Bring USB"),
        Message(id="m2", sender="alex", receiver="marcus", body="What time?"),
        Message(id="m3", sender="marcus", receiver="alex", body="8pm at cafe")
    ]


@pytest.fixture
def empty_body_message():
    """Message with empty body."""
    return Message(id="m1", sender="alice", receiver="bob", body="")


@pytest.fixture
def long_message():
    """Message that exceeds default max chunk size."""
    return Message(
        id="m_long",
        sender="alice",
        receiver="bob",
        body="x" * 1500
    )


# Basic Functionality Tests
def test_chunk_single_message_returns_one_chunk(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert len(result) == 1


def test_chunk_multiple_messages_returns_correct_count(chunker, multiple_messages):
    result = chunker.chunk_messages(multiple_messages)
    assert len(result) == 3


def test_chunk_id_format_is_correct(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert result[0].id == "chunk_m1"


def test_chunk_text_includes_sender(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert "marcus" in result[0].text


def test_chunk_text_includes_receiver(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert "alex" in result[0].text


def test_chunk_text_includes_body(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert "Bring the USB drive tonight" in result[0].text


# Metadata Tests
def test_chunk_metadata_contains_sender(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert result[0].metadata["sender"] == "marcus"


def test_chunk_metadata_contains_receiver(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert result[0].metadata["receiver"] == "alex"


def test_chunk_metadata_contains_message_id(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert result[0].metadata["message_id"] == "m1"


def test_chunk_source_message_ids_list_is_correct(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    assert result[0].source_message_ids == ["m1"]


# Order & Traceability Tests
def test_chunks_preserve_message_order(chunker, multiple_messages):
    result = chunker.chunk_messages(multiple_messages)
    assert [c.id for c in result] == ["chunk_m1", "chunk_m2", "chunk_m3"]


def test_each_chunk_links_to_correct_source_message(chunker, multiple_messages):
    result = chunker.chunk_messages(multiple_messages)
    assert result[0].source_message_ids == ["m1"]
    assert result[1].source_message_ids == ["m2"]
    assert result[2].source_message_ids == ["m3"]


def test_chunk_ids_are_unique(chunker, multiple_messages):
    result = chunker.chunk_messages(multiple_messages)
    chunk_ids = [c.id for c in result]
    assert len(chunk_ids) == len(set(chunk_ids))


# Edge Cases
def test_empty_message_list_returns_empty_chunks(chunker):
    result = chunker.chunk_messages([])
    assert result == []


def test_message_with_empty_body_creates_chunk(chunker, empty_body_message):
    result = chunker.chunk_messages([empty_body_message])
    assert len(result) == 1


def test_message_with_missing_sender_handled_gracefully(chunker):
    message = Message(id="m1", sender=None, receiver="bob", body="Test")
    result = chunker.chunk_messages([message])
    assert result[0].metadata["sender"] is None


def test_message_with_unicode_preserved(chunker):
    message = Message(
        id="m1",
        sender="alice",
        receiver="bob",
        body="Meeting at café ☕ with 🚀"
    )
    result = chunker.chunk_messages([message])
    assert "café ☕ with 🚀" in result[0].text


# Size Validation Tests
def test_message_exceeding_max_chunk_size_raises_error(chunker, long_message):
    with pytest.raises(ValueError, match="exceeds max chunk size"):
        chunker.chunk_messages([long_message])


def test_message_at_exact_max_chunk_size_succeeds():
    chunker = MessageChunker(max_chunk_size=200)
    message = Message(id="m1", sender="a", receiver="b", body="x" * 100)
    result = chunker.chunk_messages([message])
    assert len(result) == 1


def test_custom_max_chunk_size_parameter_works():
    chunker = MessageChunker(max_chunk_size=5000)
    message = Message(id="m1", sender="a", receiver="b", body="x" * 2000)
    result = chunker.chunk_messages([message])
    assert len(result) == 1


# Format Tests
def test_chunk_text_uses_natural_language_format(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    text = result[0].text
    assert "marcus to alex:" in text.lower() or ("marcus" in text and "alex" in text)


def test_chunk_text_is_concise(chunker, single_message):
    result = chunker.chunk_messages([single_message])
    # Natural format should be shorter than XML tags
    assert len(result[0].text) < 100


def test_chunk_with_none_sender_uses_unknown(chunker):
    message = Message(id="m1", sender=None, receiver="bob", body="Test message")
    result = chunker.chunk_messages([message])
    assert "unknown" in result[0].text.lower()


def test_chunk_with_none_receiver_uses_unknown(chunker):
    message = Message(id="m1", sender="alice", receiver=None, body="Test message")
    result = chunker.chunk_messages([message])
    assert "unknown" in result[0].text.lower()