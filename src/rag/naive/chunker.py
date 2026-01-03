from typing import List
from src.core.models import Message, Chunk
from typing import List, Optional

class MessageChunker:
    """
    Converts individual messages into chunks for embedding.
    Each message becomes one chunk in natural language format.
    """
    
    def __init__(self, max_chunk_size: int = 1000):
        """
        Initialize the MessageChunker.
        
        Args:
            max_chunk_size: Maximum characters allowed per chunk.
                          Raises ValueError if a message exceeds this.
        """
        self.max_chunk_size = max_chunk_size
    
    def chunk_messages(self, messages: List[Message]) -> List[Chunk]:
        """
        Convert each message into a chunk.
        
        Args:
            messages: List of parsed Message objects
            
        Returns:
            List of Chunk objects, one per message
            
        Raises:
            ValueError: If any message exceeds max_chunk_size
        """
        if not messages:
            return []
        
        chunks = []
        
        for message in messages:
            # Build chunk text in XML format
            chunk_text = self._build_chunk_text(message)
            
            # Validate size
            if len(chunk_text) > self.max_chunk_size:
                raise ValueError(
                    f"Message '{message.id}' exceeds max chunk size: "
                    f"{len(chunk_text)} > {self.max_chunk_size}"
                )
            
            # Create chunk
            chunk = Chunk(
                id=f"chunk_{message.id}",
                text=chunk_text,
                metadata={
                    "sender": message.sender,
                    "receiver": message.receiver,
                    "message_id": message.id
                },
                source_message_ids=[message.id]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _build_chunk_text(self, message: Message) -> str:
        """
        Build the chunk text in natural language format.
        
        Args:
            message: Message object to convert
            
        Returns:
            Natural language string: "sender to receiver: body"
        """
        sender = message.sender if message.sender else "unknown"
        receiver = message.receiver if message.receiver else "unknown"
        body = message.body if message.body else ""
        
        return f"{sender} to {receiver}: {body}"


if __name__ == "__main__":
    print("=" * 70)
    print("MessageChunker - Simple Manual Test")
    print("=" * 70)
    
    # Test 1: Single message chunking
    print("\n[TEST 1] Single message chunking")
    print("-" * 70)
    message = Message(
        id="m1",
        sender="marcus",
        receiver="alex",
        body="Bring the USB drive tonight at 8pm"
    )
    
    chunker = MessageChunker()
    chunks = chunker.chunk_messages([message])
    
    print(f"Input: 1 message")
    print(f"Output: {len(chunks)} chunk(s)")
    print(f"\nChunk ID: {chunks[0].id}")
    print(f"Chunk text:\n{chunks[0].text}")
    print(f"\nMetadata: {chunks[0].metadata}")
    print(f"Source IDs: {chunks[0].source_message_ids}")
    
    # Test 2: Multiple messages
    print("\n" + "=" * 70)
    print("[TEST 2] Multiple messages chunking")
    print("-" * 70)
    messages = [
        Message(id="m1", sender="marcus", receiver="alex", body="Bring USB"),
        Message(id="m2", sender="alex", receiver="marcus", body="What time?"),
        Message(id="m3", sender="marcus", receiver="alex", body="8pm at cafe")
    ]
    
    chunks = chunker.chunk_messages(messages)
    print(f"Input: {len(messages)} messages")
    print(f"Output: {len(chunks)} chunks")
    print(f"\nChunk IDs: {[c.id for c in chunks]}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.text)
    
    # Test 3: Unicode handling
    print("\n" + "=" * 70)
    print("[TEST 3] Unicode and special characters")
    print("-" * 70)
    unicode_message = Message(
        id="m_unicode",
        sender="alice",
        receiver="bob",
        body="Meeting at café ☕ with rocket 🚀"
    )
    
    chunks = chunker.chunk_messages([unicode_message])
    print(f"Chunk text:\n{chunks[0].text}")
    
    # Test 4: Empty body
    print("\n" + "=" * 70)
    print("[TEST 4] Empty body handling")
    print("-" * 70)
    empty_message = Message(id="m_empty", sender="alice", receiver="bob", body="")
    chunks = chunker.chunk_messages([empty_message])
    print(f"Chunk created: {chunks[0].id}")
    print(f"Chunk text:\n{chunks[0].text}")
    
    # Test 5: Size validation
    print("\n" + "=" * 70)
    print("[TEST 5] Size validation (should fail)")
    print("-" * 70)
    tiny_chunker = MessageChunker(max_chunk_size=50)
    long_message = Message(id="m_long", sender="a", receiver="b", body="x" * 100)
    
    try:
        tiny_chunker.chunk_messages([long_message])
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test 6: Custom max size
    print("\n" + "=" * 70)
    print("[TEST 6] Custom max chunk size")
    print("-" * 70)
    large_chunker = MessageChunker(max_chunk_size=5000)
    large_message = Message(
        id="m_large",
        sender="sender",
        receiver="receiver",
        body="This is a long message. " * 50
    )
    
    chunks = large_chunker.chunk_messages([large_message])
    print(f"✓ Successfully chunked large message")
    print(f"Chunk size: {len(chunks[0].text)} characters")
    
    print("\n" + "=" * 70)
    print("All manual tests completed!")
    print("=" * 70)