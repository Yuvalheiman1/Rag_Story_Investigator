from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np



# -------------------------
# Story domain models
# -------------------------

@dataclass(frozen=True)
class Message:
    """
    Represents a single message in the story.
    Parsed from the XML file.
    """
    id: str
    sender: str
    receiver: str
    body: str
    timestamp: str = ""  # ISO8601 or empty if not present

@dataclass
class Chunk:
    id: str                      # e.g., "chunk_m1"
    text: str                    # XML format preserved: <sender.../><receiver.../><body>...</body>
    metadata: dict               # {sender, receiver, message_id, chapter_id (optional), timestamp (optional)}
    source_message_ids: List[str] # ["m1"] - for traceability
    embedding: Optional[np.ndarray] = None

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    
@dataclass(frozen=True)
class RetrievedContext:
    """
    Represents context retrieved by a RAG system
    and passed to the LLM.
    """
    text: str
    source_message_ids: List[str] #metadata

@dataclass(frozen=True)
class Answer:
    """
    Final answer returned to the user.
    """
    answer_text: str
    evidence: List[str]
    explanation: Optional[str] = None


# -------------------------
# RAG Engine Contract
# -------------------------

class RagEngine(ABC):
    """
    Contract implemented by all RAG systems:
    - naive
    - LightRAG
    - nano-graphrag

    Note: the current app orchestrator (src/main.py) uses RAG engines as retrievers
    that return ranked chunks; prompt building + LLM answering happens outside the
    engine.
    """

    @abstractmethod
    def retrieve(
        self,
        question: str,
        threshold: float = 0.7,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """Retrieve relevant chunks for a question."""
        raise NotImplementedError

    def answer(self, question: str) -> Answer:
        """
        Optional higher-level API: given a user question, return a structured Answer.

        The default application flow does not rely on this; it calls `retrieve(...)`
        and then uses PromptBuilder + LLMClient.
        """
        raise NotImplementedError("RagEngine.answer() is not implemented")
