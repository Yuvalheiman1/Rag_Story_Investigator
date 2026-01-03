from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod


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

@dataclass
class Chunk:
    id: str                      # e.g., "chunk_m1"
    text: str                    # XML format preserved: <sender.../><receiver.../><body>...</body>
    metadata: dict               # {sender, receiver, message_id, chapter_id (optional)}
    source_message_ids: List[str] # ["m1"] - for traceability
    

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
    """

    @abstractmethod
    def answer(self, question: str) -> Answer:
        """
        Given a user question, return a structured Answer.
        """
        raise NotImplementedError
