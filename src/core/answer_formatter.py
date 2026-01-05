"""
Answer formatter for displaying evidence from retrieved chunks.
"""
from typing import List
from src.core.models import SearchResult


class AnswerFormatter:
    """
    Formats evidence from search results for display to user.
    """
    
    def format_evidence(self, results: List[SearchResult]) -> List[str]:
        """
        Format search results as evidence citations.
        
        Args:
            results: List of search results
            
        Returns:
            List of formatted evidence strings
        """
        if not results:
            return ["[No evidence found]"]
        
        evidence = []
        for i, result in enumerate(results, 1):
            # Extract metadata
            chunk = result.chunk
            score = result.score
            
            # Format evidence entry
            evidence_str = f"{i}. [{score:.3f}] {chunk.text}"
            evidence.append(evidence_str)
        
        return evidence
