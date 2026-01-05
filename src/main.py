"""
Main entry point for the RAG Story Investigator.
Handles user interaction and orchestrates all RAG systems.
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.config_loader import ConfigLoader
from src.core.story_loader import parse_messages
from src.core.answer_formatter import AnswerFormatter
from src.core.models import Message, SearchResult, RetrievedContext

logger = logging.getLogger(__name__)


class StoryInvestigator:
    """
    Main application orchestrator using dependency injection.
    """
    
    def __init__(self, config: ConfigLoader, story_path: Optional[str] = None):
        """
        Initialize the investigator with config-based DI.
        
        Args:
            config: ConfigLoader instance
            story_path: Optional override for story path
        """
        self.config = config
        
        # Use provided path or get from config
        story_file = story_path or config.get_story_path()
        self.story_path = Path(story_file)
        
        if not self.story_path.exists():
            raise FileNotFoundError(f"Story file not found: {story_file}")
        
        # Parse story once at startup
        logger.info(f"Loading story from {story_file}...")
        self.messages = parse_messages(self.story_path)
        logger.info(f"Loaded {len(self.messages)} messages")
        
        # Initialize plumbing components from config
        self.prompt_builder = config.create_prompt_builder()
        self.llm_client = config.create_llm_client()
        self.answer_formatter = AnswerFormatter()
        
        # RAG-specific components (initialized on demand)
        self.rag_engine = None
        self.current_rag_type = None
    
    def select_rag_system(self, rag_type: str) -> None:
        """
        Initialize the selected RAG system using config-based DI.
        
        Args:
            rag_type: One of 'naive', 'lightrag', 'graphrag'
            
        Raises:
            ValueError: If invalid RAG type
        """
        rag_type = rag_type.lower()
        
        if rag_type == "naive":
            logger.info("Initializing Naive RAG system...")
            self.rag_engine = self.config.create_naive_rag(self.messages)
            self.current_rag_type = "naive"
            logger.info("Naive RAG initialized successfully")
            
        elif rag_type == "lightrag":
            raise NotImplementedError("LightRAG not implemented yet")
            
        elif rag_type == "graphrag":
            raise NotImplementedError("GraphRAG not implemented yet")
            
        else:
            raise ValueError(
                f"Invalid RAG type: {rag_type}. "
                f"Choose from: naive, lightrag, graphrag"
            )
    
    def answer_question(
        self,
        question: str,
        threshold: Optional[float] = None,
        max_results: Optional[int] = None
    ) -> dict:
        """
        Answer a question using the selected RAG system.
        
        Args:
            question: User's question
            threshold: Similarity threshold (None = use config default)
            max_results: Max results (None = use config default)
            
        Returns:
            Dictionary with question, results, context, prompt, and answer
        """
        if self.rag_engine is None:
            raise ValueError("No RAG system selected. Call select_rag_system() first.")
        
        # Use config defaults if not provided
        if threshold is None:
            threshold = self.config.get_similarity_threshold()
        if max_results is None:
            max_results = self.config.get_max_results()
        
        logger.info(f"Processing question: '{question}'")
        
        # Step 1: Retrieve relevant chunks
        results = self.rag_engine.retrieve(
            question=question,
            threshold=threshold,
            max_results=max_results
        )
        
        # Step 2: Build context from results
        context = self._build_context(results)
        
        # Step 3: Build prompt
        prompt = self.prompt_builder.build_prompt(question, context)
        
        # Step 4: Generate answer with LLM
        logger.debug("Generating answer with LLM...")
        try:
            answer_text = self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer_text = f"[Error generating answer: {e}]"
        
        # Step 5: Format evidence
        evidence = self.answer_formatter.format_evidence(results)
        
        return {
            "question": question,
            "answer": answer_text,
            "evidence": evidence,
            "results": results,
            "context": context,
            "prompt": prompt,
            "rag_type": self.current_rag_type
        }
    
    def _build_context(self, results: List[SearchResult]) -> RetrievedContext:
        """
        Build RetrievedContext from search results.
        
        Args:
            results: List of search results
            
        Returns:
            RetrievedContext object
        """
        if not results:
            return RetrievedContext(
                text="[No relevant context found]",
                source_message_ids=[]
            )
        
        # Combine chunk texts
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"--- Context {i} (score: {result.score:.3f}) ---")
            context_parts.append(result.chunk.text)
        
        context_text = "\n\n".join(context_parts)
        
        # Collect unique source message IDs
        source_message_ids = []
        seen = set()
        for result in results:
            for msg_id in result.chunk.source_message_ids:
                if msg_id not in seen:
                    seen.add(msg_id)
                    source_message_ids.append(msg_id)
        
        return RetrievedContext(
            text=context_text,
            source_message_ids=source_message_ids
        )


def interactive_mode(investigator: StoryInvestigator):
    """
    Run interactive question-answering mode.
    
    Args:
        investigator: Initialized StoryInvestigator instance
    """
    print("\n" + "=" * 70)
    print("Interactive Mode - Ask questions about the story")
    print("=" * 70)
    print("Type 'exit' or 'quit' to end, 'switch' to change RAG system\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if question.lower() == 'switch':
                rag_type = input("Select RAG system (naive/lightrag/graphrag): ").strip()
                try:
                    investigator.select_rag_system(rag_type)
                    print(f"✓ Switched to {rag_type} RAG\n")
                except (ValueError, NotImplementedError) as e:
                    print(f"❌ Error: {e}\n")
                continue
            
            # Answer the question
            print()
            response = investigator.answer_question(question)
            
            print("\n" + "=" * 70)
            print("ANSWER")
            print("=" * 70)
            print(response["answer"])
            
            print("\n" + "=" * 70)
            print("EVIDENCE")
            print("=" * 70)
            for evidence_item in response["evidence"]:
                print(evidence_item)
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            print(f"❌ Error: {e}\n")


def main():
    """Main entry point."""
    print("=" * 70)
    print("AI Investigator 1.0")
    print("Ask me any question about the story")
    print("=" * 70)
    print()
    
    # Load configuration
    config_path = "config.yaml"
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
    
    try:
        config = ConfigLoader(config_path)
        config.configure_logging()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure config.yaml exists in the project root.")
        sys.exit(1)
    
    # Get story path (CLI arg or config)
    story_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--story" and i + 1 < len(sys.argv):
            story_path = sys.argv[i + 1]
    
    # Initialize investigator
    try:
        investigator = StoryInvestigator(config, story_path=story_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    # Ask user to select RAG system
    print("Select RAG system:")
    print("  1. naive     - Simple embedding-based retrieval")
    print("  2. lightrag  - LightRAG (not implemented yet)")
    print("  3. graphrag  - Nano-GraphRAG (not implemented yet)")
    print()
    
    default_rag = config.get_default_rag_system()
    choice = input(f"Enter choice [1-3] (default: {default_rag}): ").strip()
    
    rag_map = {"1": "naive", "2": "lightrag", "3": "graphrag"}
    
    if not choice:
        rag_type = default_rag
    elif choice in rag_map:
        rag_type = rag_map[choice]
    else:
        rag_type = choice.lower()
    
    try:
        investigator.select_rag_system(rag_type)
    except (ValueError, NotImplementedError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    # Enter interactive mode
    interactive_mode(investigator)


if __name__ == "__main__":
    main()
