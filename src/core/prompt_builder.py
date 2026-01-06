from typing import Optional
from src.core.models import RetrievedContext

DEFAULT_SYSTEM_INSTRUCTIONS = """You are AI Investigator 1.0, an advanced AI assistant specialized in analyzing story messages and answering questions based on evidence.

    Your responsibilities:
    - Answer ONLY the question asked - be short and precise, no extra commentary
    - Base your answer solely on the provided story context
    - If you don't know, briefly explain why (not in the story, not conclusive, etc)
    - Cite specific evidence from the messages with timestamps when available
    - Be factual and concise in your responses
    - Format your response with:
      1. A direct, brief answer to the question (1-3 sentences)
      2. An "Evidence" section listing relevant messages with timestamps
    - For evidence, format each item as: "(at timestamp) sender to receiver: message content"""


class PromptBuilder:
    """
    Assembles prompts for the LLM by combining system instructions,
    story context, and user questions.
    """
    
    def __init__(
        self,
        system_instructions: Optional[str] = None,
        max_length: int = 3000
    ):
        """
        Initialize the PromptBuilder.
        
        Args:
            system_instructions: Custom instructions for the LLM.
                                If None, uses DEFAULT_SYSTEM_INSTRUCTIONS.
            max_length: Maximum allowed length for the final prompt.
        """
        self.system_instructions = (
            system_instructions if system_instructions is not None
            else DEFAULT_SYSTEM_INSTRUCTIONS
        )
        self.max_length = max_length

    def build_prompt(self, question: str, context: RetrievedContext) -> str:
        """
        Build the prompt to send to the LLM.
        
        Args:
            question: The user's question
            context: Retrieved story context with messages
            
        Returns:
            Formatted prompt string ready for LLM
            
        Raises:
            ValueError: If the final prompt exceeds max_length
        """
        # Build prompt sections
        sections = []
        
        # Add system instructions if not empty
        if self.system_instructions.strip():
            sections.append("System Instructions:")
            sections.append(self.system_instructions.strip())
            sections.append("")  # blank line
        
        # Add story context
        sections.append("Story Context:")
        if context.text.strip():
            sections.append(context.text.strip())
        else:
            sections.append("[No context available]")
        sections.append("")  # blank line
        
        # Add user question
        sections.append("User Question:")
        sections.append(question.strip() if question.strip() else "[No question provided]")
        
        # Join all sections
        prompt = "\n".join(sections)
        
        # Validate length
        if len(prompt) > self.max_length:
            raise ValueError(
                f"Prompt exceeds max length: {len(prompt)} > {self.max_length}"
            )
        
        return prompt
    
    

if __name__ == "__main__":
    print("=" * 70)
    print("PromptBuilder - Simple Manual Test")
    print("=" * 70)
    
    # Test 1: Basic prompt building
    print("\n[TEST 1] Basic prompt with default settings")
    print("-" * 70)
    context = RetrievedContext(
        text="<sender ref='marcus'/><receiver ref='alex'/><body>Bring the USB drive tonight at 8pm</body>",
        source_message_ids=["msg_12"]
    )
    builder = PromptBuilder()
    prompt = builder.build_prompt("Who requested the USB drive?", context)
    print(prompt)
    print(f"\nPrompt length: {len(prompt)} characters")
    
    # Test 2: Custom system instructions
    print("\n" + "=" * 70)
    print("[TEST 2] Custom system instructions")
    print("-" * 70)
    custom_builder = PromptBuilder(
        system_instructions="You are a detective. Find clues in the messages."
    )
    prompt2 = custom_builder.build_prompt("What time was mentioned?", context)
    print(prompt2)
    print(f"\nPrompt length: {len(prompt2)} characters")
    
    # Test 3: Multiple messages context
    print("\n" + "=" * 70)
    print("[TEST 3] Multiple messages")
    print("-" * 70)
    multi_context = RetrievedContext(
        text="""<sender ref='marcus'/><receiver ref='alex'/><body>Bring the USB</body>
<sender ref='alex'/><receiver ref='marcus'/><body>Sure, where should we meet?</body>
<sender ref='marcus'/><receiver ref='alex'/><body>At the café on 5th street</body>""",
        source_message_ids=["msg_12", "msg_13", "msg_14"]
    )
    prompt3 = builder.build_prompt("Where did they plan to meet?", multi_context)
    print(prompt3)
    print(f"\nPrompt length: {len(prompt3)} characters")
    
    # Test 4: Length validation
    print("\n" + "=" * 70)
    print("[TEST 4] Length validation (should fail)")
    print("-" * 70)
    tiny_builder = PromptBuilder(max_length=100)
    try:
        tiny_builder.build_prompt("Question?", context)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test 5: Empty context handling
    print("\n" + "=" * 70)
    print("[TEST 5] Empty context")
    print("-" * 70)
    empty_context = RetrievedContext(text="", source_message_ids=[])
    prompt5 = builder.build_prompt("What happened?", empty_context)
    print(prompt5)
    print(f"\nPrompt length: {len(prompt5)} characters")
    
    print("\n" + "=" * 70)
    print("All manual tests completed!")
    print("=" * 70)
