"""
LLM client for generating answers using Google Gemini API.
"""
import logging
from os import getenv
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / '.env')

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for generating text responses using Google Gemini LLM.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Model to use (default: gemini-2.0-flash-exp)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it as environment variable or pass to constructor."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"LLM client initialized with model: {model}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a text response from a prompt.
        
        Args:
            prompt: The input prompt text
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If prompt is empty
            Exception: If API call fails
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.debug(f"Generating response (temp={temp}, max_tokens={max_tok})...")
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_tok
                )
            )
            
            # Extract text from response
            if response.text:
                logger.info(f"Generated response ({len(response.text)} chars)")
                return response.text
            else:
                logger.warning("Empty response from LLM")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise


if __name__ == "__main__":
    """Quick test of LLM client."""
    print("=" * 70)
    print("LLM Client Test")
    print("=" * 70)
    
    # Initialize client
    client = LLMClient()
    
    # Test prompt
    prompt = """You are a helpful assistant.

User Question: What is the capital of France?

Please answer the question."""
    
    print("\nPrompt:")
    print("-" * 70)
    print(prompt)
    
    print("\n\nGenerating response...")
    print("-" * 70)
    
    response = client.generate(prompt)
    
    print("\nResponse:")
    print("-" * 70)
    print(response)
    print()

