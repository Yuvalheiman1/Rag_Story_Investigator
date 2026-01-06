"""
LLM client for generating answers using OpenAI APIs.
"""
import logging
from os import getenv
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / '.env')

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for generating text responses using OpenAI.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        fallback_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        enable_fallback: bool = True
    ):
        """
        Initialize the LLM client with OpenAI.
        
        Args:
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Primary OpenAI model to use (gpt-5-nano, gpt-4o, gpt-4o-mini)
            fallback_model: Fallback OpenAI model
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            enable_fallback: Enable automatic fallback to fallback_model on errors
        """
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_fallback = enable_fallback
        
        # Initialize OpenAI
        self.openai_api_key = openai_api_key or getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info(f"OpenAI client initialized with model: {model}")
        else:
            self.openai_client = None
            logger.error("OPENAI_API_KEY not found")
    
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
            ValueError: If prompt is empty or no client available
            Exception: If OpenAI call fails
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not self.openai_client:
            raise ValueError("OpenAI client not available - check OPENAI_API_KEY")
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.debug(f"Generating response (temp={temp}, max_tokens={max_tok})...")
        
        # Try primary model first
        try:
            return self._generate_openai(self.model, prompt, temp, max_tok)
        except Exception as e:
            logger.error(f"OpenAI {self.model} error: {e}")
            
            # Try fallback if enabled and different model
            if self.enable_fallback and self.fallback_model != self.model:
                logger.info(f"Falling back to {self.fallback_model}...")
                return self._generate_openai(self.fallback_model, prompt, temp, max_tok)
            raise
    
    def _generate_openai(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Generate response using OpenAI.
        
        Args:
            model: OpenAI model name
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated text response
        """
        try:
            logger.debug(f"Using OpenAI {model}...")
            
            # gpt-5-nano only supports temperature=1 (default)
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens
            }
            
            # Only add temperature if not gpt-5-nano
            if "gpt-5-nano" not in model.lower():
                params["temperature"] = temperature
            
            response = self.openai_client.chat.completions.create(**params)
            
            text = response.choices[0].message.content
            if text:
                logger.info(f"OpenAI generated response ({len(text)} chars)")
                logger.debug(f"Response preview: {text[:200]}...")
            else:
                logger.warning(f"Empty response from OpenAI")
                logger.warning(f"Full response: {response}")
            return text
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
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

