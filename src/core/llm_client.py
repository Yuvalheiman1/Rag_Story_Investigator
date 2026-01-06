"""
LLM client for generating answers using Google Gemini or OpenAI APIs with automatic fallback.
"""
import logging
from os import getenv
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / '.env')

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for generating text responses using Google Gemini with OpenAI fallback.
    Automatically falls back to OpenAI when Gemini quota is exceeded.
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        fallback_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        enable_fallback: bool = True
    ):
        """
        Initialize the LLM client with Gemini primary and OpenAI fallback.
        
        Args:
            gemini_api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Primary Gemini model to use
            fallback_model: OpenAI model for fallback (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            enable_fallback: Enable automatic fallback to OpenAI on quota errors
        """
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_fallback = enable_fallback
        
        # Initialize Gemini
        self.gemini_api_key = gemini_api_key or getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logger.info(f"Gemini client initialized with model: {model}")
        else:
            self.gemini_client = None
            logger.warning("GEMINI_API_KEY not found, Gemini disabled")
        
        # Initialize OpenAI fallback
        self.openai_api_key = openai_api_key or getenv("OPENAI_API_KEY")
        if self.openai_api_key and enable_fallback:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info(f"OpenAI fallback enabled with model: {fallback_model}")
        else:
            self.openai_client = None
            if enable_fallback:
                logger.warning("OPENAI_API_KEY not found, fallback disabled")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a text response from a prompt with automatic fallback.
        
        Args:
            prompt: The input prompt text
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If prompt is empty
            Exception: If both Gemini and OpenAI fail
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.debug(f"Generating response (temp={temp}, max_tokens={max_tok})...")
        
        # Try Gemini first
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temp,
                        max_output_tokens=max_tok
                    )
                )
                
                if response.text:
                    logger.info(f"Gemini generated response ({len(response.text)} chars)")
                    return response.text
                else:
                    logger.warning("Empty response from Gemini")
                    
            except ClientError as e:
                # Check if it's a quota/rate limit error (429)
                if e.code == 429 and self.enable_fallback and self.openai_client:
                    logger.warning(f"Gemini quota exceeded (429), falling back to OpenAI...")
                    return self._generate_openai(prompt, temp, max_tok)
                else:
                    logger.error(f"Gemini error: {e}")
                    if self.enable_fallback and self.openai_client:
                        logger.info("Attempting OpenAI fallback...")
                        return self._generate_openai(prompt, temp, max_tok)
                    raise
                    
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                if self.enable_fallback and self.openai_client:
                    logger.info("Attempting OpenAI fallback...")
                    return self._generate_openai(prompt, temp, max_tok)
                raise
        
        # If no Gemini client, try OpenAI directly
        if self.openai_client:
            return self._generate_openai(prompt, temp, max_tok)
        
        raise ValueError("No LLM client available (Gemini and OpenAI both disabled)")
    
    def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Generate response using OpenAI.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated text response
        """
        try:
            logger.debug(f"Using OpenAI {self.fallback_model}...")
            
            # gpt-5-nano only supports temperature=1 (default)
            params = {
                "model": self.fallback_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens
            }
            
            # Only add temperature if not gpt-5-nano
            if "gpt-5-nano" not in self.fallback_model.lower():
                params["temperature"] = temperature
            
            response = self.openai_client.chat.completions.create(**params)
            
            text = response.choices[0].message.content
            logger.info(f"OpenAI generated response ({len(text)} chars)")
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

