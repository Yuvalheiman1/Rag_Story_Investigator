import requests
import subprocess
from src.config_loader import ConfigLoader


class LLMClient:
    def __init__(self, config_path="config.yaml", model_name=None):
        self.cfg = ConfigLoader(config_path)
        self.model_name = model_name or self.cfg.get('lightrag.llm.model', 'gemma3:1b')
        self.ollama_api_url = self.cfg.get('lightrag.llm.api_url', 'http://localhost:11434/api/generate')
        self.max_tokens = self.cfg.get('lightrag.llm.max_tokens', 4000)
        self.request_timeout = self.cfg.get('lightrag.llm.timeout', 50)
        self.prompt_categorize = self.cfg.get('lightrag.llm.prompt_categorize', 'Categorize this email:')
        self.prompt_linkdin = self.cfg.get('lightrag.llm.prompt_linkdin', 'Search LinkedIn for:')
        self._ensure_model_downloaded()

    def _ensure_model_downloaded(self):
        """
        Checks if the Ollama model is downloaded. If not, downloads it using the Ollama CLI.
        """
        try:
            tags_url = self.cfg.get('lightrag.llm.tags_url', 'http://localhost:11434/api/tags')
            resp = requests.get(tags_url, timeout=10)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in tags]
            if self.model_name not in model_names:
                print(f"Model '{self.model_name}' not found locally. Downloading with 'ollama pull'...")
                result = subprocess.run([
                    "ollama", "pull", self.model_name
                ], capture_output=True, text=True, encoding="utf-8", errors="replace")
                if result.returncode != 0:
                    print(f"Failed to download model '{self.model_name}': {result.stderr}")
                else:
                    print(f"Model '{self.model_name}' downloaded successfully.")
        except Exception as e:
            print(f"Could not check or download Ollama model '{self.model_name}': {e}")

    def ask_llm(self, email_text, action="categorize_email", timeout_for_llm=None, raise_on_error: bool = False):
        """
        Sends the email text to the local Ollama LLM and returns the category.
        """
        if action == "categorize_email":
            prompt = f"{self.prompt_categorize} {email_text}"
        elif action == "search_linkedin":
            prompt = f"{self.prompt_linkdin} {email_text}"
        else:
            prompt = email_text

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "max_tokens": self.max_tokens
        }

        if timeout_for_llm is None:
            timeout_for_llm = self.request_timeout

        try:
            response = requests.post(self.ollama_api_url, json=payload, timeout=timeout_for_llm)
            response.raise_for_status()
            if response.ok:
                return response.json().get("response")
            else:
                print("Error:", response.text)
        except Exception as e:
            print("LLM error:", e)
            if raise_on_error:
                raise
            return "unknown"