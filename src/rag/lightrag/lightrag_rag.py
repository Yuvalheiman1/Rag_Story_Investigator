"""src.rag.lightrag.lightrag_rag

Step 2 (adapters): provide minimal functions LightRAG can call:
- `embedding_func(texts: list[str]) -> np.ndarray`
- `llm_model_func(prompt: str, ...) -> str`

This file also contains a tiny `__main__` smoke demo that:
1) initializes LightRAG storage
2) inserts a couple of short texts
3) runs a query

It intentionally keeps the integration surface small and does NOT yet wire into
the main app (that happens in later steps).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Union

import numpy as np

from lightrag import LightRAG, QueryParam

from src.core.embedding_service import EmbeddingService
from src.config_loader import ConfigLoader
from src.rag.lightrag.llm_service import LLMClient as OllamaHttpLLMClient


def _ensure_working_dir(path: str | Path) -> str:
	working_dir = str(path)
	Path(working_dir).mkdir(parents=True, exist_ok=True)
	return working_dir


def make_lightrag_embedding_func(
	embedding_service: EmbeddingService,
	*,
	max_token_size: int = 8192,
) -> Any:
	"""Create an async embedding function compatible with LightRAG.

	We reuse the repo's local sentence-transformers embeddings.
	"""

	async def embedding_func(texts: list[str], **kwargs: Any) -> np.ndarray:
		batch_size = int(kwargs.get("batch_size", 32))
		embeddings_list = embedding_service.embed_batch(texts, batch_size=batch_size)
		return np.asarray(embeddings_list, dtype=np.float32)

	# LightRAG uses these attributes (via its helper decorator) to understand
	# the embedding dimensionality and context window.
	embedding_dim = int(getattr(embedding_service, "output_dimensionality", 0) or 0)
	model_name = str(getattr(embedding_service, "model_name", "sentence-transformers"))
	if embedding_dim <= 0:
		# last resort: infer by embedding one string
		embedding_dim = int(embedding_service.embed_text("test").shape[0])

	try:
		from lightrag.utils import wrap_embedding_func_with_attrs

		return wrap_embedding_func_with_attrs(
			embedding_dim=embedding_dim,
			max_token_size=max_token_size,
			model_name=model_name,
		)(embedding_func)
	except Exception:
		# Fallback: attach expected attributes directly.
		setattr(embedding_func, "embedding_dim", embedding_dim)
		setattr(embedding_func, "max_token_size", max_token_size)
		setattr(embedding_func, "model_name", model_name)
		return embedding_func


async def ollama_complete(
	prompt: str,
	system_prompt: Optional[str] = None,
	history_messages: Optional[list[dict[str, str]]] = None,
	enable_cot: bool = False,
	keyword_extraction: bool = False,
	**kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
	"""Minimal Ollama-backed completion func compatible with LightRAG.

	Defaults to `gamma3:1b` (as requested) but can be overridden via:
	- `kwargs['model']`
	- `kwargs['hashing_kv'].global_config['llm_model_name']` (if present)
	- env var `OLLAMA_MODEL`
	"""
	_ = enable_cot  # not supported; keep signature compatible

	if keyword_extraction:
		# Some LightRAG flows request JSON output for keyword extraction.
		# We'll ask Ollama for JSON format when requested.
		kwargs.setdefault("format", "json")

	# IMPORTANT: LightRAG may pass a default `llm_model_name` inside hashing_kv
	# (often an OpenAI model like gpt-4o-mini). For Ollama we must NOT pick that
	# up accidentally. Prefer an explicit model or OLLAMA_MODEL, else gamma3:1b.
	model = kwargs.pop("model", None) or os.getenv("OLLAMA_MODEL") or "gamma3:1b"

	host = kwargs.pop("host", None) or os.getenv("OLLAMA_HOST")
	timeout = kwargs.pop("timeout", None)

	# LightRAG may pass parameters we don't support; ignore safely.
	kwargs.pop("max_tokens", None)
	kwargs.pop("response_format", None)
	kwargs.pop("hashing_kv", None)

	# Streaming is optional; keep it off by default for simplicity.
	stream = bool(kwargs.pop("stream", False))

	import ollama

	client = ollama.AsyncClient(host=host, timeout=timeout)
	messages: list[dict[str, str]] = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	if history_messages:
		messages.extend(history_messages)
	messages.append({"role": "user", "content": prompt})

	if stream:
		response = await client.chat(model=model, messages=messages, stream=True, **kwargs)

		async def inner() -> AsyncIterator[str]:
			async for chunk in response:
				yield chunk["message"]["content"]

		return inner()

	response = await client.chat(model=model, messages=messages, **kwargs)
	return response["message"]["content"]


_OLLAMA_HTTP_CLIENTS: dict[str, OllamaHttpLLMClient] = {}


async def ollama_http_complete(
	prompt: str,
	system_prompt: Optional[str] = None,
	history_messages: Optional[list[dict[str, str]]] = None,
	enable_cot: bool = False,
	keyword_extraction: bool = False,
	**kwargs: Any,
) -> str:
	"""LLM completion using our existing requests-based Ollama client.

	This keeps configuration centralized in config.yaml via ConfigLoader.
	"""
	_ = enable_cot
	_ = keyword_extraction
	_ = history_messages

	config_path = str(kwargs.pop("config_path", "config.yaml"))
	client = _OLLAMA_HTTP_CLIENTS.get(config_path)
	if client is None:
		client = OllamaHttpLLMClient(config_path=config_path)
		_OLLAMA_HTTP_CLIENTS[config_path] = client

	# Allow LightRAG/DI to override the model explicitly.
	model = kwargs.pop("model", None)
	if model:
		client.model_name = model

	full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
	# Run blocking HTTP call in a worker thread.
	return await asyncio.to_thread(
		client.ask_llm,
		full_prompt,
		action="freeform",
		timeout_for_llm=client.request_timeout,
		raise_on_error=True,
	)


async def initialize_lightrag(
	*,
	working_dir: str | Path,
	embedding_service: EmbeddingService,
	llm_model_func: Any = ollama_http_complete,
) -> LightRAG:
	working_dir_str = _ensure_working_dir(working_dir)
	embedding_func = make_lightrag_embedding_func(embedding_service)
	rag = LightRAG(
		working_dir=working_dir_str,
		embedding_func=embedding_func,
		llm_model_func=llm_model_func,
	)
	await rag.initialize_storages()
	return rag


async def _smoke_demo() -> None:
	cfg = ConfigLoader("config.yaml")
	if not cfg.is_lightrag_enabled():
		print("LightRAG is disabled in config.yaml (lightrag.enabled: false)")
		print("Set lightrag.enabled: true to run this demo.")
		return

	print("=" * 70)
	print("LightRAG Smoke Demo (Ollama + local embeddings)")
	print("=" * 70)

	embedding_service = cfg.create_embedding_service()
	llm_model_func = cfg.create_lightrag_llm_model_func()
	rag = await initialize_lightrag(
		working_dir=cfg.get_lightrag_working_dir(),
		embedding_service=embedding_service,
		llm_model_func=llm_model_func,
	)
	try:
		await rag.ainsert(
			"Marcus asked Alex to bring the USB drive to the meeting. "
			"Alex later said he forgot it at home."
		)
		await rag.ainsert("The meeting was scheduled for 8pm.")

		question = "Who was supposed to bring the USB?"
		result = await rag.aquery(
			question,
			param=QueryParam(
				mode=cfg.get_lightrag_query_mode(),
				only_need_context=cfg.get_lightrag_only_need_context(),
			),
		)

		print("\nQuestion:")
		print(question)
		print("\nReturned context:")
		print(result)
	finally:
		await rag.finalize_storages()


if __name__ == "__main__":
	asyncio.run(_smoke_demo())

