"""LightRAG engine wrapper for the console app.

The existing app architecture assumes a synchronous CLI loop.
LightRAG is async-first and can generate answers end-to-end.

This wrapper:
- initializes LightRAG storages
- indexes the story messages once (cached in working_dir)
- exposes a synchronous `answer(question)` method returning Answer
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam

from src.core.models import Answer, Message, RagEngine

logger = logging.getLogger(__name__)


def _load_dotenv_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

class _AsyncLoopRunner:
    """Run asyncio coroutines on a single dedicated event loop.

    LightRAG creates asyncio primitives (e.g., PriorityQueue workers) that are
    bound to the event loop they were created on. Calling asyncio.run() multiple
    times creates multiple loops and can trigger cross-loop errors.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._loop is None:
            raise RuntimeError("Failed to start asyncio loop thread")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def run(self, coro: Any):
        if self._loop is None:
            raise RuntimeError("Asyncio loop not initialized")
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def _format_message_doc(message: Message) -> str:
    ts = f" at {message.timestamp}" if message.timestamp else ""
    return (
        f"Message {message.id}{ts}\n"
        f"From: {message.sender}\n"
        f"To: {message.receiver}\n"
        f"Body: {message.body}"
    )


@dataclass(frozen=True)
class LightRAGEngineConfig:
    working_dir: str
    query_param: QueryParam
    force_reindex: bool = False


class LightRAGEngine(RagEngine):
    def __init__(
        self,
        *,
        messages: list[Message],
        llm_model_func: Any,
        embedding_func: Any,
        config: LightRAGEngineConfig,
    ) -> None:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        _load_dotenv_from_repo_root()

        if not getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Put it in .env or your environment variables."
            )

        self._messages = messages
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func
        self._config = config
        self._working_dir = Path(config.working_dir)

        self._rag: Optional[LightRAG] = None
        self._initialized = False

        self._runner = _AsyncLoopRunner()
        self._indexed_marker = self._working_dir / ".indexed"

        self._initialize_sync()

    def _initialize_sync(self) -> None:
        self._runner.run(self._ainitialize())

    async def _ainitialize(self) -> None:
        if self._initialized:
            return

        if self._config.force_reindex and self._working_dir.exists():
            logger.info(f"LightRAG force reindex: removing {self._working_dir}")
            shutil.rmtree(self._working_dir, ignore_errors=True)

        # Refresh marker path (working_dir may have been deleted)
        self._indexed_marker = self._working_dir / ".indexed"

        self._working_dir.mkdir(parents=True, exist_ok=True)

        rag = LightRAG(
            working_dir=str(self._working_dir),
            llm_model_func=self._llm_model_func,
            embedding_func=self._embedding_func,
        )
        await rag.initialize_storages()
        self._rag = rag

        if self._needs_indexing():
            logger.info("Indexing story into LightRAG...")
            docs = [_format_message_doc(m) for m in self._messages]
            ids = [m.id for m in self._messages]
            await rag.ainsert(docs, ids=ids)
            try:
                self._indexed_marker.write_text("ok\n", encoding="utf-8")
            except Exception:
                logger.debug("Failed to write LightRAG index marker", exc_info=True)
            logger.info("LightRAG indexing complete")
        else:
            logger.info("LightRAG storage already populated; skipping indexing")

        self._initialized = True

    def _needs_indexing(self) -> bool:
        if self._config.force_reindex:
            return True
        if not self._working_dir.exists():
            return True
        return not self._indexed_marker.exists()

    def retrieve(self, question: str, threshold: float = 0.7, max_results: Optional[int] = None):
        raise NotImplementedError(
            "LightRAGEngine is configured for end-to-end answering, not chunk retrieval."
        )

    def answer(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not self._rag or not self._initialized:
            self._initialize_sync()

        result = self._runner.run(self._rag.aquery(question, param=self._config.query_param))
        if hasattr(result, "__aiter__"):
            # We don't enable stream in config, but keep a clear error if it happens.
            raise RuntimeError("Streaming responses are not supported in this CLI mode")

        return Answer(answer_text=str(result), evidence=[])

    def close(self) -> None:
        if self._rag is None:
            self._runner.stop()
            return
        try:
            self._runner.run(self._rag.finalize_storages())
        except Exception:
            logger.debug("Failed to finalize LightRAG storages", exc_info=True)
        finally:
            self._runner.stop()
