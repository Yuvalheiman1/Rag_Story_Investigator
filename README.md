# RAG Story Investigator

A console-based Python application that answers questions about a fictional story using three different RAG (Retrieval-Augmented Generation) approaches with dependency injection and configuration-based architecture.

## Overview

This project implements a modular RAG system for investigating story messages (SMS/chat format) using multiple retrieval strategies. The system uses Google Gemini for both embeddings and answer generation, with support for three RAG engines:

1. **Naive RAG** - Simple embedding-based semantic search with chunking ✅ **Implemented**
2. **LightRAG** - Advanced lightweight RAG system 🚧 **Coming Soon**
3. **GraphRAG** - Graph-based retrieval using nano-graphrag 🚧 **Coming Soon**

## Features

- ✅ **Configuration-based Dependency Injection**: All components configured via `config.yaml`
- ✅ **Three RAG Engines**: Naive (implemented), LightRAG and GraphRAG (planned)
- ✅ **Google Gemini Integration**: 
  - `gemini-embedding-001` for embeddings (with rate limiting & retry)
  - `gemini-2.0-flash-exp` for answer generation
- ✅ **Timestamp Support**: Messages include timestamps, displayed in evidence
- ✅ **Smart Caching**: Embeddings are cached to avoid redundant API calls
- ✅ **Rate Limiting**: Automatic rate limiting and retry for API calls
- ✅ **Batch Processing**: Handles large datasets with automatic batching (100 items/batch)
- ✅ **Interactive Console**: Question/answer loop with RAG system switching
- ✅ **TDD Approach**: Comprehensive test coverage for core components

## Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Rag_Story_Investigator
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Configure System (Optional)
Edit `config.yaml` to customize:
- RAG system settings (chunk size, similarity threshold, etc.)
- LLM parameters (model, temperature, max tokens)
- Embedding configuration (model, dimensionality)
- Logging level

## Usage

### Run the Application
```bash
python -m src.main
```

### Interactive Mode
```
AI Investigator 1.0
Ask me any question about the story
======================================================================

Select RAG system:
  1. naive     - Simple embedding-based retrieval
  2. lightrag  - LightRAG (not implemented yet)
  3. graphrag  - Nano-GraphRAG (not implemented yet)

Enter choice [1-3] (default: naive): 1

======================================================================
System ready. Enter your questions (or 'quit' to exit)
======================================================================

Question: where is the usb?

======================================================================
ANSWER
======================================================================
The USB is mentioned in several messages, but no specific location is given.

Evidence:
1. [0.842] (at 2025-08-29T17:45:12+10:00) <sender ref="marcus"/><receiver ref="alex"/><body>Don't forget the USB.</body>
2. [0.791] (at 2025-08-29T18:02:33+10:00) <sender ref="alex"/><receiver ref="marcus"/><body>I have the USB with me.</body>

Question:
```

### Command Line Options
```bash
# Use custom config file
python -m src.main --config custom_config.yaml

# Use custom story file
python -m src.main --story data/my_story.xml
```

## Project Structure

```
Rag_Story_Investigator/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not in git)
├── README.md
│
├── data/
│   └── story.xml              # Story messages in XML format
│
├── cache/                     # Embedding cache (auto-generated)
│
├── src/
│   ├── main.py                # Entry point & interactive loop
│   ├── config_loader.py       # DI factory for all components
│   │
│   ├── core/                  # Core domain logic
│   │   ├── models.py          # Data models (Message, Chunk, SearchResult, etc.)
│   │   ├── story_loader.py    # XML parsing
│   │   ├── prompt_builder.py  # Prompt construction for LLM
│   │   └── llm_client.py      # Google Gemini client
│   │
│   └── rag/                   # RAG implementations
│       ├── naive/             # ✅ Naive RAG (implemented)
│       │   ├── chunker.py     # Message chunking
│       │   ├── embedding_service.py  # Gemini embeddings
│       │   ├── chunk_indexer.py      # Embedding & caching
│       │   ├── similarity.py         # Semantic search
│       │   └── naive_rag.py          # Main RAG orchestrator
│       │
│       ├── lightrag/          # 🚧 LightRAG (planned)
│       │   ├── lightrag_config.py
│       │   └── lightrag_rag.py
│       │
│       └── graphrag/          # 🚧 GraphRAG (planned)
│           ├── graphrag_config.py
│           └── nano_graphrag_rag.py
│
└── tests/                     # Unit tests
    ├── test_story_loader.py
    ├── test_prompt_builder.py
    └── test_naive_rag.py
```

## Configuration

config.yaml

## Architecture

### Dependency Injection Pattern
The system uses configuration-based dependency injection:

1. **ConfigLoader** reads `config.yaml` and creates all components
2. **StoryInvestigator** receives pre-configured components
3. **RAG engines** receive all dependencies (no internal instantiation)

This approach provides:
- ✅ Modularity - Easy to swap implementations
- ✅ Testability - Simple to mock components
- ✅ Flexibility - Change behavior via config, not code
- ✅ Scalability - Easy to add new RAG systems

### RAG Engine Interface
All RAG engines implement a common interface:
```python
class RagEngine(ABC):
    @abstractmethod
    def retrieve(self, question: str, threshold: float, max_results: int) -> List[SearchResult]:
        """Retrieve relevant chunks for a question."""
        pass
```

### Data Flow
```
User Question
    ↓
RAG Engine (retrieve relevant chunks)
    ↓
PromptBuilder (assemble prompt with context)
    ↓
LLMClient (generate answer)
    ↓
Display to User (with evidence & timestamps)
```

## Testing

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Test Individual Modules
```bash
pytest tests/test_story_loader.py
pytest tests/test_prompt_builder.py
pytest tests/test_naive_rag.py
```

### Manual Testing
Each module includes `if __name__ == "__main__"` blocks for quick manual testing:
```bash
python -m src.core.story_loader
python -m src.core.prompt_builder
python -m src.rag.naive.embedding_service
python -m src.rag.naive.similarity
```

## Tech Stack

- **Language**: Python 3.8+
- **LLM & Embeddings**: Google Gemini API
  - `gemini-embedding-001` for embeddings
  - `gemini-2.0-flash-exp` for text generation
- **Configuration**: PyYAML
- **Environment**: python-dotenv
- **Testing**: pytest, pytest-cov
- **RAG Libraries** (planned):
  - lightrag-hku
  - nano-graphrag

## API Rate Limits

### Gemini Free Tier
- Embedding: 100 requests/minute
- Generation: 15 requests/minute

The system automatically handles rate limiting with:
- Automatic retry on 429 errors
- Configurable delay between batches
- Progress logging for long operations

## Roadmap

### Completed ✅
- [x] Project structure and configuration
- [x] Story loader with timestamp support
- [x] Prompt builder with system instructions
- [x] LLM client (Gemini)
- [x] Naive RAG implementation
  - [x] Message chunking
  - [x] Embedding service with rate limiting
  - [x] Chunk indexing and caching
  - [x] Semantic similarity search
- [x] Dependency injection system
- [x] Interactive console interface
- [x] Evidence formatting in LLM output

### Next Steps 🚧
- [ ] LightRAG implementation
- [ ] GraphRAG (nano-graphrag) implementation
- [ ] Performance benchmarking
- [ ] Advanced caching strategies
- [ ] Multi-hop reasoning
- [ ] Answer quality metrics

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Google Gemini for LLM and embedding APIs
- LightRAG-HKU for lightweight RAG framework
- nano-graphrag for graph-based retrieval
