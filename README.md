# RAG Story Investigator

A console-based Python application that answers questions about a fictional story using three different RAG (Retrieval-Augmented Generation) approaches with dependency injection and configuration-based architecture.

## Overview

This project implements a modular RAG system for investigating story messages (SMS/chat format) using multiple retrieval strategies. The system supports three different RAG engines with unified evidence formatting:

1. **Naive RAG** - Simple embedding-based semantic search with chunking ✅ **Implemented**
2. **LightRAG** - Advanced lightweight RAG system with end-to-end answering ✅ **Implemented (OpenAI gpt-4o-mini)**
3. **GraphRAG** - Graph-based retrieval using Neo4j vector index ✅ **Implemented (neo4j-graphrag-python)**

## Features

- ✅ **Configuration-based Dependency Injection**: All components configured via `config.yaml`
- ✅ **Three RAG Engines**: Naive (implemented), LightRAG and GraphRAG (planned)
- ✅ **OpenAI Integration**:
  - `gpt-4o-mini` for LightRAG answering
  - OpenAI Chat Completions for the default LLM client
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

### 4. Configure Environment Variables
Create a `.env` file in the project root (copy from `.env.example`):
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# OpenAI API Key (required for LightRAG and Naive RAG LLM)
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration (required for GraphRAG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

**Get API Keys:**
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 5. Set Up Neo4j (Required for GraphRAG)
GraphRAG requires a local Neo4j instance. Choose one option:

**Option A: Docker (Recommended)**
```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

**Option B: Neo4j Desktop**
1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new database
3. Set password to match your `.env` file (or update `.env` to match)
4. Start the database

**Verify Neo4j is running:**
- Open http://localhost:7474 in your browser
- Login with username: `neo4j`, password: `password` (or your custom password)

### 6. Configure System (Optional)
Edit `config.yaml` to customize:
- RAG system settings (chunk size, similarity threshold, etc.)
- LLM parameters (model, temperature, max tokens)
- Embedding configuration (model, dimensionality)
- Neo4j connectioneo4j GraphRAG (requires Neo4j running
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
  2. lightrag  - LightRAG (OpenAI gpt-4o-mini)
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
# Use custom config fileRAG storage (auto-generated)
│   ├── lightrag/              # LightRAG cache (38MB: graphs, embeddings, LLM responses)
│   └── graphrag/              # GraphRAG working directory
python -m src.main --config custom_config.yaml

# Use custom story file
python -m src.main --story data/my_story.xml
```

## Project Structure

```
Rag_Story_Investigator/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dep
│       │   ├── chunker.py     # Message chunking
│       │   ├── chunk_indexer.py      # Embedding & caching
│       │   ├── similarity.py         # Semantic search
│       │   └── naive_rag.py          # Main RAG orchestrator
│       │
│       ├── lightrag/          # ✅ LightRAG
│       │   ├── lightrag_engine.py    # Engine wrapper
│       │   ├── lightrag_rag.py       # RagEngine implementation
│       │   └── llm_service.py        # LLM integration
│       │
│       └── graphrag/          # ✅ GraphRAG
│           └── graphrag_engine.py    # Neo4j vector retrieval
│   ├── core/                  # Core domain logic
│   │   ├── embedding_service.py  # Gemini embeddings
│   │   ├── models.py          # Data models (Message, Chunk, SearchResult, etc.)
│   │   ├── story_loader.py    # XML parsing
│   │   ├── prompt_builder.py  # Prompt construction for LLM
│   │   └── llm_client.py      # Google Gemini client
│   │
│   └── rag/                   # RAG implementations
│       ├── naive/             # ✅ Naive RAG (implemented)
│       │   ├── chunker.py     # Message chunking
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

## Tech Stack10+
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: 
  - Sentence Transformers (all-mpnet-base-v2, 768-dim) for Naive RAG and GraphRAG
  - OpenAI embeddings for LightRAG
- **Graph Database**: Neo4j 5.x with vector index (GraphRAG)
- **RAG Libraries**:
  - `lightrag-hku` - LightRAG implementation
  - `neo4j-graphrag-python` - Neo4j GraphRAG retrievers
- **Configuration**: PyYAML, python-dotenv
- **Testing**: pytest, pytest-cov** (planned):
  -RAG System Details

### Naive RAG
- **Retrieval**: Cosine similarity search on local embeddings
- **Embedding**: sentence-transformers (all-mpnet-base-v2)
- **Storage**: JSON cache files
- **Best for**: Simple, fast queries with local control

### LightRAG
- **Retrieval**: End-to-end RAG with built-in knowledge graph
- **LLM**: OpenAI gpt-4o-mini
- **Embedding**: OpenAI embeddings
- **Storage**: 38MB cache (graphs, embeddings, LLM responses)
- **Modes**: naive, local, global, hybrid
- **Best for**: Complex queries requiring graph relationships

### GraphRAG
- **Retrieval**: with OpenAI integration
- [x] Naive RAG implementation
  - [x] Message chunking
  - [x] Embedding service with local models
  - [x] Chunk indexing and caching
  - [x] Semantic similarity search
- [x] LightRAG implementation
  - [x] OpenAI gpt-4o-mini integration
  - [x] Knowledge graph construction
  - [x] Custom user prompt formatting
  - [x] Cache management (38MB cached data)
- [x] GraphRAG implementation
  - [x] Neo4j vector index creation
  - [x] Message ingestion as graph nodes
  - [x] VectorRetriever integration
  - [x] Environment variable configuration
- [x] Dependency injection system
- [x] Interactive console interface
- [x] Unified evidence formatting across all engines
- [x] Secure credential management (.env)

### Next Steps 🚧
- [ ] Performance benchmarking comparison
- [ ] Graph traversal retrieval (beyond vector-only)
- [ ] Hybrid retrieval strategies
- [ ] Answer quality metrics
- [ ] Advanced Neo4j relationship modelingors
- Configurable delay between batches
- Progress logging for long operations

## Roadmap

### Completed ✅
- [Troubleshooting

### GraphRAG: "Could not connect to Neo4j"
1. Verify Neo4j is running: `docker ps` (should show neo4j container)
2. Check http://localhost:7474 is accessible
3. Verify credentials in `.env` match your Neo4j setup
4. Ensure ports 7474 and 7687 are not blocked

### LightRAG: Cache files too large
- LightRAG cache grows to ~38MB (normal)
- **Do NOT delete cache files** - prevents expensive re-indexing
- Files in `cache/lightrag/`:
  - `graph_chunk_entity_relation.graphml` - Knowledge graph
  - `kv_store_*.json` - Embeddings and entities
  - `vdb_*.json` - Vector databases

### OpenAI API Errors
- Verify `OPENAI_API_KEY` is set in `.env`
- Check API quota at [platform.openai.com](https://platform.openai.com/usage)

## Acknowledgments

- [OpenAI](https://openai.com) for GPT-4o-mini and embeddings
- [LightRAG-HKU](https://github.com/HKUDS/LightRAG) for lightweight RAG framework
- [neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python) for graph-based retrieval
- [Sentence Transformers](https://www.sbert.net/) for local embedding models
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
- neo4j-graphrag-python for graph-based retrieval
