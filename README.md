# RAG Story Investigator

A console app that answers questions about a fictional chat-log story, using three
interchangeable RAG (Retrieval-Augmented Generation) strategies behind one interface.

## Overview

The app loads a story transcript (`data/story.xml`) and lets you ask questions in an
interactive CLI. Pick one of three retrieval engines at startup:

1. **Naive RAG**: chunk the messages, embed them locally, retrieve by cosine similarity
2. **LightRAG**: LightRAG-HKU entity/relationship indexing and end-to-end answering
3. **GraphRAG**: message chunks stored as Neo4j nodes, retrieved through a Neo4j vector index

All three return a short answer followed by an **Evidence** section.

Embeddings are generated locally with `sentence-transformers`, so no embedding API is
called. OpenAI is used to write the final answer (and by LightRAG during indexing), so
`OPENAI_API_KEY` is required for every engine.

## Requirements

- Python 3.11
- An OpenAI API key
- Docker, if you want to run Neo4j for the GraphRAG engine

## Quick Start (Docker)

This runs both the app and Neo4j:

```bash
cp .env.example .env    # then put your OPENAI_API_KEY in it
docker compose up --build
```

The `.env` file is not optional here: `docker-compose.yml` declares it under `env_file`,
so the stack will not start without it. Neo4j Browser is served at http://localhost:7474.

To change the Neo4j password, set both `NEO4J_PASSWORD` and `NEO4J_AUTH` in `.env`.

The first LightRAG or GraphRAG run indexes the whole story, which takes a few minutes
and costs OpenAI tokens. Results are cached (see [Caching](#caching)).

## Local Setup (venv)

### 1) Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

The first run also downloads the `all-mpnet-base-v2` embedding model (about 420 MB).

### 3) Create `.env`

```bash
cp .env.example .env
```

Fill in `OPENAI_API_KEY`. The Neo4j defaults in the template match the local Neo4j
container below, so you only need to change them if you use a different password.

### 4) Run Neo4j (only needed for GraphRAG)

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
```

### 5) Run the application

```bash
python -m src.main
```

Optionally point it at a different transcript with `python -m src.main --story path/to/story.xml`.

## Evidence format

Evidence items are displayed as:

```
(at timestamp) sender to receiver: message content
```

Example:

```
(at 2025-08-29T19:15:50+10:00) zoe to six: I just found Liam's phone in Marcus's jacket pocket.
```

## Configuration

- Application settings: `config.yaml` (chunk size, embedding model, retrieval thresholds,
  LLM model and fallback, which engine starts by default)
- Secrets: `.env`, loaded with `python-dotenv`

Neo4j settings appear in both `config.yaml` and `.env`; the environment variables win:
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Caching

`cache/` is generated at runtime and is not tracked in git.

- `cache/story_naive.pkl` holds the naive engine's chunk embeddings
- `cache/lightrag/` holds LightRAG state (entity graph, KV stores, LLM response cache)
- `cache/graphrag/` holds a marker file so GraphRAG does not re-ingest into Neo4j

Delete a subdirectory to force that engine to re-index.

## Project Structure

```
Rag_Story_Investigator/
├── config.yaml            # all tunable settings
├── requirements.txt
├── Dockerfile
├── docker-compose.yml     # app + neo4j
├── .env.example
├── data/
│   └── story.xml          # the transcript being questioned
├── src/
│   ├── main.py            # CLI entry point
│   ├── config_loader.py   # config parsing + engine construction
│   ├── core/              # story loader, chunk models, embeddings, LLM client, prompts
│   └── rag/
│       ├── naive/         # chunker, indexer, similarity search
│       ├── lightrag/      # LightRAG-HKU wiring
│       └── graphrag/      # Neo4j vector retrieval
└── tests/                 # pytest suite for the core and naive-RAG components
```

`cache/` is created on first run.

## Testing

```bash
pytest
```

The suite covers the pure components: story loading, chunking, the chunk indexer,
similarity search and prompt building. The LightRAG and GraphRAG engines are thin
wrappers over external services and are not covered.

## Troubleshooting

### GraphRAG cannot connect to Neo4j

- Verify Neo4j is running: `docker ps`
- Neo4j Browser: http://localhost:7474
- Check `NEO4J_URI`:
  - Local run: `bolt://localhost:7687`
  - Docker Compose: `bolt://neo4j:7687`

### Neo4j GraphRAG package name

The pip package is `neo4j-graphrag`; the Python module is `neo4j_graphrag`.

### OpenAI API errors

- Verify `OPENAI_API_KEY` is set in `.env`
- Check your quota at [platform.openai.com](https://platform.openai.com/usage)

## Acknowledgments

- [OpenAI](https://openai.com) for the GPT models that generate the answers
- [LightRAG-HKU](https://github.com/HKUDS/LightRAG) for the lightweight RAG framework
- [neo4j-graphrag](https://github.com/neo4j/neo4j-graphrag-python) for graph-based retrieval
- [Sentence Transformers](https://www.sbert.net/) for the local embedding models

## License

Built as a personal learning project. No license has been attached, so all rights
are reserved.
