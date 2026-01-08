# RAG Story Investigator

A console-based Python application that answers questions about a fictional story using multiple RAG (Retrieval-Augmented Generation) strategies.

## Overview

The application loads a story transcript (XML) and lets you ask questions in an interactive CLI. You can choose between three retrieval engines:

1. **Naive RAG**: local embedding similarity search
2. **LightRAG**: LightRAG-HKU end-to-end answering
3. **GraphRAG**: Neo4j vector index retrieval

All engines aim to return a concise answer followed by an **Evidence** section.

## Quick Start (Docker)

This runs both the app and Neo4j using Docker Compose:

```bash
docker compose up --build
```

Neo4j Browser will be available at http://localhost:7474.

### Environment variables (Docker)

Docker Compose automatically reads a `.env` file in the project root if it exists.
At minimum, set:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

If you want to change the default Neo4j password, set:

```bash
NEO4J_PASSWORD=your_password
NEO4J_AUTH=neo4j/your_password
```

## Local Setup (venv)

### 1) Create a virtual environment

Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Create `.env`

Copy the template and fill in values:

```bash
cp .env.example .env
```

For GraphRAG (local Neo4j), the defaults are:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

### 4) Run Neo4j (required for GraphRAG)

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
```

### 5) Run the application

```bash
python -m src.main
```

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

- Application settings: `config.yaml`
- Secrets: `.env` (loaded via `python-dotenv`)

Neo4j settings can be set in both `config.yaml` and `.env`; environment variables take precedence:
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Caching

- `cache/lightrag/` stores LightRAG state (graphs, embeddings, LLM response cache). Deleting it forces re-indexing.
- `cache/graphrag/` stores a local marker file used by GraphRAG to avoid re-ingesting.

## Project Structure

```
Rag_Story_Investigator/
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── data/
├── cache/
├── src/
└── tests/
```

## Testing

```bash
pytest
```

## Troubleshooting

### GraphRAG cannot connect to Neo4j

- Verify Neo4j is running (local): `docker ps`
- Neo4j Browser: http://localhost:7474
- Check `NEO4J_URI`:
  - Local run: `bolt://localhost:7687`
  - Docker Compose: `bolt://neo4j:7687`

### Neo4j GraphRAG package name

- The pip package is `neo4j-graphrag` and the Python module is `neo4j_graphrag`.

### OpenAI API Errors
- Verify `OPENAI_API_KEY` is set in `.env`
- Check API quota at [platform.openai.com](https://platform.openai.com/usage)

## Acknowledgments

- [OpenAI](https://openai.com) for GPT-4o-mini and embeddings
- [LightRAG-HKU](https://github.com/HKUDS/LightRAG) for lightweight RAG framework
- [neo4j-graphrag](https://pypi.org/project/neo4j-graphrag/) (repo: https://github.com/neo4j/neo4j-graphrag-python) for graph-based retrieval
- [Sentence Transformers](https://www.sbert.net/) for local embedding models

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
- neo4j-graphrag for graph-based retrieval
