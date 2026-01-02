# RAG Story Investigator

A console-based Python application that answers questions about a fictional story using three different RAG (Retrieval-Augmented Generation) approaches.

## Features

- **Three RAG engines**: naive (embeddings), LightRAG, and nano-graphrag
- **LLM-powered answers**: Uses Google Gemini for answer generation
- **Evidence citations**: Displays relevant story messages as supporting evidence
- Interactive console interface

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Rag_Story_Investigator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - `GEMINI_API_KEY`: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
     - `OPENAI_API_KEY`: Get from [OpenAI Platform](https://platform.openai.com/api-keys) (required only for naive RAG)

## Usage

Run the application:
```bash
python src/main.py
```

Follow the prompts:
1. Select a RAG engine: `naive`, `lightrag`, or `graphrag`
2. Ask questions about the story
3. Type `exit` to quit

### Example

```
AI Investigator 1.0. Ask me any question about the story
Select RAG engine (naive/lightrag/graphrag): naive
Ask a question: Who requested to bring the USB?

Marcus requested the USB.

Evidence:
<sender ref="marcus"/>
<receiver ref="alex"/>
<body>Bring that USB you borrowed.</body>
```

## Project Structure

```
src/
  main.py                      # Entry point and CLI
  core/
    models.py                  # Domain models and interfaces
    story_loader.py            # XML story parsing
    prompt_builder.py          # LLM prompt construction
    llm_client.py              # Gemini client
    answer_formatter.py        # Evidence formatting
  rag/
    naive/                     # Embedding-based RAG
    lightrag/                  # LightRAG implementation
    graphrag/                  # Nano-GraphRAG implementation
data/
  story.xml                    # Story messages
tests/                         # Unit tests
```

## Testing

Run tests:
```bash
pytest
```

With coverage:
```bash
pytest --cov=src --cov-report=html
```

## Tech Stack

- **Language**: Python 3.8+
- **LLM**: Google Gemini
- **Embeddings**: OpenAI (naive RAG only)
- **RAG Libraries**: LightRAG, nano-graphrag
- **Testing**: pytest
