---
layout: default
title: Development Environment
parent: "07 - Hands-on Engineering"
nav_order: 1
---

# Development Environment

## TL;DR

A proper AI development environment includes: Python with virtual environments, essential libraries, API key management, and local testing tools. This guide gets you set up quickly.

## Quick Setup

### 1. Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or use conda
conda create -n ai-dev python=3.11
conda activate ai-dev
```

### 2. Core Dependencies

```bash
# requirements.txt
openai>=1.0.0
anthropic>=0.18.0
langchain>=0.1.0
langchain-openai>=0.0.5
chromadb>=0.4.0
python-dotenv>=1.0.0
pydantic>=2.0.0
httpx>=0.25.0
rich>=13.0.0  # Better console output
```

```bash
pip install -r requirements.txt
```

### 3. API Key Management

```bash
# .env file (add to .gitignore!)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

```python
# Load in Python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

## Recommended Project Structure

```
my-ai-project/
├── .env                    # API keys (git-ignored)
├── .env.example            # Template without secrets
├── .gitignore
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py       # LLM client wrapper
│   │   └── prompts.py      # Prompt templates
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py   # Embedding logic
│   │   ├── retriever.py    # Vector search
│   │   └── indexer.py      # Document indexing
│   └── agents/
│       ├── __init__.py
│       └── tools.py        # Tool definitions
│
├── data/
│   └── documents/          # Source documents for RAG
│
├── tests/
│   ├── __init__.py
│   ├── test_llm.py
│   └── test_rag.py
│
├── evals/
│   ├── datasets/           # Evaluation datasets
│   └── run_eval.py         # Evaluation scripts
│
└── notebooks/
    └── experiments.ipynb   # Quick experiments
```

## Essential Libraries by Use Case

### Basic LLM Calls

```python
# OpenAI
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

```python
# Anthropic
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### Structured Output

```python
# Using Instructor (recommended)
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class User(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-4o",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}]
)
print(user)  # User(name='John', age=30)
```

### RAG with LangChain

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split documents
loader = TextLoader("document.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("What is the main topic?")
```

## Local Development Tips

### Caching LLM Calls

```python
# Save money during development
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def cached_completion(prompt: str, model: str = "gpt-4o"):
    cache_key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    
    cache_file.write_text(json.dumps(result))
    return result
```

### Rate Limit Handling

```python
from tenacity import retry, wait_exponential, retry_if_exception_type
from openai import RateLimitError

@retry(
    wait=wait_exponential(min=1, max=60),
    retry=retry_if_exception_type(RateLimitError)
)
def call_llm(prompt: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
```

### Cost Tracking

```python
import tiktoken

def estimate_cost(prompt: str, model: str = "gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    tokens = len(enc.encode(prompt))
    
    # Approximate costs (check current pricing)
    cost_per_1k = {"gpt-4o": 0.005, "gpt-3.5-turbo": 0.0005}
    
    return tokens * cost_per_1k.get(model, 0.01) / 1000
```

## IDE Setup

### VS Code Extensions

- Python
- Pylance
- Jupyter
- GitLens
- REST Client (for API testing)

### Cursor-specific

```json
// .cursorrules or settings
{
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```

## Testing LLM Code

```python
# tests/test_llm.py
import pytest
from unittest.mock import patch

def test_summarization():
    """Test with mocked LLM response"""
    with patch('src.llm.client.call_llm') as mock_llm:
        mock_llm.return_value = "This is a summary"
        result = summarize("Long text...")
        assert len(result) < len("Long text...")

def test_real_llm_integration():
    """Integration test with real API (use sparingly)"""
    result = call_llm("Say 'test' and nothing else")
    assert "test" in result.lower()
```

## Key Takeaways

- Use virtual environments and `.env` for clean setup
- Structure projects for maintainability from the start
- Cache LLM calls during development to save money
- Use libraries like Instructor for type-safe outputs
- Test with mocks for unit tests, real APIs for integration

## References

- [Python Best Practices](https://docs.python-guide.org/)
- [Instructor Documentation](https://python.useinstructor.com/)
- [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart)
