
# RAG Fundamentals

## TL;DR

RAG (Retrieval-Augmented Generation) enhances LLM responses by retrieving relevant context from external documents and injecting it into the prompt. It's how you give LLMs access to private, recent, or specialized knowledge.

## The Problem RAG Solves

LLMs have limitations:

| Limitation | Example |
|------------|---------|
| Knowledge cutoff | "What happened in the news today?" |
| No private data | "Summarize our Q3 report" |
| Hallucination | Makes up plausible-sounding facts |
| Context limits | Can't read your entire codebase |

RAG addresses all of these by providing relevant context at query time.

## The RAG Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query: "How do I configure authentication?"              │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────┐                  │
│   │           1. EMBED QUERY                 │                  │
│   │   "How do I configure..." → [0.2, 0.5]  │                  │
│   └─────────────────────────────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────┐                  │
│   │           2. RETRIEVE                    │                  │
│   │   Vector search → Top-k documents       │                  │
│   │   • auth-config.md (0.92 similarity)    │                  │
│   │   • security-guide.md (0.87 similarity) │                  │
│   │   • api-reference.md (0.71 similarity)  │                  │
│   └─────────────────────────────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────┐                  │
│   │           3. AUGMENT PROMPT              │                  │
│   │   "Given these docs: {retrieved_docs}   │                  │
│   │    Answer: {user_query}"                 │                  │
│   └─────────────────────────────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────┐                  │
│   │           4. GENERATE                    │                  │
│   │   LLM produces answer with citations    │                  │
│   └─────────────────────────────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│   Response: "To configure auth, set these env vars..."         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Indexing Pipeline (Offline)

Before you can query, you need to index your documents:

```
Documents → Load → Split → Embed → Store
```

### 1. Load Documents

```python
# Various sources
docs = []
docs += load_pdfs("./documents/")
docs += load_markdown("./docs/")
docs += scrape_website("https://docs.example.com")
docs += load_notion_pages(notion_client)
```

### 2. Split Into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(docs)
```

### 3. Embed

```python
from openai import OpenAI

client = OpenAI()
embeddings = [
    client.embeddings.create(input=chunk.text, model="text-embedding-3-small")
    for chunk in chunks
]
```

### 4. Store

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")
collection.add(
    documents=[c.text for c in chunks],
    embeddings=embeddings,
    ids=[c.id for c in chunks]
)
```

## Query Pipeline (Online)

```python
def query_rag(user_question: str) -> str:
    # 1. Embed the query
    query_embedding = embed(user_question)
    
    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    # 3. Build augmented prompt
    context = "\n\n".join(results['documents'][0])
    prompt = f"""Answer based on the following context:

{context}

Question: {user_question}
Answer:"""
    
    # 4. Generate response
    response = llm.generate(prompt)
    return response
```

## Key Components

### Embedding Model

| Model | Provider | Dimensions | Notes |
|-------|----------|------------|-------|
| text-embedding-3-small | OpenAI | 1536 | Good balance |
| text-embedding-3-large | OpenAI | 3072 | Higher quality |
| voyage-2 | Voyage | 1024 | Strong for code |
| bge-large | BAAI | 1024 | Open source |

### Vector Store

| Store | Type | Best For |
|-------|------|----------|
| Chroma | Embedded | Prototyping |
| Pinecone | Managed | Production scale |
| Weaviate | Both | Hybrid search |
| pgvector | Extension | Postgres users |

### Retrieval Strategy

| Strategy | Description |
|----------|-------------|
| Dense | Vector similarity only |
| Sparse | Keyword matching (BM25) |
| Hybrid | Dense + Sparse combined |
| Rerank | Two-stage with cross-encoder |

## Common Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Wrong chunks retrieved | Irrelevant answers | Better chunking, hybrid search |
| Too few chunks | Missing information | Increase k, better embeddings |
| Too many chunks | Context overflow | Reranking, summarization |
| Chunk boundaries | Cut-off information | Overlap, parent-child chunks |

## Key Takeaways

- RAG = Retrieve + Augment + Generate
- Quality depends heavily on chunking and retrieval
- Start simple (dense retrieval), optimize as needed
- Always measure retrieval quality separately from generation

## References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Pinecone — RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
