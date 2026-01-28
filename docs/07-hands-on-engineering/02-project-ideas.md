
# Project Ideas

## TL;DR

Learning by building is essential. These projects are ordered by complexity—start simple and build up. Each project teaches specific concepts while being genuinely useful.

## Difficulty Levels

```
🟢 Beginner     (1-2 days)  - Basic API calls, simple chains
🟡 Intermediate (1-2 weeks) - RAG, tool calling, structured output
🔴 Advanced     (2-4 weeks) - Agents, evaluation, production systems
```

## Project 1: Smart CLI Assistant 🟢

### What You Build
A command-line tool that answers questions using an LLM.

### Concepts Learned
- LLM API basics
- Conversation history
- System prompts
- Streaming responses

### Implementation

```python
from openai import OpenAI

client = OpenAI()
history = []

SYSTEM = "You are a helpful CLI assistant. Be concise."

def chat(user_input: str) -> str:
    history.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM}] + history,
        stream=True
    )
    
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response += chunk.choices[0].delta.content
    
    history.append({"role": "assistant", "content": full_response})
    return full_response

if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("\nAssistant: ", end="")
        chat(user_input)
```

### Extensions
- Add command history persistence
- Implement slash commands (/clear, /export)
- Add code syntax highlighting

---

## Project 2: Document Q&A (RAG) 🟡

### What You Build
Upload documents, ask questions, get answers with citations.

### Concepts Learned
- Document loading and chunking
- Embeddings and vector search
- RAG pipeline
- Source attribution

### Architecture

```
Documents → Chunk → Embed → Store (Chroma)
                                    ↓
User Question → Embed → Search → Top-K Chunks
                                    ↓
                          Augmented Prompt → LLM → Answer + Citations
```

### Implementation

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

class DocumentQA:
    def __init__(self, docs_path: str):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.vectorstore = None
        self._index_documents(docs_path)
    
    def _index_documents(self, docs_path: str):
        documents = []
        for file in Path(docs_path).glob("*.txt"):
            content = file.read_text()
            documents.append({"content": content, "source": file.name})
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        chunks = []
        for doc in documents:
            splits = splitter.split_text(doc["content"])
            for i, split in enumerate(splits):
                chunks.append({
                    "content": split,
                    "metadata": {"source": doc["source"], "chunk": i}
                })
        
        self.vectorstore = Chroma.from_texts(
            texts=[c["content"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embedding=self.embeddings
        )
    
    def query(self, question: str, k: int = 3) -> dict:
        # Retrieve relevant chunks
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # Build context
        context = "\n\n".join([
            f"[Source: {d.metadata['source']}]\n{d.page_content}"
            for d in docs
        ])
        
        # Generate answer
        prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. 
Include source citations in your answer.

Context:
{context}

Question: {question}

Answer:""")
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        
        return {
            "answer": response.content,
            "sources": [d.metadata["source"] for d in docs]
        }
```

### Extensions
- Support PDF, markdown, web pages
- Add hybrid search (semantic + keyword)
- Implement reranking
- Build a web interface

---

## Project 3: Code Review Agent 🟡

### What You Build
An agent that reviews code, runs linters, and suggests improvements.

### Concepts Learned
- Tool calling
- Agent loops
- Code analysis
- Structured output

### Implementation

```python
import subprocess
from openai import OpenAI
from pydantic import BaseModel
import json

client = OpenAI()

class ReviewResult(BaseModel):
    issues: list[str]
    suggestions: list[str]
    overall_quality: str

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_linter",
            "description": "Run Python linter (ruff) on code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to lint"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_types",
            "description": "Run type checker (mypy) on code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to check"}
                },
                "required": ["code"]
            }
        }
    }
]

def run_linter(code: str) -> str:
    # Write to temp file and run ruff
    with open("/tmp/code.py", "w") as f:
        f.write(code)
    result = subprocess.run(
        ["ruff", "check", "/tmp/code.py"],
        capture_output=True, text=True
    )
    return result.stdout or "No issues found"

def check_types(code: str) -> str:
    with open("/tmp/code.py", "w") as f:
        f.write(code)
    result = subprocess.run(
        ["mypy", "/tmp/code.py"],
        capture_output=True, text=True
    )
    return result.stdout or "No type errors"

def execute_tool(name: str, args: dict) -> str:
    if name == "run_linter":
        return run_linter(args["code"])
    elif name == "check_types":
        return check_types(args["code"])
    return "Unknown tool"

def review_code(code: str) -> ReviewResult:
    messages = [
        {"role": "system", "content": """You are a code reviewer. 
Use the available tools to analyze the code, then provide a structured review."""},
        {"role": "user", "content": f"Review this code:\n\n```python\n{code}\n```"}
    ]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            # Parse final response into structured format
            # (In production, use instructor for this)
            break
        
        for tool_call in msg.tool_calls:
            result = execute_tool(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return msg.content
```

### Extensions
- Add security vulnerability scanning
- Support multiple languages
- Generate automated fixes
- Integrate with GitHub PRs

---

## Project 4: Personal Knowledge Base 🔴

### What You Build
A system that ingests your notes, bookmarks, and documents, then answers questions about your knowledge.

### Concepts Learned
- Multi-source ingestion
- Incremental indexing
- Query expansion
- Personalization

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Sources:                                                   │
│   ├── Notion (API)                                          │
│   ├── Bookmarks (browser export)                            │
│   ├── Local markdown files                                  │
│   ├── Saved articles (Pocket/Instapaper)                    │
│   └── GitHub stars                                          │
│                                                              │
│   Ingestion Pipeline:                                        │
│   └── Scheduled sync → Dedupe → Chunk → Embed → Store       │
│                                                              │
│   Query Pipeline:                                            │
│   └── Question → Expand → Retrieve → Rerank → Generate     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Project 5: Evaluation Pipeline 🔴

### What You Build
A system to evaluate and compare LLM performance across prompts and models.

### Concepts Learned
- Eval dataset creation
- Automated metrics
- LLM-as-judge
- A/B testing

### Core Components

```python
from dataclasses import dataclass
from typing import Callable
import json

@dataclass
class EvalExample:
    id: str
    input: str
    expected: str | None
    metadata: dict

@dataclass
class EvalResult:
    example_id: str
    model: str
    output: str
    scores: dict[str, float]
    latency_ms: float

class EvalPipeline:
    def __init__(self, examples: list[EvalExample]):
        self.examples = examples
        self.metrics: list[Callable] = []
    
    def add_metric(self, name: str, fn: Callable):
        self.metrics.append((name, fn))
    
    def run(self, model_fn: Callable) -> list[EvalResult]:
        results = []
        for ex in self.examples:
            start = time.time()
            output = model_fn(ex.input)
            latency = (time.time() - start) * 1000
            
            scores = {}
            for name, fn in self.metrics:
                scores[name] = fn(output, ex.expected, ex.metadata)
            
            results.append(EvalResult(
                example_id=ex.id,
                model=model_fn.__name__,
                output=output,
                scores=scores,
                latency_ms=latency
            ))
        return results
    
    def compare(self, results_a: list, results_b: list) -> dict:
        """Compare two runs"""
        comparison = {}
        for metric in self.metrics:
            name = metric[0]
            mean_a = sum(r.scores[name] for r in results_a) / len(results_a)
            mean_b = sum(r.scores[name] for r in results_b) / len(results_b)
            comparison[name] = {"model_a": mean_a, "model_b": mean_b}
        return comparison
```

---

## Progression Roadmap

```
Month 1:
├── Week 1-2: Project 1 (CLI Assistant)
└── Week 3-4: Project 2 (Document Q&A)

Month 2:
├── Week 1-2: Project 3 (Code Review Agent)
└── Week 3-4: Extend Project 2 with web UI

Month 3:
├── Week 1-2: Project 4 (Knowledge Base)
└── Week 3-4: Project 5 (Eval Pipeline)

Month 4+:
└── Combine learnings into a production application
```

## Key Takeaways

- Start simple, add complexity gradually
- Each project should teach new concepts
- Extend projects rather than starting fresh
- Build things you'll actually use
- Document as you go

## References

- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [OpenAI Examples](https://platform.openai.com/examples)
- [Vercel AI SDK Examples](https://sdk.vercel.ai/examples)
