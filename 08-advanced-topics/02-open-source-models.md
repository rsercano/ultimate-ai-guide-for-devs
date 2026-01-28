---
layout: default
title: Open Source Models
parent: "08 - Advanced Topics"
nav_order: 2
---

# Open Source Models

## TL;DR

Open source models (Llama, Mistral, Qwen) offer API-provider independence, data privacy, and customization. Understanding the landscape, how to evaluate models, and how to run them locally is essential for production flexibility.

## Why Open Source?

| Closed (GPT-4, Claude) | Open Source |
|------------------------|-------------|
| Best quality (usually) | Competitive quality |
| Data sent to provider | Data stays local |
| API costs scale linearly | Fixed infrastructure cost |
| Limited customization | Full fine-tuning possible |
| Vendor dependency | Model portability |

## Model Landscape (2024-2025)

### Tier 1: Production-Ready

| Model | Sizes | Strengths | License |
|-------|-------|-----------|---------|
| Llama 3 | 8B, 70B | General purpose, community | Meta |
| Mistral | 7B, 8x7B, 8x22B | Efficient, strong reasoning | Apache 2.0 |
| Qwen 2 | 0.5B-72B | Multilingual, long context | Qwen |
| DeepSeek | 7B-67B | Code, math, reasoning | MIT |
| Gemma | 2B, 7B | Lightweight, efficient | Google |

### Tier 2: Specialized

| Model | Specialty | Use Case |
|-------|-----------|----------|
| CodeLlama | Code generation | IDE assistants |
| StarCoder | Code completion | Autocomplete |
| Phi-3 | Small but capable | Edge deployment |
| Mixtral MoE | Mixture of experts | High throughput |

### How to Choose

```
Decision Tree:

Need multilingual?
├── Yes → Qwen or Llama 3
└── No → Continue

Need code specialization?
├── Yes → CodeLlama or DeepSeek Coder
└── No → Continue

GPU memory constraint?
├── < 8GB → Phi-3 (3.8B) or Gemma 2B
├── 8-16GB → Mistral 7B or Llama 8B (quantized)
├── 24GB → Llama 8B FP16 or 70B INT4
├── 40GB+ → Llama 70B INT8
└── 80GB+ → Llama 70B FP16

High throughput needed?
├── Yes → Mixtral MoE
└── No → Dense model fine
```

## Running Models Locally

### Option 1: Ollama (Simplest)

```bash
# Install
curl -fsSL https://ollama.ai/install.sh | sh

# Run models
ollama run llama3
ollama run mistral
ollama run codellama

# API usage
curl http://localhost:11434/api/generate \
    -d '{"model": "llama3", "prompt": "Explain recursion"}'
```

### Option 2: llama.cpp (CPU-friendly)

```bash
# Get quantized model
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Run
./main -m llama-2-7b.Q4_K_M.gguf -p "Hello, I am"
```

### Option 3: HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Option 4: vLLM (Production)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
```

## Fine-tuning Open Source Models

### When to Fine-tune

| Do Fine-tune | Don't Fine-tune |
|--------------|-----------------|
| Custom output format | Adding knowledge (use RAG) |
| Domain terminology | One-off tasks |
| Consistent style/tone | Frequently changing needs |
| Task specialization | Small improvements |

### Quick Fine-tuning with Unsloth

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_dataset

# Load model (4x faster than HF)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Load your dataset
dataset = load_dataset("your_dataset")

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()

# Save
model.save_pretrained("my-finetuned-model")
```

## Model Evaluation

### Benchmarks to Know

| Benchmark | What It Measures |
|-----------|------------------|
| MMLU | Knowledge across subjects |
| HellaSwag | Common sense reasoning |
| HumanEval | Code generation |
| GSM8K | Math reasoning |
| TruthfulQA | Factual accuracy |
| MT-Bench | Conversation quality |

### Quick Local Evaluation

```python
# Using lm-eval
from lm_eval import simple_evaluate

results = simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["hellaswag", "mmlu"],
    batch_size=8
)
print(results)
```

### Practical Evaluation

Benchmarks don't tell the whole story. Always test on your specific use case:

```python
test_cases = [
    {"input": "Your typical user query", "expected_contains": ["key", "words"]},
    {"input": "Edge case query", "expected_behavior": "refuses appropriately"},
    # ... more cases
]

for case in test_cases:
    response = model.generate(case["input"])
    # Assert expected behavior
```

## Deployment Options

| Option | Complexity | Cost | Best For |
|--------|------------|------|----------|
| Ollama on laptop | Low | Free | Development |
| vLLM on cloud GPU | Medium | $$$ | Production |
| HuggingFace Endpoints | Low | $$$$ | Managed production |
| Replicate | Low | $$$ | Quick deployment |
| Modal | Medium | $$ | Serverless GPU |
| RunPod | Medium | $$ | Flexible GPU rental |

## GGUF Quantization Naming

Understanding quantization file names:

```
llama-2-7b-chat.Q4_K_M.gguf
           │    │ │ │
           │    │ │ └── Method variant (S/M/L)
           │    │ └──── K-quant (better quality)
           │    └────── 4-bit quantization
           └─────────── Model name

Quality ranking (same bits):
Q4_K_M > Q4_K_S > Q4_0
```

## Key Takeaways

- Open source models are competitive with closed models for many tasks
- Llama 3 and Mistral are the go-to general-purpose models
- Quantization makes large models accessible on consumer hardware
- Fine-tuning is easier than ever with tools like Unsloth
- Always evaluate on your specific use case, not just benchmarks

## References

- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Ollama](https://ollama.ai/)
- [Unsloth](https://github.com/unslothai/unsloth)
- [TheBloke's Quantized Models](https://huggingface.co/TheBloke)
