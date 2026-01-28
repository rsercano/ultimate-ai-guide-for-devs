---
layout: default
title: Model Serving & Inference
parent: 08 - Advanced Topics
nav_order: 1
---

# Model Serving & Inference

## TL;DR

Serving LLMs in production requires specialized infrastructure. vLLM and TGI provide optimized inference. Quantization reduces memory/cost. Batching improves throughput. Understanding these is essential for cost-effective production systems.

## Why Not Just Use APIs?

| API Providers | Self-Hosting |
|---------------|--------------|
| Simple, managed | Full control |
| Per-token costs | Fixed GPU costs |
| Data leaves your infra | Data stays internal |
| Rate limits | Your capacity limits |
| Vendor lock-in | Model flexibility |

**Self-host when:** High volume, data privacy, cost optimization, or specific model needs.

## Inference Optimization Techniques

### 1. KV Cache

Store computed key-value pairs to avoid recomputation:

```
Without KV Cache:
Token 1: Compute attention for [1]
Token 2: Compute attention for [1, 2]
Token 3: Compute attention for [1, 2, 3]  ← Redundant!

With KV Cache:
Token 1: Compute and cache K1, V1
Token 2: Load K1, V1, compute K2, V2
Token 3: Load K1, V1, K2, V2, compute K3, V3  ← Much faster!
```

### 2. Continuous Batching

Traditional batching waits for all requests to finish:

```
Traditional:
Request A (10 tokens) ████████░░  }
Request B (50 tokens) ████████████████████████████████████  } Wait for longest
Request C (20 tokens) ████████████░░░░░░░░░░░░░░░░░░░░░░  }

Continuous Batching:
Request A (10 tokens) ████████ → Done, slot freed
Request D joins →     ████████████  ← Uses freed slot immediately
Request B (50 tokens) ████████████████████████████████████
Request C (20 tokens) ████████████ → Done, slot freed
```

vLLM pioneered this with PagedAttention.

### 3. Speculative Decoding

Use small model to draft, large model to verify:

```
Draft Model (7B):  Generates 4 tokens quickly
Large Model (70B): Verifies/corrects in one pass

Result: 2-3x speedup for the cost of small model inference
```

## Quantization

Reduce precision to save memory and speed up inference:

| Precision | Bits | Memory (7B) | Quality | Use Case |
|-----------|------|-------------|---------|----------|
| FP32 | 32 | 28GB | Best | Training |
| FP16 | 16 | 14GB | Excellent | Default inference |
| INT8 | 8 | 7GB | Very Good | Production |
| INT4 | 4 | 3.5GB | Good | Resource-constrained |
| GPTQ | 4 | 3.5GB | Good | Optimized 4-bit |
| AWQ | 4 | 3.5GB | Better | Activation-aware |
| GGUF | 2-8 | Variable | Variable | CPU inference (llama.cpp) |

### Quantization Example

```python
# Load 4-bit quantized model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
# 70B model now fits on single A100!
```

## Serving Frameworks

### vLLM

Fastest open-source LLM serving:

```bash
# Install
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000

# Use OpenAI-compatible API
curl http://localhost:8000/v1/completions \
    -d '{"model": "meta-llama/Llama-2-7b-chat-hf", "prompt": "Hello"}'
```

Features:
- PagedAttention (efficient KV cache)
- Continuous batching
- Tensor parallelism
- OpenAI-compatible API

### Text Generation Inference (TGI)

HuggingFace's production server:

```bash
# Docker deployment
docker run --gpus all -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf

# Or use HuggingFace Inference Endpoints (managed)
```

Features:
- Production-hardened
- Quantization support
- Prometheus metrics
- HuggingFace integration

### Ollama

Simplest local deployment:

```bash
# Install and run
curl -fsSL https://ollama.ai/install.sh | sh
ollama run llama2

# API
curl http://localhost:11434/api/generate \
    -d '{"model": "llama2", "prompt": "Hello"}'
```

Best for: Development, testing, personal use.

## GPU Selection

| GPU | VRAM | Models That Fit | Monthly Cost* |
|-----|------|-----------------|---------------|
| RTX 3090 | 24GB | 7B FP16, 13B INT8 | $300 (own) |
| RTX 4090 | 24GB | 7B FP16, 13B INT8 | $400 (own) |
| A10 | 24GB | 7B FP16, 13B INT8 | ~$800 cloud |
| A100 40GB | 40GB | 13B FP16, 70B INT4 | ~$1,500 cloud |
| A100 80GB | 80GB | 30B FP16, 70B INT8 | ~$2,500 cloud |
| H100 | 80GB | 70B FP16 | ~$4,000 cloud |

*Approximate cloud costs vary by provider

## Scaling Patterns

### Tensor Parallelism

Split model across GPUs:

```
Layer 1, Part A → GPU 0  |  Layer 1, Part B → GPU 1
Layer 2, Part A → GPU 0  |  Layer 2, Part B → GPU 1
...

vLLM: --tensor-parallel-size 2
```

### Pipeline Parallelism

Split layers across GPUs:

```
Layers 1-16  → GPU 0
Layers 17-32 → GPU 1
```

### Load Balancing

```
                    ┌─────────────┐
User Requests → ─── │ Load Balancer│ ───┬──→ vLLM Instance 1
                    └─────────────┘    ├──→ vLLM Instance 2
                                       └──→ vLLM Instance 3
```

## Cost Optimization

| Technique | Savings | Trade-off |
|-----------|---------|-----------|
| Quantization (INT8) | 50% memory | Minor quality loss |
| Quantization (INT4) | 75% memory | Noticeable quality loss |
| Spot instances | 60-90% cost | Interruption risk |
| Batch requests | Higher throughput | Added latency |
| Caching | Varies | Cache management |
| Smaller model | Proportional | Capability reduction |

## Key Takeaways

- Use vLLM or TGI for production serving
- Quantization is essential for cost-effective deployment
- Continuous batching dramatically improves throughput
- GPU selection depends on model size and precision
- Consider total cost: GPU + maintenance + latency requirements

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [vLLM Documentation](https://docs.vllm.ai/)
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference)
- [Quantization Guide](https://huggingface.co/docs/transformers/quantization)
