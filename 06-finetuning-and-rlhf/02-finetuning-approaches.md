---
layout: default
title: Fine-tuning Approaches
parent: 06 - Fine-tuning & RLHF
nav_order: 2
---

# Fine-tuning Approaches

## TL;DR

Full fine-tuning updates all parameters (expensive, powerful). LoRA adds small trainable adapters (cheap, effective). QLoRA combines LoRA with quantization (runs on consumer GPUs). Choose based on your compute budget and data size.

## When to Fine-tune

### Good Reasons

| Use Case | Example |
|----------|---------|
| Custom output format | Always return specific JSON schema |
| Domain adaptation | Medical, legal, code-specific language |
| Style/tone | Match brand voice consistently |
| Task specialization | Classification, extraction |
| Smaller model, same quality | Distill GPT-4 behavior to smaller model |

### Bad Reasons

| Don't Fine-tune For | Do This Instead |
|--------------------|-----------------|
| Adding knowledge | RAG |
| One-off tasks | Prompt engineering |
| Frequently changing data | RAG |
| Small improvements | Better prompts |

## Full Fine-tuning

Update all model parameters:

```python
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)
trainer.train()
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Maximum capability | Requires full model in memory |
| Can change any behavior | Expensive compute |
| Best for large datasets | Risk of catastrophic forgetting |
| | Need to store full model copy |

### Requirements

| Model Size | GPU Memory | Approximate Cost |
|------------|------------|------------------|
| 7B | 80GB+ (A100) | $1K-5K |
| 13B | 160GB+ | $5K-20K |
| 70B | 640GB+ | $50K+ |

## LoRA (Low-Rank Adaptation)

Add small trainable matrices, freeze original weights:

```
Original: Y = X × W           (W is frozen)
LoRA:     Y = X × W + X × A × B   (A, B are trained)

Where:
- W: Original weight matrix [d × d]
- A: Low-rank matrix [d × r]
- B: Low-rank matrix [r × d]
- r: Rank (typically 8-64), much smaller than d
```

### Why It Works

```
Full fine-tuning:    Update 7B parameters
LoRA (rank 16):      Update ~10M parameters (0.1%)

Hypothesis: The update needed for fine-tuning 
lies in a low-rank subspace
```

### Implementation

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                     # Rank
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none"
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
peft_model = get_peft_model(model, config)

peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

### Pros/Cons

| Pros | Cons |
|------|------|
| 10-100x fewer parameters | Slightly less capable than full |
| Fits on consumer GPUs | Need to choose target modules |
| Quick training | May not capture all changes |
| Merge into base model | |

## QLoRA (Quantized LoRA)

LoRA + 4-bit quantization:

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config
)

model = prepare_model_for_kbit_training(model)
# Now add LoRA adapters as before
```

### Memory Comparison

| Model | Full FT | LoRA | QLoRA |
|-------|---------|------|-------|
| 7B | 80GB | 16GB | 6GB |
| 13B | 160GB | 32GB | 10GB |
| 70B | 640GB | 160GB | 48GB |

QLoRA makes 70B fine-tuning possible on a single A100!

## Comparison Table

| Method | Parameters | Memory | Quality | Speed | Use Case |
|--------|------------|--------|---------|-------|----------|
| Full FT | 100% | Very High | Best | Slow | Unlimited budget |
| LoRA | 0.1-1% | Medium | Very Good | Fast | Most cases |
| QLoRA | 0.1-1% | Low | Good | Medium | Consumer GPU |

## Data Requirements

| Data Size | Recommendation |
|-----------|----------------|
| < 100 examples | Probably use prompting |
| 100-1K examples | LoRA might work |
| 1K-10K examples | LoRA works well |
| 10K+ examples | Consider full fine-tuning |

## Training Tips

### Data Quality > Quantity

```python
# ❌ Bad: Low-quality data
{"input": "hi", "output": "hello"}

# ✓ Good: High-quality examples
{
    "input": "Explain quantum entanglement to a high school student",
    "output": "Imagine you have two coins that are magically connected..."
}
```

### Hyperparameters

```python
training_args = TrainingArguments(
    learning_rate=2e-4,          # Lower than pretraining
    num_train_epochs=3,          # Usually 1-5 epochs
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
)
```

### Validation

Always hold out a validation set:

```python
dataset = dataset.train_test_split(test_size=0.1)
```

Watch for:
- Validation loss increasing (overfitting)
- Training loss not decreasing (learning rate too low)
- Loss exploding (learning rate too high)

## API-based Fine-tuning

### OpenAI

```python
# Prepare JSONL file
{"messages": [{"role": "system", "content": "..."}, 
              {"role": "user", "content": "..."}, 
              {"role": "assistant", "content": "..."}]}

# Upload and train
client.files.create(file=open("data.jsonl"), purpose="fine-tune")
client.fine_tuning.jobs.create(training_file="file-xxx", model="gpt-4o-mini")
```

Pros: No GPU needed, simple API
Cons: Data leaves your control, less flexibility

## Key Takeaways

- Use LoRA for most fine-tuning (best cost/quality trade-off)
- QLoRA for limited GPU memory
- Full fine-tuning only with large data and budget
- Data quality matters more than quantity
- Always validate—watch for overfitting

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
