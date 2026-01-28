---
layout: default
title: MLOps for LLMs
parent: "08 - Advanced Topics"
nav_order: 5
---

# MLOps for LLMs

## TL;DR

LLMOps extends traditional MLOps for language model applications. Key concerns: prompt versioning, experiment tracking, model management, deployment pipelines, and production monitoring. This enables reliable, reproducible AI systems.

## LLMOps vs Traditional MLOps

| Aspect | Traditional MLOps | LLMOps |
|--------|-------------------|--------|
| Artifacts | Model weights | Prompts + configs + model refs |
| Training | Regular | Rare (mostly use pre-trained) |
| Evaluation | Metrics on test set | Eval sets + LLM-as-judge |
| Versioning | Model checkpoints | Prompt templates + versions |
| Deployment | Model serving | API gateway + guardrails |
| Monitoring | Accuracy drift | Quality + cost + latency |

## The LLMOps Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLMOps STACK                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    DEVELOPMENT                            │   │
│  │  Prompt Engineering → Testing → Evaluation                │   │
│  │  Tools: LangSmith, Braintrust, PromptLayer               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    VERSIONING                             │   │
│  │  Prompts + Configs + Evals (Git-tracked)                 │   │
│  │  Tools: Git, DVC, Weights & Biases                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    CI/CD PIPELINE                         │   │
│  │  Lint → Test → Eval → Stage → Deploy                     │   │
│  │  Tools: GitHub Actions, CircleCI, Jenkins                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    PRODUCTION                             │   │
│  │  Serving + Monitoring + Logging + Alerting               │   │
│  │  Tools: LangSmith, Helicone, DataDog, custom             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prompt Management

### Version Control for Prompts

```
prompts/
├── v1/
│   ├── summarize.yaml
│   └── classify.yaml
├── v2/
│   ├── summarize.yaml    # Updated version
│   └── classify.yaml
└── current -> v2/         # Symlink to current version
```

### Prompt Template Format

```yaml
# prompts/v2/summarize.yaml
name: summarize
version: "2.0.0"
description: "Summarize documents concisely"
model: gpt-4o
temperature: 0.3
max_tokens: 500

system: |
  You are an expert summarizer. Create concise summaries that 
  capture the key points. Use bullet points for clarity.

user_template: |
  Summarize the following document:
  
  Title: {title}
  Content: {content}
  
  Requirements:
  - Maximum {max_bullets} bullet points
  - Focus on {focus_area}

input_variables:
  - title
  - content
  - max_bullets
  - focus_area

defaults:
  max_bullets: 5
  focus_area: "key findings"
```

### Loading Prompts

```python
import yaml
from pathlib import Path

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts/current"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts = {}
        self._load_all()
    
    def _load_all(self):
        for file in self.prompts_dir.glob("*.yaml"):
            with open(file) as f:
                prompt = yaml.safe_load(f)
                self.prompts[prompt["name"]] = prompt
    
    def render(self, name: str, **kwargs) -> dict:
        prompt = self.prompts[name]
        
        # Apply defaults
        variables = {**prompt.get("defaults", {}), **kwargs}
        
        return {
            "model": prompt["model"],
            "temperature": prompt["temperature"],
            "max_tokens": prompt["max_tokens"],
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user_template"].format(**variables)}
            ]
        }
```

## Experiment Tracking

### Using Weights & Biases

```python
import wandb
from openai import OpenAI

wandb.init(project="llm-experiments")

client = OpenAI()

def tracked_completion(prompt: str, **kwargs):
    # Log inputs
    wandb.log({"prompt": prompt, **kwargs})
    
    start = time.time()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )
    latency = time.time() - start
    
    # Log outputs
    wandb.log({
        "response": response.choices[0].message.content,
        "latency_ms": latency * 1000,
        "tokens_prompt": response.usage.prompt_tokens,
        "tokens_completion": response.usage.completion_tokens,
        "model": kwargs.get("model", "gpt-4o")
    })
    
    return response
```

### Using LangSmith

```python
from langsmith import traceable

@traceable(name="summarize")
def summarize(document: str) -> str:
    prompt = prompt_manager.render("summarize", content=document)
    response = client.chat.completions.create(**prompt)
    return response.choices[0].message.content
```

## CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/llm-ci.yml
name: LLM CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint prompts
        run: python scripts/lint_prompts.py

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: pytest tests/ -v

  eval:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Run evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python evals/run_eval.py --dataset evals/core.json --threshold 0.85
      
      - name: Upload eval results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: evals/results/

  deploy:
    runs-on: ubuntu-latest
    needs: eval
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Update prompt versions
          # Restart services
          # Update feature flags
```

### Evaluation in CI

```python
# evals/run_eval.py
import argparse
import json
from evaluator import EvalPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    
    # Load eval dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    
    # Run evaluation
    pipeline = EvalPipeline(dataset)
    results = pipeline.run()
    
    # Check threshold
    avg_score = results["summary"]["accuracy"]["mean"]
    
    if avg_score < args.threshold:
        print(f"FAILED: Score {avg_score:.2f} < threshold {args.threshold}")
        exit(1)
    
    print(f"PASSED: Score {avg_score:.2f} >= threshold {args.threshold}")
    
    # Save results
    with open("evals/results/latest.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

## Production Monitoring

### Key Metrics

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LLMMetrics:
    # Latency
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    # Tokens
    avg_prompt_tokens: float
    avg_completion_tokens: float
    
    # Cost
    total_cost_usd: float
    cost_per_request_usd: float
    
    # Quality
    error_rate: float
    timeout_rate: float
    user_feedback_positive_rate: float
    
    # Volume
    requests_per_minute: float
    
    timestamp: datetime

class MetricsCollector:
    def record(self, request_id: str, metrics: dict):
        # Send to your metrics system
        self.prometheus.observe(metrics)
        self.datadog.log(metrics)
        
    def alert_if_needed(self, metrics: LLMMetrics):
        if metrics.error_rate > 0.05:
            self.alert("High error rate", metrics.error_rate)
        if metrics.latency_p95_ms > 5000:
            self.alert("High latency", metrics.latency_p95_ms)
        if metrics.cost_per_request_usd > 0.10:
            self.alert("High cost per request", metrics.cost_per_request_usd)
```

### Logging Best Practices

```python
import structlog

logger = structlog.get_logger()

def logged_completion(request_id: str, prompt: str, **kwargs):
    log = logger.bind(
        request_id=request_id,
        model=kwargs.get("model"),
        prompt_preview=prompt[:100]
    )
    
    log.info("llm_request_started")
    
    try:
        start = time.time()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        latency = time.time() - start
        
        log.info("llm_request_completed",
            latency_ms=latency * 1000,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            finish_reason=response.choices[0].finish_reason
        )
        
        return response
        
    except Exception as e:
        log.error("llm_request_failed", error=str(e))
        raise
```

## A/B Testing Framework

```python
class ABTest:
    def __init__(self, name: str, variants: dict):
        self.name = name
        self.variants = variants  # {"control": prompt_v1, "treatment": prompt_v2}
    
    def get_variant(self, user_id: str) -> str:
        # Deterministic assignment
        hash_val = hash(f"{self.name}:{user_id}") % 100
        return "treatment" if hash_val < 50 else "control"
    
    def run(self, user_id: str, input_data: dict) -> tuple[str, str]:
        variant = self.get_variant(user_id)
        prompt = self.variants[variant]
        
        response = execute_prompt(prompt, input_data)
        
        # Log for analysis
        self.log_experiment(user_id, variant, input_data, response)
        
        return variant, response
```

## Model Management

### Model Registry

```python
class ModelRegistry:
    def __init__(self):
        self.models = {}
    
    def register(self, name: str, config: dict):
        """Register a model configuration"""
        self.models[name] = {
            "config": config,
            "registered_at": datetime.now(),
            "status": "staging"
        }
    
    def promote(self, name: str):
        """Promote model to production"""
        self.models[name]["status"] = "production"
        self.models[name]["promoted_at"] = datetime.now()
    
    def rollback(self, name: str):
        """Rollback to previous version"""
        previous = self.get_previous_production(name)
        self.promote(previous)
        self.models[name]["status"] = "deprecated"
    
    def get_production(self, name: str) -> dict:
        """Get current production config"""
        for version in sorted(self.models.keys(), reverse=True):
            if version.startswith(name) and self.models[version]["status"] == "production":
                return self.models[version]["config"]
```

## Key Takeaways

- Version control prompts like code
- Track experiments systematically
- Run evals in CI/CD pipeline
- Monitor latency, cost, and quality in production
- Enable A/B testing for prompt improvements
- Have rollback procedures ready

## References

- [LangSmith](https://docs.smith.langchain.com/)
- [Weights & Biases](https://docs.wandb.ai/)
- [Braintrust](https://www.braintrustdata.com/docs)
- [Chip Huyen — ML Systems Design](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
