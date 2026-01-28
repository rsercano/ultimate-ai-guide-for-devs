---
layout: default
title: Evaluation Strategies
parent: 05 - Prompting & Eval
nav_order: 3
---

# Evaluation Strategies

## TL;DR

You can't improve what you don't measure. LLM evaluation requires a mix of automated metrics, model-as-judge approaches, and human evaluation. Build eval sets early; they're your most valuable asset.

## Why Eval Is Hard

Unlike traditional ML:
- No single correct answer
- Quality is subjective
- Failure modes are diverse
- Distribution shifts constantly

## Evaluation Types

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PYRAMID                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                    ▲                                         │
│                   ╱ ╲       Human Evaluation                 │
│                  ╱   ╲      (expensive, slow, gold standard) │
│                 ╱─────╲                                      │
│                ╱       ╲    LLM-as-Judge                     │
│               ╱         ╲   (scalable, good proxy)           │
│              ╱───────────╲                                   │
│             ╱             ╲  Automated Metrics               │
│            ╱               ╲ (fast, limited scope)           │
│           ╱─────────────────╲                                │
│          ╱                   ╲ Unit Tests                    │
│         ╱                     ╲(format, basic correctness)   │
│        ╱───────────────────────╲                             │
│                                                              │
│   Speed/Scale ──────────────────────────────▶ Quality        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Building an Eval Set

### Start with Real Examples

```python
eval_set = [
    {
        "id": "customer-001",
        "input": "I can't log into my account",
        "expected_topics": ["password_reset", "account_access"],
        "expected_tone": "helpful",
        "source": "production_logs"
    },
    {
        "id": "customer-002",
        "input": "Your product broke my computer!!!",
        "expected_topics": ["escalation", "technical_issue"],
        "expected_tone": "empathetic",
        "source": "support_tickets"
    }
]
```

### Categories to Cover

| Category | Examples |
|----------|----------|
| Happy path | Normal, expected inputs |
| Edge cases | Empty input, very long input |
| Adversarial | Prompt injection, policy violations |
| Ambiguous | Multiple valid interpretations |
| Domain-specific | Technical terms, jargon |

### Size Guidelines

| Stage | Eval Set Size | Purpose |
|-------|---------------|---------|
| Development | 20-50 | Quick iteration |
| Pre-deployment | 200-500 | Confidence check |
| Production | 1000+ | Statistical significance |

## Automated Metrics

### Exact Match

```python
def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip() == reference.strip() else 0.0
```

Use for: Classification, structured output, yes/no questions.

### Semantic Similarity

```python
def semantic_similarity(prediction: str, reference: str) -> float:
    pred_embedding = embed(prediction)
    ref_embedding = embed(reference)
    return cosine_similarity(pred_embedding, ref_embedding)
```

Use for: Paraphrase detection, answer similarity.

### ROUGE / BLEU

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference, prediction)
```

Use for: Summarization, translation.

### Custom Metrics

```python
def response_quality(response: str) -> dict:
    return {
        "has_greeting": response.lower().startswith(("hi", "hello")),
        "within_length": 50 < len(response) < 500,
        "no_forbidden_words": not any(w in response.lower() 
                                       for w in FORBIDDEN),
        "valid_json": is_valid_json(response) if JSON_EXPECTED else True
    }
```

## LLM-as-Judge

Use a capable model to evaluate outputs:

```python
JUDGE_PROMPT = """
Evaluate the following response on a scale of 1-5 for each criterion.

Response to evaluate:
{response}

Reference answer (if available):
{reference}

Criteria:
1. Accuracy: Is the information correct?
2. Relevance: Does it address the question?
3. Clarity: Is it well-written and easy to understand?
4. Completeness: Does it cover all important points?

Respond in JSON format:
{
  "accuracy": {"score": 1-5, "reason": "..."},
  "relevance": {"score": 1-5, "reason": "..."},
  "clarity": {"score": 1-5, "reason": "..."},
  "completeness": {"score": 1-5, "reason": "..."}
}
"""

def llm_judge(response: str, reference: str = None) -> dict:
    prompt = JUDGE_PROMPT.format(response=response, reference=reference)
    result = llm.complete(prompt, model="gpt-4")
    return json.loads(result)
```

### Best Practices for LLM-as-Judge

| Practice | Why |
|----------|-----|
| Use stronger model as judge | GPT-4 judging GPT-3.5 outputs |
| Randomize order | Avoid position bias |
| Include rubric | Consistent scoring criteria |
| Request reasoning | Improves reliability |
| Calibrate with humans | Validate judge accuracy |

## A/B Testing in Production

```python
class ABExperiment:
    def __init__(self, variants: dict, traffic_split: dict):
        self.variants = variants  # {"control": prompt_a, "treatment": prompt_b}
        self.split = traffic_split  # {"control": 0.5, "treatment": 0.5}
    
    def get_variant(self, user_id: str) -> str:
        # Deterministic assignment based on user_id
        hash_val = hash(user_id) % 100
        cumulative = 0
        for variant, percentage in self.split.items():
            cumulative += percentage * 100
            if hash_val < cumulative:
                return variant
    
    def track(self, user_id: str, variant: str, metrics: dict):
        log_event("ab_experiment", {
            "user_id": user_id,
            "variant": variant,
            "metrics": metrics
        })
```

### Metrics to Track

| Metric | Type | What It Measures |
|--------|------|------------------|
| Task success rate | Outcome | Did user complete goal? |
| User satisfaction | Survey | Thumbs up/down, ratings |
| Engagement | Behavior | Follow-up questions, session length |
| Latency | Technical | Response time |
| Cost | Technical | Tokens used per request |

## Eval Pipeline Example

```python
class EvalPipeline:
    def __init__(self, eval_set: list, metrics: list):
        self.eval_set = eval_set
        self.metrics = metrics
    
    def run(self, model_fn) -> dict:
        results = []
        
        for example in self.eval_set:
            prediction = model_fn(example["input"])
            
            scores = {}
            for metric in self.metrics:
                scores[metric.name] = metric.compute(
                    prediction=prediction,
                    reference=example.get("expected"),
                    metadata=example
                )
            
            results.append({
                "id": example["id"],
                "input": example["input"],
                "prediction": prediction,
                "scores": scores
            })
        
        # Aggregate
        return {
            "results": results,
            "summary": self._aggregate(results)
        }
    
    def _aggregate(self, results: list) -> dict:
        summary = {}
        for metric in self.metrics:
            scores = [r["scores"][metric.name] for r in results]
            summary[metric.name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores)
            }
        return summary
```

## Regression Testing

```python
def regression_check(new_results: dict, baseline: dict, threshold: float = 0.05):
    regressions = []
    
    for metric, new_score in new_results["summary"].items():
        old_score = baseline["summary"][metric]["mean"]
        new_score_mean = new_score["mean"]
        
        if new_score_mean < old_score - threshold:
            regressions.append({
                "metric": metric,
                "old": old_score,
                "new": new_score_mean,
                "delta": new_score_mean - old_score
            })
    
    if regressions:
        raise RegressionError(f"Regressions detected: {regressions}")
```

## Key Takeaways

- Build eval sets from real production examples
- Use multiple evaluation methods (automated + LLM-judge + human)
- Track regressions in CI/CD pipeline
- A/B test significant changes in production
- Eval sets are living documents—update them regularly

## References

- [Braintrust Evals](https://www.braintrustdata.com/docs)
- [LMSYS Chatbot Arena](https://chat.lmsys.org/)
- [OpenAI Evals Framework](https://github.com/openai/evals)
- [Anthropic — Evaluations](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests)
