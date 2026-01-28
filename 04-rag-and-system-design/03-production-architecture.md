
# Production Architecture

## TL;DR

Production LLM systems need more than just API calls. You need caching, guardrails, observability, fallbacks, and cost controls. This document covers patterns for building reliable, maintainable LLM applications.

## Reference Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION LLM SYSTEM                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Client Request                                                      │
│        │                                                              │
│        ▼                                                              │
│   ┌─────────────┐                                                    │
│   │   Gateway   │ ← Rate limiting, auth, request validation          │
│   └─────────────┘                                                    │
│        │                                                              │
│        ▼                                                              │
│   ┌─────────────┐     ┌─────────────┐                                │
│   │   Cache     │────▶│ Cache Hit?  │──Yes──▶ Return cached          │
│   └─────────────┘     └─────────────┘                                │
│                             │ No                                      │
│                             ▼                                         │
│   ┌─────────────┐     ┌─────────────┐                                │
│   │  Guardrails │◀────│   Input     │ ← PII detection, prompt inject │
│   │   (Input)   │     │  Processing │                                │
│   └─────────────┘     └─────────────┘                                │
│                             │                                         │
│                             ▼                                         │
│   ┌─────────────┐     ┌─────────────┐                                │
│   │  Vector DB  │◀────│  Retrieval  │ ← If using RAG                 │
│   └─────────────┘     │   (RAG)     │                                │
│                       └─────────────┘                                │
│                             │                                         │
│                             ▼                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│   │   Primary   │◀────│     LLM     │────▶│   Fallback  │           │
│   │   Model     │     │   Router    │     │    Model    │           │
│   └─────────────┘     └─────────────┘     └─────────────┘           │
│                             │                                         │
│                             ▼                                         │
│   ┌─────────────┐     ┌─────────────┐                                │
│   │  Guardrails │◀────│   Output    │ ← Content filtering, format    │
│   │  (Output)   │     │  Processing │                                │
│   └─────────────┘     └─────────────┘                                │
│                             │                                         │
│                             ▼                                         │
│   ┌─────────────┐     ┌─────────────┐                                │
│   │   Logging   │◀────│  Response   │──────▶ Client                  │
│   │ Observability│    └─────────────┘                                │
│   └─────────────┘                                                    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Caching

Cache identical or similar requests:

```python
import hashlib
import redis

def get_cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

def cached_completion(prompt: str, model: str):
    cache_key = get_cache_key(prompt, model)
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Generate
    response = llm.complete(prompt, model=model)
    
    # Cache with TTL
    redis_client.setex(cache_key, 3600, json.dumps(response))
    return response
```

**Semantic caching**: Cache similar (not just identical) queries using embedding similarity.

### 2. Guardrails

#### Input Guardrails

```python
def validate_input(user_input: str) -> tuple[bool, str]:
    # PII detection
    if contains_pii(user_input):
        return False, "Please don't include personal information"
    
    # Prompt injection detection
    if detect_injection(user_input):
        log_security_event(user_input)
        return False, "Invalid input"
    
    # Content policy
    if violates_policy(user_input):
        return False, "This request cannot be processed"
    
    return True, user_input
```

#### Output Guardrails

```python
def validate_output(response: str) -> str:
    # Remove any leaked system prompts
    response = remove_system_prompt_leaks(response)
    
    # Check for harmful content
    if is_harmful(response):
        return "I cannot provide that information."
    
    # Validate format if structured output expected
    if not valid_json(response):
        response = repair_json(response)
    
    return response
```

### 3. Model Router / Fallbacks

```python
class LLMRouter:
    def __init__(self):
        self.primary = OpenAI(model="gpt-4")
        self.fallback = Anthropic(model="claude-3-sonnet")
        self.cheap = OpenAI(model="gpt-3.5-turbo")
    
    def complete(self, prompt: str, complexity: str = "auto"):
        # Route by complexity
        if complexity == "simple":
            return self._try_with_fallback(self.cheap, prompt)
        elif complexity == "complex":
            return self._try_with_fallback(self.primary, prompt)
        else:
            # Auto-detect or use classifier
            return self._smart_route(prompt)
    
    def _try_with_fallback(self, primary, prompt):
        try:
            return primary.complete(prompt, timeout=30)
        except (RateLimitError, TimeoutError):
            return self.fallback.complete(prompt)
```

### 4. Observability

Track everything:

```python
from opentelemetry import trace
import structlog

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)

def llm_call(prompt: str):
    with tracer.start_as_current_span("llm_call") as span:
        start = time.time()
        
        span.set_attribute("prompt_tokens", count_tokens(prompt))
        span.set_attribute("model", "gpt-4")
        
        response = llm.complete(prompt)
        
        latency = time.time() - start
        span.set_attribute("response_tokens", count_tokens(response))
        span.set_attribute("latency_ms", latency * 1000)
        
        logger.info("llm_call",
            model="gpt-4",
            prompt_tokens=count_tokens(prompt),
            response_tokens=count_tokens(response),
            latency_ms=latency * 1000,
            cost=calculate_cost(prompt, response)
        )
        
        return response
```

**Key metrics to track:**
- Latency (p50, p95, p99)
- Token usage (input, output)
- Cost per request
- Error rates by type
- Cache hit rate
- User satisfaction (thumbs up/down)

### 5. Rate Limiting & Cost Control

```python
class CostController:
    def __init__(self, daily_budget: float):
        self.daily_budget = daily_budget
        self.spent_today = 0
    
    def can_proceed(self, estimated_cost: float) -> bool:
        if self.spent_today + estimated_cost > self.daily_budget:
            alert_ops_team("Budget limit approaching")
            return False
        return True
    
    def record_spend(self, actual_cost: float):
        self.spent_today += actual_cost

# Per-user rate limiting
rate_limiter = RateLimiter(
    requests_per_minute=20,
    tokens_per_day=100_000
)
```

## Latency Optimization

| Technique | Latency Reduction | Trade-off |
|-----------|------------------|-----------|
| Streaming | Perceived latency | Complexity |
| Caching | 100% for hits | Staleness |
| Smaller model | 50-80% | Quality |
| Shorter prompts | 20-40% | Context |
| Edge deployment | Network latency | Cost |

### Streaming Example

```python
async def stream_response(prompt: str):
    async for chunk in llm.stream(prompt):
        yield chunk.content
        
# Client receives tokens as they're generated
# First token in ~200ms vs ~2s for full response
```

## Error Handling Patterns

```python
class LLMService:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, TimeoutError))
    )
    async def complete(self, prompt: str):
        try:
            return await self.primary.complete(prompt)
        except RateLimitError:
            await self.switch_to_fallback()
            raise
        except ContentPolicyError:
            return self.safe_response()
        except InvalidResponseError:
            return await self.retry_with_clearer_prompt(prompt)
```

## Key Takeaways

- Production systems need more than LLM API calls
- Cache aggressively (identical and semantic)
- Always have fallback models
- Guardrails on both input and output
- Observe everything, alert on anomalies
- Control costs with budgets and rate limits

## References

- [Eugene Yan — Patterns for LLM Systems](https://eugeneyan.com/writing/llm-patterns/)
- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Full Stack Deep Learning — LLM Ops](https://fullstackdeeplearning.com/llm-bootcamp/)
