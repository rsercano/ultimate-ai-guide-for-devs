
# Production Checklist

## TL;DR

Before deploying an LLM application to production, ensure you've addressed reliability, security, cost, observability, and user experience. This checklist covers the essentials.

## Pre-Launch Checklist

### 1. Reliability

- [ ] **Fallback models configured**
  ```python
  # Primary → Fallback → Fallback 2
  models = ["gpt-4o", "claude-3-sonnet", "gpt-3.5-turbo"]
  ```

- [ ] **Retry logic with exponential backoff**
  ```python
  @retry(wait=wait_exponential(min=1, max=60))
  def call_llm(prompt):
      ...
  ```

- [ ] **Timeouts set appropriately**
  ```python
  response = client.chat.completions.create(
      ...,
      timeout=30.0  # Don't wait forever
  )
  ```

- [ ] **Graceful degradation**
  - What happens when LLM is unavailable?
  - Can users still use core functionality?

- [ ] **Health checks**
  ```python
  @app.get("/health")
  def health():
      # Check LLM connectivity
      # Check vector DB connectivity
      return {"status": "healthy"}
  ```

### 2. Security

- [ ] **API keys secured**
  - Not in code or version control
  - Rotated regularly
  - Minimal permissions

- [ ] **Input validation**
  ```python
  def validate_input(user_input: str) -> bool:
      if len(user_input) > MAX_INPUT_LENGTH:
          return False
      if contains_injection_patterns(user_input):
          return False
      return True
  ```

- [ ] **Output filtering**
  - PII detection and removal
  - Content policy enforcement

- [ ] **Rate limiting**
  ```python
  # Per user/API key
  rate_limit = RateLimiter(requests_per_minute=20)
  ```

- [ ] **Prompt injection mitigations**
  - Input/output delimiters
  - Instruction hierarchy
  - Input sanitization

- [ ] **Data privacy compliance**
  - GDPR/CCPA considerations
  - Data retention policies
  - User consent for data usage

### 3. Cost Control

- [ ] **Budget limits set**
  ```python
  daily_budget = 100  # USD
  if spent_today > daily_budget:
      reject_request()
  ```

- [ ] **Token counting**
  ```python
  import tiktoken
  
  def count_tokens(text: str, model: str) -> int:
      enc = tiktoken.encoding_for_model(model)
      return len(enc.encode(text))
  ```

- [ ] **Cost alerts configured**
  - 50%, 80%, 100% of budget
  - Daily/weekly summaries

- [ ] **Caching strategy**
  - Semantic caching for similar queries
  - TTL appropriate for use case

- [ ] **Model selection by task**
  - Simple tasks → cheaper models
  - Complex tasks → capable models

### 4. Observability

- [ ] **Structured logging**
  ```python
  logger.info("llm_call", extra={
      "model": model,
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "latency_ms": latency_ms,
      "user_id": user_id,
      "cost_usd": cost
  })
  ```

- [ ] **Tracing**
  - Request ID through entire pipeline
  - Span for each component (retrieval, generation, etc.)

- [ ] **Metrics dashboards**
  - Latency (p50, p95, p99)
  - Error rates by type
  - Token usage over time
  - Cost per user/feature

- [ ] **Alerting**
  - Error rate spikes
  - Latency degradation
  - Budget thresholds
  - Model availability

### 5. User Experience

- [ ] **Streaming responses**
  ```python
  async def stream_response(prompt: str):
      async for chunk in client.chat.completions.create(
          ..., stream=True
      ):
          yield chunk.choices[0].delta.content
  ```

- [ ] **Loading states**
  - Clear indication that AI is processing
  - Estimated wait time if long

- [ ] **Error messages**
  - User-friendly, not technical
  - Actionable when possible

- [ ] **Feedback mechanism**
  - Thumbs up/down
  - Report issues
  - Regenerate option

- [ ] **Transparency**
  - Indicate AI-generated content
  - Show sources/citations when available

### 6. Evaluation & Quality

- [ ] **Eval dataset exists**
  - Representative of production traffic
  - Updated regularly

- [ ] **Regression tests in CI**
  ```yaml
  # .github/workflows/eval.yml
  - name: Run LLM Evals
    run: python evals/run_eval.py --threshold 0.85
  ```

- [ ] **A/B testing infrastructure**
  - Can compare prompts/models in production
  - Statistical significance checks

- [ ] **Human review process**
  - Regular sampling of responses
  - Escalation path for issues

### 7. Documentation

- [ ] **System architecture documented**
- [ ] **Prompt templates versioned**
- [ ] **Runbooks for common issues**
- [ ] **On-call procedures**

## Launch Day Checklist

```
□ All CI/CD checks passing
□ Staging environment tested
□ Rollback plan documented
□ On-call engineer assigned
□ Monitoring dashboards open
□ Alert channels configured
□ Customer support briefed
□ Feature flag ready for kill switch
```

## Post-Launch

### Week 1
- [ ] Monitor error rates and latency
- [ ] Review user feedback
- [ ] Check cost tracking
- [ ] Address critical issues

### Month 1
- [ ] Analyze usage patterns
- [ ] Update eval dataset with production examples
- [ ] Optimize prompts based on feedback
- [ ] Review and adjust rate limits

### Ongoing
- [ ] Monthly cost reviews
- [ ] Quarterly eval updates
- [ ] Regular prompt optimization
- [ ] Model version updates

## Quick Reference: Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Rate limiting | 429 errors spike | Add backoff, caching, fallbacks |
| High latency | p95 > 5s | Streaming, smaller models, caching |
| High cost | Budget exceeded | Caching, model routing, usage limits |
| Bad responses | User complaints | Update prompts, add guardrails |
| Hallucinations | Incorrect info | Add RAG, fact-checking, citations |

## Key Takeaways

- Don't launch without fallbacks and monitoring
- Cost control is critical—set limits early
- User experience matters as much as accuracy
- Build eval into your CI/CD pipeline
- Plan for things to go wrong

## References

- [OpenAI Production Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [Anthropic Safety Guide](https://docs.anthropic.com/claude/docs/claude-safety-guide)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
