
# Production Checklist: Do's and Don'ts

## TL;DR

Hard-earned lessons from deploying LLM applications. Follow the do's, avoid the don'ts.

---

## API & Model Management

### Do's

| Practice | Why |
|----------|-----|
| Configure fallback models | GPT-4 down? Claude takes over. Zero user impact |
| Set explicit timeouts (10-30s) | LLM calls can hang forever without them |
| Implement circuit breakers | Prevent cascade failures when provider is degraded |
| Version your model configurations | Know exactly what was running when issues occurred |
| Use async calls where possible | Don't block threads waiting for 5s LLM responses |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Hardcode model names in business logic | Can't switch providers without code changes |
| Trust provider uptime | Every major provider has had multi-hour outages |
| Retry indefinitely | You'll burn budget and still fail |
| Ignore model deprecation notices | Your app breaks when they sunset the model |
| Share API keys across environments | Can't track costs, security nightmare |

---

## Prompt Engineering in Production

### Do's

| Practice | Why |
|----------|-----|
| Version control all prompts | Prompts are code. Treat them that way |
| Use prompt templates with typed variables | Catch injection and missing vars early |
| A/B test prompt changes | "Improved" prompts often regress on edge cases |
| Include few-shot examples for critical tasks | Dramatically improves consistency |
| Set temperature to 0 for deterministic tasks | Same input should give same output |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Edit prompts directly in production | No rollback, no audit trail, instant regret |
| Use user input directly in system prompts | Prompt injection waiting to happen |
| Make prompts too long | Wastes tokens, confuses the model |
| Assume prompts transfer between models | GPT prompt ≠ Claude prompt ≠ Llama prompt |
| Skip prompt regression testing | One "fix" breaks three other use cases |

---

## Cost Control

### Do's

| Practice | Why |
|----------|-----|
| Set hard budget limits per user/tenant | One runaway user can burn your monthly budget in hours |
| Implement semantic caching | Same questions don't need new API calls |
| Route simple tasks to cheaper models | Classification doesn't need GPT-4 |
| Count tokens before sending | Know costs before they hit your bill |
| Alert at 50%, 80%, 100% of budget | Never be surprised by costs |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Use GPT-4 for everything | 10-30x more expensive than necessary |
| Let users send unlimited context | One user pastes War and Peace, you pay $50 |
| Ignore token costs in free tier | Sets wrong expectations, impossible to monetize |
| Cache without TTL | Stale responses, wasted storage |
| Wait for monthly bills to notice cost spikes | Damage already done |

---

## Security

### Do's

| Practice | Why |
|----------|-----|
| Validate and sanitize all inputs | First line of defense against injection |
| Use separate system/user message roles | Models handle instruction hierarchy |
| Scan outputs for PII before returning | Models can leak training data or echo user PII |
| Implement rate limiting per user | Prevents abuse and runaway costs |
| Log all prompts and responses (encrypted) | Essential for debugging and compliance |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Put secrets in prompts | They end up in logs, caches, error reports |
| Trust model output without validation | Models hallucinate, lie, and get confused |
| Let users see raw error messages | Exposes system details, prompt fragments |
| Skip content filtering on outputs | Your app says something regrettable |
| Store prompts with user data unencrypted | GDPR/CCPA violation, breach liability |

---

## Reliability

### Do's

| Practice | Why |
|----------|-----|
| Implement graceful degradation | "AI unavailable" is better than broken app |
| Use exponential backoff with jitter | Prevents thundering herd on recovery |
| Health check all external dependencies | Know before users tell you |
| Design for partial failures | Vector DB down? Serve cached results |
| Set up dead letter queues for failed requests | Retry or analyze later, never lose data |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Make LLM calls in the critical path without fallback | LLM down = app down |
| Retry failed requests immediately | Makes provider problems worse |
| Assume idempotency | Same request can give wildly different responses |
| Ignore cold start latency | First request takes 2-5x longer |
| Deploy on Friday afternoon | Murphy's law applies to LLM systems too |

---

## Observability

### Do's

| Practice | Why |
|----------|-----|
| Log: model, tokens, latency, cost per request | Debug anything, optimize everything |
| Trace full request lifecycle | Find where time/tokens are being spent |
| Track p50, p95, p99 latency separately | Averages hide real user pain |
| Alert on error rate changes, not thresholds | Catch degradation before it's critical |
| Sample and review random production responses | Find quality issues before users report |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Log only errors | Can't debug slow responses or quality issues |
| Aggregate all requests in one metric | Can't tell which features are problems |
| Skip correlation IDs | Impossible to trace user issues end-to-end |
| Alert on every error | Alert fatigue, real issues get ignored |
| Wait for user complaints to find issues | They won't complain, they'll just leave |

---

## User Experience

### Do's

| Practice | Why |
|----------|-----|
| Stream responses | Perceived latency drops dramatically |
| Show "AI is thinking" indicators | Users tolerate waits when they see progress |
| Let users regenerate responses | Models are non-deterministic, embrace it |
| Provide feedback buttons (👍👎) | Free evaluation data from real users |
| Display sources/citations when available | Builds trust, enables verification |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Block UI while waiting for LLM | 5-30 seconds of frozen screen |
| Show raw model errors to users | "context_length_exceeded" means nothing to them |
| Pretend AI is human | Users feel deceived when they find out |
| Hide AI-generated content markers | Legal issues in many jurisdictions |
| Make users wait for non-essential AI features | Core app should work without AI |

---

## Evaluation & Quality

### Do's

| Practice | Why |
|----------|-----|
| Build eval dataset from day one | Can't improve what you can't measure |
| Run evals in CI/CD | Catch regressions before production |
| Include edge cases and adversarial examples | Models fail creatively on unusual inputs |
| Update eval set with production failures | Your test set should reflect real problems |
| Use LLM-as-judge for scalable evaluation | Human eval doesn't scale, LLM eval does |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Test only happy path | Edge cases cause 80% of production issues |
| Use synthetic data exclusively | Doesn't represent real user behavior |
| Skip evaluation before prompt changes | "Obviously better" changes often aren't |
| Measure only accuracy | Fast and wrong is still wrong |
| Ignore evaluation failures in CI | They become expected, then ignored |

---

## RAG-Specific

### Do's

| Practice | Why |
|----------|-----|
| Monitor retrieval quality separately | Bad retrieval = bad generation, know which failed |
| Set similarity thresholds | Return nothing rather than irrelevant context |
| Implement chunk overlap | Important info often spans chunk boundaries |
| Update embeddings when source docs change | Stale embeddings = stale answers |
| Show retrieved sources to users | Let them verify, builds trust |

### Don'ts

| Anti-pattern | Consequence |
|--------------|-------------|
| Stuff context window with every chunk | Model gets confused, costs explode |
| Use single embedding model for everything | Different content types need different models |
| Skip hybrid search for production | Semantic-only misses keyword matches |
| Embed once and forget | Source data changes, embeddings become stale |
| Trust top-k results blindly | k=10 of bad results is still bad |

---

## Pre-Launch Checklist

```
□ Fallback models configured and tested
□ Timeouts set on all external calls
□ Budget limits implemented and alerting
□ Rate limiting per user/tenant
□ Input validation and output filtering
□ Structured logging for all LLM calls
□ Error handling returns user-friendly messages
□ Streaming responses for long generations
□ Health check endpoints
□ Rollback plan documented
□ On-call rotation assigned
□ Evaluation pipeline in CI
```

## Post-Launch Monitoring

| Timeframe | Actions |
|-----------|---------|
| Hour 1 | Watch error rates, latency, first user feedback |
| Day 1 | Review cost tracking, sample responses for quality |
| Week 1 | Analyze usage patterns, update rate limits |
| Month 1 | Add production failures to eval set, optimize prompts |
| Ongoing | Monthly cost review, quarterly prompt optimization |

## Quick Fixes for Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| 429 errors | Rate limited | Add backoff, caching, or increase limits |
| High latency (>5s) | Model/context size | Stream, use smaller model, trim context |
| Inconsistent outputs | Temperature > 0 | Set temperature=0 for deterministic tasks |
| Hallucinations | No grounding | Add RAG, fact-checking, require citations |
| Cost spikes | No limits | Implement per-user budgets, caching |
| Context errors | Token overflow | Count tokens, implement sliding window |

## References

- [OpenAI Production Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [LangSmith Production Monitoring](https://docs.smith.langchain.com/)
- [Helicone LLM Observability](https://www.helicone.ai/)
