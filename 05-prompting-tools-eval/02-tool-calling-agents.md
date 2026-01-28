---
layout: default
title: Tool Calling & Agents
parent: 05 - Prompting & Eval
nav_order: 2
---

# Tool Calling & Agents

## TL;DR

Tool calling lets LLMs execute functions—search the web, run code, query databases. Agents are LLMs in a loop: think → act → observe → repeat. This is how you build systems that do things, not just generate text.

## Function/Tool Calling

### The Concept

```
User: "What's the weather in Tokyo?"

Without tools:
  LLM → "I don't have real-time weather data" (or hallucinates)

With tools:
  LLM → "I should call get_weather('Tokyo')"
  System → Calls API, gets result
  LLM → "It's currently 22°C and sunny in Tokyo"
```

### OpenAI Format

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'Tokyo'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools
)

# Response contains tool_calls if model wants to use a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Execute the function
    result = get_weather(json.loads(tool_call.function.arguments))
    # Send result back to model
```

### The Full Loop

```python
def chat_with_tools(user_message: str):
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        # If no tool calls, we're done
        if not assistant_message.tool_calls:
            return assistant_message.content
        
        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            result = execute_tool(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
```

## Agent Architecture

Agents = LLM + Tools + Loop

### ReAct Pattern (Reasoning + Acting)

```
Thought: I need to find the current stock price of Apple
Action: search("AAPL stock price")
Observation: AAPL is trading at $178.50

Thought: Now I need to calculate the market cap
Action: calculate("178.50 * 15.8 billion shares")
Observation: Market cap is approximately $2.82 trillion

Thought: I have the information needed
Action: respond("Apple's current stock price is $178.50, 
                 with a market cap of approximately $2.82T")
```

### Implementation

```python
class Agent:
    def __init__(self, tools: list, llm):
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.max_iterations = 10
    
    def run(self, task: str) -> str:
        messages = [{"role": "user", "content": task}]
        
        for i in range(self.max_iterations):
            response = self.llm.complete(messages, tools=self.tools)
            
            if response.is_final_answer:
                return response.content
            
            # Execute tool
            tool_name = response.tool_name
            tool_args = response.tool_args
            
            try:
                result = self.tools[tool_name].execute(**tool_args)
                messages.append({
                    "role": "tool",
                    "content": f"Result: {result}"
                })
            except Exception as e:
                messages.append({
                    "role": "tool",
                    "content": f"Error: {str(e)}"
                })
        
        return "Max iterations reached"
```

## Common Tool Types

| Category | Examples |
|----------|----------|
| Information | Web search, Wikipedia, documentation lookup |
| Computation | Calculator, code execution, data analysis |
| APIs | Weather, stocks, databases, internal services |
| Actions | Send email, create ticket, deploy code |
| File System | Read/write files, list directories |

## Tool Design Best Practices

### 1. Clear Names and Descriptions

```python
# ❌ Bad
{"name": "do_thing", "description": "Does something"}

# ✓ Good  
{
    "name": "search_knowledge_base",
    "description": "Search the internal knowledge base for articles. "
                   "Use when user asks about company policies, procedures, "
                   "or internal documentation. Returns top 5 relevant articles."
}
```

### 2. Explicit Parameters

```python
{
    "name": "create_calendar_event",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title"},
            "start_time": {"type": "string", "description": "ISO 8601 format"},
            "duration_minutes": {"type": "integer", "default": 60},
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Email addresses of attendees"
            }
        },
        "required": ["title", "start_time"]
    }
}
```

### 3. Useful Error Messages

```python
def execute_tool(name: str, args: dict) -> str:
    try:
        result = tools[name](**args)
        return json.dumps({"success": True, "data": result})
    except PermissionError:
        return json.dumps({
            "success": False, 
            "error": "Permission denied. User lacks access to this resource."
        })
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}. Please check the parameters."
        })
```

## Agent Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Infinite loops | Can't decide to stop | Max iterations, timeout |
| Wrong tool choice | Ambiguous descriptions | Better tool descriptions |
| Hallucinated tools | Model invents tools | Validate tool names |
| Stuck on errors | Keeps retrying same thing | Error counting, bailout |
| Context overflow | Too many iterations | Summarize history |

## Multi-Agent Patterns

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR PATTERN                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Request                                               │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────┐                                           │
│   │ Orchestrator│ ← Decides which specialist to use         │
│   └─────────────┘                                           │
│        │                                                     │
│   ┌────┼────┬────────┐                                      │
│   ▼    ▼    ▼        ▼                                      │
│ [Code] [Data] [Search] [Writing]  ← Specialist agents       │
│   │    │      │        │                                    │
│   └────┴──────┴────────┘                                    │
│        │                                                     │
│        ▼                                                     │
│   Final Response                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Takeaways

- Tool calling extends LLMs from generators to actors
- The agent loop: think → act → observe → repeat
- Good tool descriptions are critical for reliability
- Always have escape hatches (max iterations, timeouts)
- Start simple, add complexity as needed

## References

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
