---
layout: default
title: Advanced Agent Architectures
parent: 08 - Advanced Topics
nav_order: 4
---

# Advanced Agent Architectures

## TL;DR

Simple ReAct agents hit limits with complex tasks. Advanced patterns include: graph-based workflows (LangGraph), multi-agent systems (AutoGen, CrewAI), hierarchical planning, and reflection. Understanding these patterns enables building robust autonomous systems.

## Limitations of Simple Agents

Basic ReAct loop problems:

| Problem | Symptom | Cause |
|---------|---------|-------|
| Loops | Agent repeats same action | No state tracking |
| Context overflow | Forgets early info | Unbounded history |
| No parallelism | Sequential execution | Single thread |
| Brittle | Fails on edge cases | No error recovery |
| No planning | Wanders aimlessly | Reactive, not proactive |

## Agent Architectures

### 1. Graph-Based Workflows (LangGraph)

Define agent behavior as a state machine:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    current_step: str
    results: dict

def research(state: AgentState) -> AgentState:
    """Research step"""
    # Do research
    state["results"]["research"] = search(state["messages"][-1])
    return state

def analyze(state: AgentState) -> AgentState:
    """Analysis step"""
    state["results"]["analysis"] = llm.analyze(state["results"]["research"])
    return state

def should_continue(state: AgentState) -> str:
    """Decide next step"""
    if needs_more_research(state):
        return "research"
    elif needs_analysis(state):
        return "analyze"
    return "end"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("research", research)
workflow.add_node("analyze", analyze)
workflow.add_conditional_edges(
    "research",
    should_continue,
    {"research": "research", "analyze": "analyze", "end": END}
)
workflow.add_edge("analyze", END)
workflow.set_entry_point("research")

app = workflow.compile()
result = app.invoke({"messages": ["Research quantum computing"], "results": {}})
```

#### When to Use LangGraph

- Complex workflows with branches
- Need for explicit state management
- Cycles/loops with termination conditions
- Human-in-the-loop patterns

### 2. Multi-Agent Systems

Multiple specialized agents collaborating:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Request                                                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐                                               │
│   │ Orchestrator│ ← Routes to appropriate agents                │
│   └─────────────┘                                               │
│        │                                                         │
│   ┌────┴────┬────────┬────────┐                                 │
│   ▼         ▼        ▼        ▼                                 │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                │
│ │Code │ │Data │ │Write│ │Review│                               │
│ │Agent│ │Agent│ │Agent│ │Agent│                                │
│ └─────┘ └─────┘ └─────┘ └─────┘                                │
│   │         │        │        │                                 │
│   └─────────┴────────┴────────┘                                 │
│                 │                                                │
│                 ▼                                                │
│   ┌─────────────────────────────┐                               │
│   │    Aggregated Response       │                               │
│   └─────────────────────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### CrewAI Example

```python
from crewai import Agent, Task, Crew

# Define agents with roles
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive information on the given topic",
    backstory="Expert at finding and synthesizing information",
    tools=[search_tool, web_scraper],
    llm=llm
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, engaging technical content",
    backstory="Experienced at explaining complex topics simply",
    llm=llm
)

reviewer = Agent(
    role="Editor",
    goal="Ensure content is accurate and well-written",
    backstory="Detail-oriented with high standards",
    llm=llm
)

# Define tasks
research_task = Task(
    description="Research {topic} thoroughly",
    agent=researcher,
    expected_output="Comprehensive research notes"
)

write_task = Task(
    description="Write an article based on the research",
    agent=writer,
    expected_output="Draft article",
    context=[research_task]  # Depends on research
)

review_task = Task(
    description="Review and improve the article",
    agent=reviewer,
    expected_output="Final polished article",
    context=[write_task]
)

# Create crew and run
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, write_task, review_task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Quantum Computing"})
```

### 3. Hierarchical Planning

High-level planner → Low-level executors:

```python
class HierarchicalAgent:
    def __init__(self):
        self.planner = PlannerLLM()  # GPT-4 for planning
        self.executor = ExecutorLLM()  # GPT-3.5 for steps
    
    def run(self, goal: str) -> str:
        # High-level planning
        plan = self.planner.create_plan(goal)
        # Plan: ["1. Research topic", "2. Gather data", "3. Write report"]
        
        results = []
        for step in plan.steps:
            # Low-level execution
            result = self.executor.execute_step(step, context=results)
            results.append(result)
            
            # Check if replanning needed
            if self.needs_replan(results, plan):
                plan = self.planner.replan(goal, results)
        
        return self.synthesize(results)
```

### 4. Reflection & Self-Critique

Agent evaluates and improves its own outputs:

```python
class ReflectiveAgent:
    def generate_with_reflection(self, task: str, max_iterations: int = 3) -> str:
        response = self.llm.generate(task)
        
        for i in range(max_iterations):
            # Self-critique
            critique = self.llm.generate(f"""
            Task: {task}
            Response: {response}
            
            Critique this response:
            1. What's good about it?
            2. What's missing or wrong?
            3. How can it be improved?
            """)
            
            # Check if good enough
            if "no improvements needed" in critique.lower():
                break
            
            # Improve based on critique
            response = self.llm.generate(f"""
            Task: {task}
            Previous response: {response}
            Critique: {critique}
            
            Generate an improved response addressing the critique.
            """)
        
        return response
```

### 5. Memory & Learning

Persistent memory across sessions:

```python
class MemoryAgent:
    def __init__(self, vector_store):
        self.short_term = []  # Current conversation
        self.long_term = vector_store  # Persistent memories
    
    def process(self, user_input: str) -> str:
        # Retrieve relevant memories
        memories = self.long_term.search(user_input, k=5)
        
        # Build context
        context = f"""
        Relevant past interactions:
        {format_memories(memories)}
        
        Current conversation:
        {format_messages(self.short_term)}
        """
        
        # Generate response
        response = self.llm.generate(context + user_input)
        
        # Update memories
        self.short_term.append({"user": user_input, "assistant": response})
        
        # Store important information
        if self.should_remember(response):
            self.long_term.add(self.extract_facts(user_input, response))
        
        return response
```

## Design Patterns

### Pattern: Supervisor

One agent coordinates others:

```python
class Supervisor:
    def __init__(self, workers: List[Agent]):
        self.workers = {w.name: w for w in workers}
    
    def delegate(self, task: str) -> str:
        # Decide which worker
        assignment = self.llm.generate(f"""
        Task: {task}
        Available workers: {list(self.workers.keys())}
        Which worker should handle this?
        """)
        
        worker = self.workers[assignment.strip()]
        return worker.execute(task)
```

### Pattern: Debate

Agents argue to find best answer:

```python
def debate(question: str, rounds: int = 3) -> str:
    agent_a = Agent("Proponent")
    agent_b = Agent("Opponent")
    
    arguments = []
    for round in range(rounds):
        # Agent A argues
        a_arg = agent_a.argue(question, history=arguments)
        arguments.append({"agent": "A", "argument": a_arg})
        
        # Agent B counters
        b_arg = agent_b.counter(question, history=arguments)
        arguments.append({"agent": "B", "argument": b_arg})
    
    # Judge decides
    return judge.decide(question, arguments)
```

### Pattern: Ensemble

Multiple agents vote:

```python
def ensemble_decide(question: str, agents: List[Agent]) -> str:
    responses = [agent.respond(question) for agent in agents]
    
    # Simple majority voting
    return most_common(responses)
    
    # Or LLM synthesis
    return synthesizer.combine(responses)
```

## Error Handling

```python
class RobustAgent:
    def execute_with_retry(self, task: str, max_retries: int = 3) -> str:
        errors = []
        
        for attempt in range(max_retries):
            try:
                result = self.execute(task)
                
                # Validate result
                if self.validate(result):
                    return result
                else:
                    errors.append(f"Validation failed: {result}")
                    
            except ToolError as e:
                errors.append(f"Tool error: {e}")
                # Try alternative tool
                result = self.execute_with_fallback_tools(task)
                
            except Exception as e:
                errors.append(f"Error: {e}")
        
        # Final fallback
        return self.graceful_degradation(task, errors)
```

## Key Takeaways

- Simple ReAct loops aren't enough for complex tasks
- LangGraph enables explicit state machines for control flow
- Multi-agent systems divide work among specialists
- Hierarchical planning separates strategy from tactics
- Reflection improves output quality through self-critique
- Always include error handling and fallbacks

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://docs.crewai.com/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [Reflexion Paper](https://arxiv.org/abs/2303.11366)
