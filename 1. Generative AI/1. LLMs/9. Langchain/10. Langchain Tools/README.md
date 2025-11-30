# LangChain : Tools in LangChain


### Playlist Recap (What was covered so far)
- First 15 videos divided into 2 parts:
  1. LangChain Fundamentals (Models, Prompts, Chains, etc.)
  2. Building RAG systems (Document Loaders → Text Splitters → Vector Stores → Retrievers → RAG pipeline)
- From this video → New Part 3: Building AI Agents using LangChain
- Topics in Agent segment (next 3-4 videos):
  1. Tools ← Today’s topic
  2. Tool Calling (how LLM decides when to use tools)
  3. Agents (full agent creation using LangChain/LangGraph)

### Core Concept: Why Do We Need Tools?

LLMs have 2 superpowers:
1. Reasoning (think step-by-step)
2. Language Generation (speak/write answers)

But LLMs have NO real “hands and legs” → they cannot:
- Book train/flight tickets
- Fetch live weather data
- Run Python code reliably (especially complex math)
- Call external APIs
- Post tweets
- Interact with databases
- Execute any real-world action

→ Tools = Give LLMs “hands and legs”

**Definition of a Tool**  
A Tool is just a Python function that performs a real-world task, packaged in a way that an LLM can understand and call it when needed.

Example analogy:  
Human body = LLM (can think + speak)  
Tools = Hands + Legs (can take action)

### Types of Tools in LangChain
1. Built-in Tools (pre-made by LangChain team)
2. Custom Tools (you create yourself)

### 1. Built-in Tools (Examples shown in video)

| Tool Name                | Purpose                                                                 | Import Path                                      |
|--------------------------|-------------------------------------------------------------------------|--------------------------------------------------|
| DuckDuckGo Search        | Real-time web search (great for current events)                         | `from langchain_community.tools import DuckDuckGoSearchRun` |
| Wikipedia                | Search Wikipedia and get summarized answer                              |                                                  |
| Python REPL              | Execute raw Python code (great for reliable math)                       |                                                  |
| Shell Tool               | Run shell/command-line commands on the host machine                    | `from langchain_community.tools import ShellTool` |
| Requests (GET/POST)      | Make HTTP requests                                                      |                                                  |
| Gmail, Slack, SQL, etc.  | Direct integration with those services                                  |                                                  |

Full list: https://python.langchain.com/docs/integrations/tools/

**Live Demo – DuckDuckGo Search**
```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
result = search.invoke("Latest IPL news")
print(result)
# Output: Real-time IPL headlines (Virat Kohli leading run-scorer, etc.)
```

**Live Demo – Shell Tool**
```python
from langchain_experimental.tools import ShellTool

shell_tool = ShellTool()
print(shell_tool.invoke("whoami"))   # → root (in Colab)
print(shell_tool.invoke("ls"))      # lists files in current directory
```
Warning: ShellTool is powerful but dangerous in production (can delete files!).

### 2. Custom Tools – How to Create Your Own

#### Method 1: Simplest & Most Common → @tool decorator (Recommended for 90% cases)

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the product."""
    return a * b

# Use it
print(multiply.invoke({"a": 3, "b": 5}))   # → 15

# Tool metadata (LLM sees this!)
print(multiply.name)         # "multiply"
print(multiply.description)  # "Multiplies two integers..."
print(multiply.args)         # {'a': {'type': 'int'}, 'b': {'type': 'int'}}
```

Key points:
- Docstring → becomes tool description
- Type hints → become argument schema
- @tool decorator → converts function into a real LangChain tool

#### Method 2: StructuredTool + Pydantic (More strict, production-ready)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

def multiply_func(a: int, b: int) -> int:
    return a * b

class MultiplyInput(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="Multiplier",
    description="Multiplies two numbers",
    args_schema=MultiplyInput
)

print(multiply_tool.invoke({"a": 4, "b": 7}))  # → 28
```

#### Method 3: Inherit from BaseTool class (Maximum customization, supports async)

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiplies two integers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

    # Optional: async version
    async def _arun(self, a: int, b: int) -> int:
        return a * b

tool = MultiplyTool()
print(tool.invoke({"a": 6, "b": 8}))  # → 48
```

### What the LLM Actually Sees (Tool Schema)

When you pass a tool to an LLM, LangChain sends the JSON schema, not the Python function:

```python
print(multiply.get_input_schema().json(indent=2))
```

Output (example):
```json
{
  "title": "multiply",
  "description": "Multiplies two integers and returns the product.",
  "type": "object",
  "properties": {
    "a": {"type": "integer"},
    "b": {"type": "integer"}
  },
  "required": ["a", "b"]
}
```

### Toolkits – Grouping Related Tools

When you have multiple related tools → group them into a Toolkit for reusability.

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b

class MathToolkit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for t in tools:
    print(t.name, "→", t.description)
```

### Relationship Between Tools & Agents

**AI Agent = LLM + Tools + Reasoning Loop**

Definition of an Agent:  
"An AI Agent is an LLM-powered system that can autonomously think, decide, and take actions using external tools and APIs to achieve a goal."

- Reasoning + Decision making → comes from LLM
- Taking real actions → comes from Tools

That’s why Tools are the foundation of any real agent.

### Summary of This Video

- Tools give LLMs the ability to act in the real world
- Two types: Built-in & Custom
- 3 main ways to create custom tools (@tool, StructuredTool, BaseTool)
- Tools have name, description, args schema → LLM reads this
- Toolkits = collections of related tools
- Next video → Tool Calling (how LLM actually decides to call tools)

### What’s Coming Next
- Tool Calling (binding tools to LLM)
- Creating full ReAct agents in LangChain
- LangGraph agents (stateful, multi-actor workflows)

Done! These are complete, structured, and detailed English notes with all code examples preserved exactly as shown in the video. Save this for your LangChain playlist revision! 🚀