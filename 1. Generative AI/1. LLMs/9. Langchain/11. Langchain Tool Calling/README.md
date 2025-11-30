# LangChain Tool Calling 

### 1. Quick Recap of Previous Video (Why Tools Exist)
- LLMs are great at 2 things:
  1. Reasoning (understanding the question)
  2. Output generation (giving text answers using parametric knowledge)
- But LLMs cannot perform real actions (no hands & legs):
  - Cannot modify a database
  - Cannot post on LinkedIn/Twitter
  - Cannot call APIs (e.g., get current weather or currency rate)
- Solution → Tools
  - Tools are special Python functions that can interact with the external world.
  - Built-in tools: DuckDuckGo search, shell tool, etc.
  - Custom tools: you can create any tool you want.

### 2. Today’s Topic: Tool Calling (Main Concept)
Four major steps to make an LLM use tools properly:

| Step              | Name               | What Happens                                                                                     |
|-------------------|--------------------|--------------------------------------------------------------------------------------------------|
| 1                 | Tool Creation      | Create Python functions decorated with `@tool`                                                   |
| 2                 | Tool Binding       | Register tools with the LLM so it knows they exist, what they do, and input schema               |
| 3                 | Tool Calling       | LLM decides it needs a tool → returns a structured tool call (name + arguments) – does NOT run it |
| 4                 | Tool Execution     | Programmer (you) actually runs the tool and gets the result                                      |

Important: The LLM NEVER executes the tool itself. It only suggests which tool + arguments. Execution is always handled by LangChain/you → this is for safety.

### 3. Step 1 – Tool Creation (Simple Example)
```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Given two numbers a and b, this tool returns their product."""
    return a * b

# Test
print(multiply.invoke({"a": 3, "b": 4}))        # → 12
print(multiply.name)        # → "multiply"
print(multiply.description) # → description text
print(multiply.args)        # → input schema
```

### 4. Step 2 – Tool Binding
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm_with_tools = llm.bind_tools([multiply])   # This is binding!
# Now the LLM knows about the "multiply" tool
```

Only some models support tool binding (gpt-4o, gpt-4o-mini, gpt-3.5-turbo-1106+, Claude-3 series, Gemini-1.5-pro, etc.)

### 5. Step 3 – Tool Calling (LLM only suggests)
```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Hi, how are you?")]
response = llm_with_tools.invoke(messages)
print(response.tool_calls)   # → []  (no tool needed)

messages = [HumanMessage(content="Can you multiply 3 with 10?")]
response = llm_with_tools.invoke(messages)
print(response.tool_calls)   # → list with one tool call
```

Output of tool_calls (example):
```python
[
  {
    'name': 'multiply',
    'args': {'a': 3, 'b': 10},
    'id': 'call_abc123',
    'type': 'tool_call'
  }
]
```

### 6. Step 4 – Tool Execution (We do it manually)
```python
tool_call = response.tool_calls[0]

# Option 1 – send only args
result = multiply.invoke(tool_call["args"])          # → 30

# Option 2 – send entire tool_call object (recommended)
tool_message = multiply.invoke(tool_call)   # returns a ToolMessage
```

ToolMessage contains:
- content → result (30)
- tool_call_id → same id as in tool call

### 7. Full Flow – Conversation with Memory (Important!)
```python
from langchain_core.messages import HumanMessage, AIMessage

messages = []
query = "What is 3 * 1000?"
messages.append(HumanMessage(content=query))

# 1. LLM decides to use tool
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)                     # add AI message (contains tool_calls)

# 2. Execute tool
tool_call = ai_msg.tool_calls[0]
tool_msg = multiply.invoke(tool_call)
messages.append(tool_msg)                   # add ToolMessage

# 3. Send everything back to LLM for final answer
final_response = llm_with_tools.invoke(messages)
print(final_response.content)
# → "The product of 3 and 1000 is 3000."
```

### 8. Real-World Application: Real-Time Currency Converter
Problem: LLMs have outdated currency rates → we give them real-time power.

Two tools needed:
1. Get current conversion factor (API call)
2. Multiply amount × factor

#### Tool 1 – Get Conversion Factor
```python
import requests
from langchain_core.tools import tool

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """This function fetches the currency conversion factor between a given base currency and a target currency."""
    url = f"https://api.exchangerate.host/convert?from={base_currency}&to={target_currency}"
    response = requests.get(url)
    data = response.json()
    return data["result"]   # actual conversion rate
```

#### Tool 2 – Convert Amount (with Injected Tool Argument!)
```python
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg   # important import

@tool
def convert(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg],   # ← LLM will NOT fill this
) -> float:
    """Given a currency conversion rate, this function calculates the target currency value from a given base currency value."""
    return base_currency_value * conversion_rate
```

#### Binding both tools
```python
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])
```

#### Full Execution Flow (Handles sequential tool calls properly)
```python
import json
from langchain_core.messages import HumanMessage

messages = []
human_query = "What is the conversion factor between USD and INR and based on that can you convert 10 USD to INR?"
messages.append(HumanMessage(content=human_query))

# 1. First LLM call → gets two tool calls
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

conversion_rate = None

# 2. Loop through tool calls and execute in correct order
for tool_call in ai_msg.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        tool_msg = get_conversion_factor.invoke(tool_call)
        messages.append(tool_msg)
        # Extract rate
        data = json.loads(tool_msg.content)
        conversion_rate = data["result"]

    elif tool_call["name"] == "convert":
        # Inject the missing argument
        args = tool_call["args"]
        args["conversion_rate"] = conversion_rate   # ← injection!
        tool_msg = convert.invoke(tool_call)        # pass original tool_call (args now updated)
        messages.append(tool_msg)

# 3. Final LLM call with full context
final_answer = llm_with_tools.invoke(messages)
print(final_answer.content)
# Output example:
# The conversion factor between USD and INR is approximately 85.34.
# Converting 10 USD to INR yields ≈ 853.4 INR.
```

### Key Takeaways
- LLM never runs tools → only suggests.
- Use InjectedToolArg when one tool’s output is needed as input for another tool → prevents LLM from hallucinating old rates.
- This entire flow (Human → AI → Tool → Tool → AI) is the foundation of real AI agents.
- What we built is NOT a true agent yet because we (the programmer) controlled the loop.
- True agents (next video) will do all of this autonomously using LangChain AgentExecutor / create_react_agent.

### Next Video Teaser
Teacher will show how to convert this manual loop into a fully autonomous AI Agent using LangChain agents (ReAct pattern).

Hope these notes are super clear and complete!  
Feel free to copy-paste the code directly – everything works in Google Colab.