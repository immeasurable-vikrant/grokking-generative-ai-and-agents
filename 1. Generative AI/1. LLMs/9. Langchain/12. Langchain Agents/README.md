# LangChain AI Agents 

### 1. Why Do We Need AI Agents? (Real-world Use Case)
Traditional way of planning a trip (Delhi → Goa, 1–7 May):
- Book flights/trains → multiple websites, forms, comparisons
- Book hotels → again forms, reviews, filters
- Make itinerary → search attractions, book cabs, entry tickets
→ Very time-consuming, multiple decisions, research, payments, calls/emails  
→ Not everyone (especially 60+ year olds) can do this easily.

AI Agent solution:
- You just say (in natural language):  
  “Create a budget travel itinerary from Delhi to Goa from 1st to 7th May.”
- The agent autonomously:
  1. Understands the high-level goal
  2. Breaks it into sub-tasks
  3. Uses tools/APIs (IRCTC, flight APIs, hotel APIs, etc.)
  4. Asks for preferences step-by-step
  5. Books everything, sends invoices, adds events to calendar, sets reminders
→ Seamless, conversational, human-like experience.

Core Problem AI Agents Solve:
Existing websites/apps have rigid forms → not natural human interaction.  
AI Agents make interaction natural and autonomous.

### 2. Technical Definition of an AI Agent
An AI Agent is an intelligent system that:
- Receives a high-level goal from the user
- Autonomously plans, decides, and executes a sequence of actions
- Uses external tools, APIs, and knowledge sources
- Maintains context, reasons over multiple steps, adapts to new information, optimizes for the outcome.

Simple words:
You give a goal → Agent figures out HOW to achieve it by itself using tools.

### 3. LLM vs AI Agent

| Feature                  | Normal LLM (e.g. ChatGPT)          | AI Agent (LLM + Tools)                     |
|--------------------------|-------------------------------------|--------------------------------------------|
| Reasoning                | Yes                                 | Yes (LLM as reasoning engine)              |
| Can take real actions?   | No (only text output)               | Yes (via Tools/APIs)                       |
| Access to external data | Only training data + prompt         | Real-time search, APIs, databases, etc.    |
| Memory/Context           | Stateless (or short)                | Long-running memory of previous steps      |
| Autonomy                 | None                                | High (plans & executes multi-step tasks)   |

→ AI Agent = LLM (brain) + Tools (hands & legs)

### 4. Key Characteristics of AI Agents
1. Goal-driven – you only give the goal, not the steps
2. Can plan & break down problems autonomously
3. Tool-aware – knows which tools exist and when to use them
4. Maintains context/memory across many steps
5. Adaptive – can re-plan if something changes or fails

### 5. First Practical Agent in LangChain (ReAct Agent)

#### Goal of Demo Agent
- Very simple agent with ONE tool: DuckDuckGo Search
- If user asks something that needs up-to-date info → agent searches the web and answers.

#### Required Installations
```bash
pip install langchain langchain-openai langchain-community duckduckgo-search
```

#### Complete Working Code (explained line-by-line)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # your key

from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# 1. Tool: DuckDuckGo Search
search_tool = DuckDuckGoSearchRun()
search_tool.name = "search_tool"   # optional, but good practice

# Test the tool
print(search_tool.run("Top news in India today"))

# 2. LLM (reasoning engine)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Test LLM
print(llm.invoke("Hi").content)

# 3. Pull the famous ReAct prompt by Harrison Chase (creator of LangChain)
react_prompt = hub.pull("hwchase17/react")
# This prompt forces the model to follow Thought → Action → Observation loop

# 4. Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=react_prompt
)

# 5. Create Agent Executor (the orchestrator that runs the loop)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,          # IMPORTANT: shows the full Thought-Action-Observation trace
    handle_parsing_errors=True
)

# 6. Run queries
response = agent_executor.invoke({
    "input": "What are three ways to reach Goa from Delhi?"
})
print(response["output"])

response = agent_executor.invoke({
    "input": "Give a five-pointer preview for today's IPL match between CSK and PBKS"
})
print(response["output"])
```

You will see the famous trace:
```
Thought: I should search...
Action: search_tool
Action Input: ways to reach Goa from Delhi
Observation: Flight, Train, Road...
...
Final Answer: The three common ways are...
```

### 6. What is ReAct? (Reasoning + Acting)

Paper (2022): “ReAct: Synergizing Reasoning and Acting in Language Models”

Core Idea:
Instead of single-turn LLM call → run a loop of:
1. Thought → reasoning step
2. Action → call a tool
3. Observation → result from tool
→ Repeat until Final Answer

Advantages:
- Handles multi-step problems beautifully
- Transparent reasoning (you see every thought)
- Works great when tools are required

### 7. How ReAct is Implemented Internally in LangChain

Two main objects:
1. Agent → the “brain” (LLM + ReAct prompt)
   - Takes user query + scratchpad (past trace)
   - Outputs either an Action or Final Answer
2. AgentExecutor → the “orchestrator”
   - Runs the Thought-Action-Observation loop
   - Calls tools
   - Updates scratchpad (memory)

The famous ReAct prompt (hwchase17/react) forces the model to output in this exact format:
```
Thought: ...
Action: tool_name
Action Input: ...
Observation: ...
(loops n times)
Thought: I now know the final answer
Final Answer: ...
```

### 8. Upgraded Agent – Adding a Custom Weather Tool

```python
from langchain.tools import tool
import requests

@tool
def get_weather_data(city: str) -> str:
    """Returns current weather for given city"""
    API_KEY = "your_weatherstack_key"
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url).json()
    return str(response)

# Add both tools
tools = [search_tool, get_weather_data]

# Recreate agent & executor with both tools
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Test multi-tool query
agent_executor.invoke({
    "input": "Find the capital of Madhya Pradesh, then find its current weather condition."
})
```

You will see:
1. First uses DuckDuckGo → finds Bhopal
2. Then uses get_weather_data → returns weather JSON
3. Finally gives clean answer

### 9. Important Note from Instructor (Future of Agents in LangChain)

Warning: The method shown (create_react_agent + AgentExecutor) is now considered legacy for production/industry-grade agents.

LangChain officially says:
> “If you want truly scalable, reliable, production-ready agents → move to LangGraph”

LangGraph = stateful, graph-based workflow engine (successor of LangChain agents)

So this video gives you:
- Strong conceptual foundation of agents
- Understanding of ReAct pattern
- Ability to build simple agents today
- Preparation for LangGraph (next-level agent building)

### Summary – Key Takeaways
- AI Agents = LLM (reasoning) + Tools (action) + Memory + Planning
- ReAct = most popular agent reasoning pattern (Thought → Action → Observation loop)
- LangChain lets you build ReAct agents very easily today
- For serious production agents → learn LangGraph (future videos will cover it)

You now fully understand what the current AI agent hype is about and how to build one from scratch!

Feel free to copy-paste the code and experiment.  
Next journey → LangGraph for scalable agents! 🚀