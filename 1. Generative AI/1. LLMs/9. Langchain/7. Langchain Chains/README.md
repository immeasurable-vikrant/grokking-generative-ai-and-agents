# Langchain Chains
**Generative AI using LangChain**


### Why Do We Need Chains?

Until now we were manually doing everything:

```python
prompt = prompt_template.invoke({"topic": "cricket"})
response = llm.invoke(prompt)
output = response.content   # manual parsing
print(output)
```

This becomes extremely painful when the application grows.  
Real LLM apps consist of **many small steps** that need to run in sequence, parallel, or conditionally.

**Chains = Declarative way to connect components into automated pipelines**  
- First step’s output → automatically becomes second step’s input  
- You only provide input to the very first step → entire pipeline runs automatically  
- No manual invoke calls, no manual parsing

---

### 1. Simple Sequential Chain (Your First Chain)

Goal: Take a topic → Generate 5 interesting facts

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Generate 5 interesting facts about {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser     # ← This pipe syntax is LCEL (LangChain Expression Language)

result = chain.invoke({"topic": "cricket"})
print(result)
```

**Key Benefits**:
- One line instead of 5–6 manual steps
- You can visualize the chain: `chain.get_graph().print_ascii()`

---

### 2. Longer Sequential Chain (Multiple LLM Calls)

Goal:  
1. Generate a detailed report on a topic  
2. Summarize that report into 5 key points

```python
prompt1 = PromptTemplate.from_template("Generate a detailed report on {topic}")
prompt2 = PromptTemplate.from_template("Given the text below, extract 5 key points:\n\n{text}")

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Unemployment in India"})
print(result)   # → 5-point summary
```

This shows how easily you can chain **multiple LLM calls** sequentially.

---

### 3. Parallel Chains (RunnableParallel)

Goal:  
Take a long document → Generate **Notes** and **Quiz** in parallel → Merge them

Used two different models:
- OpenAI for notes
- Anthropic Claude for quiz

```python
from langchain_core.runnables import RunnableParallel

prompt_notes = PromptTemplate.from_template("Generate concise notes from:\n\n{text}")
prompt_quiz  = PromptTemplate.from_template("Generate 5 Q&A from:\n\n{text}")
prompt_merge = PromptTemplate.from_template("Combine these notes and quiz:\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}")

# Parallel execution
parallel_chain = RunnableParallel({
    "notes": prompt_notes | model_openai | parser,
    "quiz" : prompt_quiz  | model_claude | parser
})

# Merge
merge_chain = parallel_chain | prompt_merge | model_openai | parser

final_chain = merge_chain
result = final_chain.invoke({"text": long_svm_document})
```

**Output**: Beautifully merged notes + quiz in one document  
**Power**: Multiple expensive LLM calls happen truly in parallel

---

### 4. Conditional Chains (RunnableBranch – If/Else Logic)

Goal:  
Customer feedback → Detect sentiment → Reply accordingly  
- Positive → "Thank you so much!"  
- Negative → "Sorry for the inconvenience…"

**Challenge**: LLM might return "The sentiment is negative" instead of just "negative" → breaks routing

**Solution**: Use **PydanticOutputParser** to force structured output

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

class FeedbackSentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback")

parser_structured = PydanticOutputParser(pydantic_object=FeedbackSentiment)

classifier_prompt = PromptTemplate.from_template(
    "What is the sentiment of this feedback? Only reply 'positive' or 'negative'.\n\n{feedback}\n{format_instructions}",
    partial_variables={"format_instructions": parser_structured.get_format_instructions()}
)

classifier_chain = classifier_prompt | model | parser_structured

# Branch prompts
positive_prompt = PromptTemplate.from_template("Write a warm thank you reply to this positive feedback:\n{feedback}")
negative_prompt = PromptTemplate.from_template("Write an apologetic reply to this negative feedback:\n{feedback}")

from langchain_core.runnables import RunnableBranch, RunnableLambda

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_prompt | model | parser),
    (lambda x: x.sentiment == "negative", negative_prompt | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment")  # default
)

final_chain = classifier_chain | branch_chain
```

**Result**:
- "This is a terrible phone" → Apologetic reply  
- "Wonderful phone!" → Thank you reply  
Only **one** branch executes (unlike parallel)

---

### Key Concepts Introduced (Will be explained deeply in next video)

| Concept                  | What it does                                                                 |
|--------------------------|-------------------------------------------------------------------------------|
| `|` pipe operator        | Connects components → part of **LCEL**                                        |
| `RunnableParallel`       | Runs multiple chains in parallel                                             |
| `RunnableBranch`         | If-else routing based on condition                                            |
| `RunnableLambda`         | Converts any lambda function into a runnable (so it can be used in chains)   |
| `chain.get_graph().print_ascii()` | Visualizes your entire pipeline (super useful for debugging)            |

---

### Why This Video is Crucial

After this video, you can build **any complex LLM application** by combining:
- Sequential chains (most common)
- Parallel chains (speed + multi-model)
- Conditional chains (smart routing)

Even **Agents** (coming later) heavily use these three patterns under the hood.

---

### What’s Coming in Next Video (Part 2)

- What are **Runnables**? (Everything in modern LangChain is a Runnable)
- Deep dive into **LangChain Expression Language (LCEL)**
- How `.invoke()`, `.stream()`, `.batch()` work uniformly
- Why the pipe `|` syntax is so powerful
- Custom runnables, fallbacks, memory attachment, etc.

**Advice from Nitish Sir**:  
> "You will appreciate today’s video 10× more after watching the next one. So don’t skip!"

---

**Like, Subscribe & See you in the next video (Runnables Deep Dive)!** 🚀