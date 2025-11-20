# Runnables in LangChain Explained 

**Generative AI using LangChain Series | November 20, 2025**

This is **the most technical and deepest video** in the entire playlist so far.  

This video is split into two parts mentally:
- **Part 1 (This video)** → The full history, philosophy, problems, and **scratch-from-zero code** to understand Runnables

- **Part 2 (Next video)** → Real `Runnable` classes, LCEL deep dive, advanced features

---

### The Full LangChain Journey (Why Runnables Were Born)

| Year / Phase | What Happened | Problem |
|--------------|---------------|--------|
| Nov 2022     | ChatGPT + OpenAI APIs released | Everyone wanted to build LLM apps |
| 2023         | LangChain created | Goal: Make building LLM apps easy |
| Early LangChain | Many components: LLM, PromptTemplate, Document Loaders, Text Splitters, Embeddings, VectorStores, Retrievers, Output Parsers | Great! But components had **different interfaces** (`predict`, `format`, `get_relevant_documents`, `parse`) |
| Mid 2023     | Introduced **Chains** (LLMChain, RetrievalQAChain, SequentialChain, APIChain, etc.) | Solved repetition → but created **too many chains** (50+) |
| Late 2023    | **Big Problem** | • Huge codebase to maintain<br>• Steep learning curve (“Which chain for which use case?”)<br>• Not flexible enough |

**The Core Issue**:  
Components were **not standardized** → To connect them, LangChain had to write **custom glue code** (chains) for every common pattern.

This is like having Lego blocks with **different shaped connectors** → you need custom adapters for everything.

---

### The Solution: **Runnables** – The Biggest Redesign in LangChain History

Runnables are the **standardized Lego blocks** of modern LangChain (v0.1+).

#### 4 Core Principles of Runnables (Exactly Like Real Lego Blocks)

| # | Runnable Principle | Lego Analogy |
|---|---------------------|------------|
| 1 | Each Runnable is a **unit of work** (does one thing) | One Lego piece has one shape/purpose |
| 2 | All Runnables follow a **common interface** (`invoke`, `batch`, `stream`) | Every Lego has studs on top & tubes below |
| 3 | Runnables can be **composed/chained** together | You can connect any two Legos |
| 4 | A chain of Runnables is **itself a Runnable** | A built structure can connect to another |

This composition property is **pure magic** – it lets you build infinitely complex workflows.

---

### From Scratch: Building Runnables (Dummy Version)

Nitish Sir coded everything live from zero to prove the concept.

#### Step 1: Old Way (Non-Standard Components)

```python
class NakliLLM:
    def predict(self, prompt): ...

class NakliPromptTemplate:
    def format(self, **kwargs): ...

# Manual connection (painful)
prompt = template.format(topic="India")
response = llm.predict(prompt)
```

#### Step 2: The Problem with Chains

```python
class NakliLLMChain:
    def run(self, input_data):
        prompt = self.prompt.format(**input_data)
        return self.llm.predict(prompt)
```

Works, but **not flexible** → can't easily chain multiple steps or multiple chains.

#### Step 3: The Runnable Revolution

```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):  # Same for ALL runnables
        pass
```

Now make every component inherit from `Runnable` and implement `invoke`:

```python
class NakliLLM(Runnable):
    def invoke(self, prompt):  # Same name!
        # same as old predict()
        ...

class NakliPromptTemplate(Runnable):
    def invoke(self, input_dict):  # Same name!
        return self.template.format(**input_dict)
```

#### Step 4: The Magic Connector – RunnableSequence

```python
class RunnableConnector(Runnable):
    def __init__(self, runnables):
        self.runnables = runnables  # list of any runnables

    def invoke(self, input_data):
        data = input_data
        for runnable in self.runnables:
            data = runnable.invoke(data)  # output → next input!
        return data
```

Now you can do this:

```python
chain = RunnableConnector([prompt_template, llm, parser])
result = chain.invoke({"topic": "AI"})
```

And even chain chains:

```python
joke_chain = RunnableConnector([prompt1, llm])
explain_chain = RunnableConnector([prompt2, llm, parser])

final_chain = RunnableConnector([joke_chain, explain_chain])
final_chain.invoke({"topic": "cricket"})  # → joke → explanation
```

**This is exactly how real LangChain works today!**

---

### Proof: Real LangChain Source Code

Nitish Sir opened actual LangChain code:

```
ChatOpenAI 
→ BaseChatOpenAI 
→ BaseChatModel 
→ BaseLanguageModel 
→ Runnable 
→ has abstract invoke()
```

Every component in modern LangChain inherits from `Runnable` and implements `invoke`.

---

### Key Takeaways – Runnables in One Table

| Feature                  | Old LangChain (Chains)         | New LangChain (Runnables + LCEL)                  |
|--------------------------|--------------------------------|----------------------------------------------------|
| Interface                | Different (`predict`, `format`) | Unified (`invoke`, `stream`, `batch`)            |
| Composition              | Needed custom chains          | Just use `|` (pipe) operator                     |
| Flexibility              | Low (50+ rigid chains)         | Infinite (compose anything)                       |
| Codebase                 | Heavy                          | Light & maintainable                               |
| Learning Curve           | Steep                          | Gentle (just learn `|` and Runnables)             |
| Syntax                   | `LLMChain(llm=..., prompt=...)`| `prompt | llm | parser`                           |

**LCEL (LangChain Expression Language)** = the beautiful `|` syntax we used in Video 7  
It only became possible **because of Runnables**

---

### Why This Video Changes Everything

After this video:
- You will **never be confused** by LangChain again
- You will understand **why** `|` works
- You will see chains as just **composed Runnables**
- You will be ready for **Agents, Tools, Memory, Streaming** – all built on Runnables

> “Agar aap ye video samajh gaye, toh LangChain ke saath aapki dosti pakki ho gayi.” – Nitish Sir

---

### What’s Next (Video 9 – Part 2)

- Real `Runnable` classes in LangChain
- `RunnableLambda`, `RunnableParallel`, `RunnableBranch`
- Advanced LCEL: `.stream()`, `.batch()`, `.with_config()`
- How memory, tools, and agents are just Runnables
- Debugging with `.get_graph().print_ascii()`

**Don’t skip the next video** – it will make this one 10× more valuable.

---

**Like, Subscribe, and Share** – this is the video that turns you from LangChain user → **LangChain master**! 🚀

See you in the next one – the **real Runnable deep dive**!