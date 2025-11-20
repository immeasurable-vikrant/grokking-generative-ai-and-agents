# Runnables in LangChain – Part 2 

**Generative AI using LangChain**

### Runnables = Two Categories

| Category                  | What it is                                      | Real-Life Examples                              |
|---------------------------|--------------------------------------------------|--------------------------------------------------|
| **Task-Specific Runnables** | Core LangChain components that do actual work   | `ChatOpenAI`, `PromptTemplate`, `Retriever`, `StrOutputParser` |
| **Runnable Primitives**    | “Glue” that connects task-specific runnables   | `RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch` |

Today’s entire video = **Deep dive into Runnable Primitives**

---

### 1. RunnableSequence → The `|` (pipe) operator

**Purpose**: Connect runnables **sequentially** (most used primitive)

```python
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence(prompt, model, parser)
# OR the beautiful LCEL way (recommended)
chain = prompt | model | parser
```

**Example**: Joke → Explain the joke

```python
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "cricket"})
# Output: Full explanation of the joke
```

**Key Insight**:  
`|` = **LangChain Expression Language (LCEL)** – the cleanest way to write sequential chains.

---

### 2. RunnableParallel → Parallel execution

**Purpose**: Run multiple chains **at the same time**, same input → dictionary output

```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    "tweet":    prompt_tweet  | model | parser,
    "linkedin": prompt_linkedin | model | parser
})

result = parallel_chain.invoke({"topic": "AI"})
print(result)
# {'tweet': 'Short catchy tweet...', 'linkedin': 'Professional post...'}
```

Real use: Generate tweet + LinkedIn post + Email version simultaneously.

---

### 3. RunnablePassthrough → “Do nothing, just forward”

**Purpose**: Pass data unchanged (super useful with Parallel)

**Classic Problem**:  
You generate a joke → want both the joke AND its explanation in final output.

**Solution**:

```python
parallel = RunnableParallel({
    "joke":        RunnablePassthrough(),           # forwards the joke as-is
    "explanation": prompt_explain | model | parser
})

final_chain = joke_chain | parallel
```

Now you get:
```python
{
  "joke": "Why did the AI go to therapy?...",
  "explanation": "This joke plays on the idea that..."
}
```

---

### 4. RunnableLambda → Turn ANY Python function into a Runnable

**Most powerful primitive** – inject custom logic anywhere.

```python
from langchain_core.runnables import RunnableLambda

word_counter = RunnableLambda(lambda text: len(text.split()))

# Cleaner version (actual lambda)
word_counter = RunnableLambda(lambda x: len(x.split()))
```

**Example**: Joke + word count side-by-side

```python
parallel = RunnableParallel({
    "joke":       RunnablePassthrough(),
    "word_count": word_counter
})

final_chain = joke_chain | parallel
# Output: {'joke': '...', 'word_count': 24}
```

Now you can add cleaning, validation, calculations, API calls – anything!

---

### 5. RunnableBranch → If-Else for chains (Conditional Routing)

**Purpose**: Execute different chains based on condition

**Use Case**:  
Generate report → If >500 words → summarize, else → print as-is

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: len(x.split()) > 500, prompt_summarize | model | parser),  # if
    RunnablePassthrough()                                               # else
)

final_chain = report_chain | branch
```

Only **one** branch runs – exactly like Python `if/else`.

---

### LangChain Expression Language (LCEL) – The Future

| Old Way                              | New LCEL Way                  | Status       |
|--------------------------------------|-------------------------------|--------------|
| `RunnableSequence(a, b, c)`         | `a | b | c`                   | Already here |
| `RunnableParallel({...})`            | Maybe `a & b` in future?      | Coming soon? |
| `RunnableBranch(...)`                | Maybe `a.if(condition).then(b).else(c)`? | Future |

**Right now**: Use `|` for sequences (99% of cases)  
**Future**: LCEL will probably cover parallel & branching too.

---

### Final Summary Table – Runnable Primitives Cheat Sheet

| Primitive            | Symbol (LCEL) | Purpose                                 | When to Use                                      |
|----------------------|---------------|-----------------------------------------|---------------------------------------------------|
| RunnableSequence     | `\|`          | Sequential chain                        | Almost always                                     |
| RunnableParallel     | (future `&`?) | Parallel execution                      | Multi-format output, speed                        |
| RunnablePassthrough  | —             | Forward data unchanged                  | Keep original + processed version                 |
| RunnableLambda       | —             | Custom Python logic                     | Cleaning, math, API calls, validation            |
| RunnableBranch       | —             | Conditional routing (if/else)           | Dynamic workflows, classification-based routing  |

---

### What You Can Build Now

After Videos 7–9 (Chains + Runnables Part 1 & 2):

- Sequential chains → `|`
- Parallel chains → `RunnableParallel`
- Conditional chains → `RunnableBranch`
- Custom logic → `RunnableLambda`
- Pass original data → `RunnablePassthrough`

**You are now ready for RAG, Agents, Tools, Memory** – everything in modern LangChain is just composed Runnables!
