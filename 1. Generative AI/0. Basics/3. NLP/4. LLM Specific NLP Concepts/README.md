# 🤖 NLP - Section D: LLM-Specific NLP Concepts

---

## 16. Tokenization for LLMs (BPE, SentencePiece)

LLMs don't work with words — they use **tokens**, which are subword units.

### 🔧 Common Tokenizers:
- **Byte Pair Encoding (BPE)**: Splits text into frequently occurring subwords (e.g., “unhappiness” → “un”, “happi”, “ness”)
- **SentencePiece**: Learns tokens in a language-agnostic way (used by T5, LLaMA)

### Why It Matters:
- Token count affects **cost, context length, and speed**
- Different LLMs use different tokenization styles

💡 Agents need to **count tokens** before sending inputs to models due to context length limits (e.g., 4K, 16K, 128K).

---

## 17. Prompt Engineering Basics

Prompt engineering is the art of crafting inputs to **elicit the right response** from an LLM.

### 🧠 Key Principles:
- LLMs are sensitive to **structure and phrasing**
- Adding examples (“few-shot”), instructions, or format hints improves performance

### Types:
| Prompt Type | Use |
|-------------|-----|
| Zero-shot | Ask directly |
| Few-shot | Show examples |
| Chain-of-thought | Ask it to explain step-by-step |
| Role-based | “You are a helpful assistant…” |

💡 Well-engineered prompts are **the interface between agent logic and LLM behavior**.

---

## 18. Embeddings for Search & Retrieval

**Embeddings** represent text as vectors that capture semantic meaning.

### Use Cases:
- **Semantic Search**: Find relevant documents, even with different wording
- **Similarity Matching**: Detect duplicates or related concepts

### How It Works:
- Convert query and documents into vectors
- Use **cosine similarity** to compare

💡 Retrieval-Augmented Generation (RAG) agents use embeddings to **fetch relevant info** before answering.

---

## 19. Text Chunking & RAG (Retrieval Augmented Generation)

LLMs can't process long documents at once (due to token limits), so we **chunk** them.

### 🧩 Chunking:
- Split documents into sections (by paragraph, sentence, tokens)
- Add metadata (source, heading, etc.)

### 🧠 RAG Flow:
1. User asks a question
2. Use **embedding search** to find top-k chunks
3. Inject chunks + question into a prompt
4. LLM answers based on retrieved context

💡 RAG powers **memory, tool-use, and factual answering** in GenAI agents.

---

## 20. Chaining & Tool Use (LangChain-style logic)

Most tasks involve **multiple LLM calls + external tools**.  
Agents manage this via **chaining**.

### 🔧 What is a Chain?
A sequence of steps:
1. Prompt → LLM → output
2. Use output as input for next prompt/tool
3. Loop until goal is met

### 🔨 Tool Use:
- Agent uses LLM to decide when to call:
  - APIs (FastAPI endpoints)
  - Code execution
  - Database queries
  - Document retrieval

💡 LangChain, LangGraph, and similar frameworks build this chaining logic to automate workflows with LLMs + tools.

---

# ✅ Summary

| Concept | What It Enables | Used In |
|--------|------------------|---------|
| Tokenization | Prepares input for LLMs | GPT, LLaMA, T5 |
| Prompt Engineering | Controls LLM behavior | All LLMs, agents |
| Embeddings | Semantic understanding | Search, RAG |
| Chunking | Fits long docs into LLMs | Document Q&A |
| RAG | Enhances LLM with memory | Agents, enterprise AI |
| Chaining & Tools | Multi-step workflows | LangChain, LangGraph |

---

🧭 Next: **Section E – Practical & API-Focused NLP Concepts**
(Serving models, using Hugging Face, building FastAPI agents)
