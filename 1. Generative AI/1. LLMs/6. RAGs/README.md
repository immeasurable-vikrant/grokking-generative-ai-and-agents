# 🔍 3. Retrieval & Knowledge Integration (RAG)

RAG (Retrieval-Augmented Generation) enhances LLMs by injecting **external knowledge** into their responses. It overcomes context limitations and hallucinations by combining **retrieval + generation**.

---

## 🧠 Why RAG? – The Problem of Limited Context

Large Language Models (LLMs) have **context windows** (e.g., 8k–200k tokens). Beyond that:

- Models **forget** earlier content
- Important info may be **truncated**
- Embedding everything upfront becomes **inefficient and expensive**

### 🔹 Example
A user asks:
"What did the company say in its June 2023 earnings report?"

You can't feed in the entire financial history—you must retrieve only the relevant chunk from June 2023.

✅ Enter RAG: Retrieve only what’s needed → Inject → Generate.

## 🧬 Embeddings (for Retrieval)
Embeddings turn text into dense vectors that capture semantic meaning.

🔹 Example

| Text                        | Embedding (simplified)              |
| --------------------------- | ----------------------------------- |
| "Apple Inc. profits rose"   | [0.21, 0.98, -0.33, ...]            |
| "Company revenue increased" | [0.22, 0.97, -0.35, ...] ✅ Similar! |


Use embedding models like:
    - OpenAI (text-embedding-3-small)

    - Cohere, HuggingFace (all-MiniLM)

    - BGE, E5, GTE (for multilingual or domain-specific use)

## 🗃️ Vector Databases

Store and retrieve embeddings efficiently.

| Tool             | Highlights                       |
| ---------------- | -------------------------------- |
| **Pinecone**     | Scalable, managed cloud DB       |
| **Weaviate**     | Semantic + hybrid search         |
| **FAISS**        | Open-source, fast local indexing |
| **Milvus**       | Distributed and performant       |
| **Redis Vector** | Integrates well with Redis stack |

These power similarity search in retrieval pipelines.

## ✂️ Chunking Strategies (Text Splitting)

Before embedding, long texts must be chunked.

🔹 Common Strategies
| Strategy    | Description                                        | Example                      |
| ----------- | -------------------------------------------------- | ---------------------------- |
| Fixed-size  | Split every N tokens/words                         | 500-token chunks             |
| Recursive   | Split on sections → paragraphs → sentences → words | Keeps structure              |
| Overlapping | Add 10–20% overlap between chunks                  | Prevents cutting mid-thought |


🔹 Real Example

    An FAQ document is split into:

    Q1 + A1 → Chunk 1

    Q2 + A2 → Chunk 2

    ...

    Each chunk is embedded and stored.


## 🔁 Retrieval Pipeline
[User Query]
    ↓
[Embedding]
    ↓
[Vector Search (Similarity)]
    ↓
[Top-k Chunks]
    ↓
[Inject into Prompt Template]
    ↓
[LLM Generates Final Answer]


🔹 Prompt Template Example
    Answer the question using the context below.

    Context:
    {retrieved_chunk_1}
    {retrieved_chunk_2}

    Question: {user_query}
    Answer:

## ⚙️ Hybrid Search (Semantic + Keyword)

Combine:
    - Semantic search: via embeddings
    - Keyword search: via BM25, Elasticsearch, etc.

This improves accuracy when:

    Specific keywords are critical

    Numeric/textual identifiers (e.g., "Model X123") are present

    Many tools (like Weaviate) offer hybrid search out-of-the-box.

## ✅ RAG Evaluation & ❌ Failure Modes
✅ Evaluation Metrics
    | Metric           | Purpose                                           |
    | ---------------- | ------------------------------------------------- |
    | **Hit rate**     | Was the relevant chunk retrieved?                 |
    | **Faithfulness** | Was the answer grounded in the retrieved context? |
    | **F1 / ROUGE**   | Compare generated vs ground truth (for QA tasks)  |
    | **Human eval**   | Check factuality and helpfulness                  |


## ❌ Failure Cases

    | Issue                  | Description                           | Example                                           |
    | ---------------------- | ------------------------------------- | ------------------------------------------------- |
    | ❌ Bad chunking         | Important info split between chunks   | “Terms and conditions” separated from explanation |
    | ❌ Irrelevant retrieval | Top-k results don’t match query       | Answer refers to wrong product version            |
    | ❌ Context overflow     | Too many chunks → truncation          | LLM drops relevant chunk                          |
    | ❌ Hallucination        | LLM answers outside retrieved context | Answer fabricated beyond provided docs            |


## 🔍 Real-World Use Cases

1. Customer Support Bot

    - Retrieves product manuals → answers user queries

    - Vector DB: Pinecone

    - Chunking: 300-token + overlap

2. Legal Document Q&A

    - Retrieve relevant case law or clauses

    - Hybrid search: keyword for statute, semantic for meaning

3. Internal Knowledge Assistant

    - Employees ask about internal policies

    - Retrieval from Notion/Confluence → context → GPT-4 answer


### TL;DR

| Concept        | Summary                                             |
| -------------- | --------------------------------------------------- |
| RAG            | Retrieval + Generation to inject external knowledge |
| Embeddings     | Turn text into meaning vectors                      |
| Vector DB      | Stores and retrieves similar chunks                 |
| Chunking       | Splits text efficiently for retrieval               |
| Pipeline       | Embed → Search → Inject → Generate                  |
| Hybrid Search  | Combines keyword and semantic matching              |
| RAG Evaluation | Measures retrieval + output quality                 |
