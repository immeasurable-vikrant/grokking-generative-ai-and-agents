# 🧠 NLP - Section B: Word Representations & Semantics

---

## 7. Bag of Words (BoW) & TF-IDF

These are **classical methods** to convert text into numbers for ML models.

### 🔹 Bag of Words (BoW)
- Represents text by the **count of words** in a vocabulary
- Ignores grammar and word order

📌 Example:
- "Chatbots are fun" → {"chatbots": 1, "are": 1, "fun": 1}

### 🔹 TF-IDF (Term Frequency - Inverse Document Frequency)
- Weighs how **important** a word is in a document relative to all other documents
- Common words like “the” get lower scores, while unique words get higher scores

💡 These are good for simple tasks like classification, but **fail to capture meaning or context**.

---

## 8. Word Embeddings

Word embeddings are **dense vector representations** of words that capture their **semantic meaning**.

### 🧠 First Principles:
- Similar words have **similar vectors** (based on context in which they appear)
- Embeddings learn relationships like:
  - `vec("king") - vec("man") + vec("woman") ≈ vec("queen")`

### Popular Models:
| Model | Method |
|-------|--------|
| Word2Vec | Predicts surrounding words (CBOW, Skip-gram) |
| GloVe | Learns from word co-occurrence statistics |
| FastText | Considers subwords (good for rare words or misspellings) |

💡 Embeddings are the **first step towards understanding context** in modern NLP pipelines.

---

## 9. Contextual Embeddings

Unlike static embeddings, **contextual embeddings change depending on surrounding words**.

### 🧠 Why Needed?
- "bank" in “river bank” ≠ "bank" in “money bank”
- Word2Vec gives one vector per word; contextual models give different vectors based on **usage**

### Key Models:
| Model | Key Idea |
|-------|----------|
| ELMo | Uses deep bi-directional LSTMs |
| BERT | Uses transformers and bidirectional attention |
| GPT | Uses transformer decoder (causal attention) |

💡 Contextual embeddings are what make **LLMs powerful**, enabling real language understanding.

---

# ✅ Summary

| Technique | Captures Meaning? | Context-Aware? | Usage |
|-----------|-------------------|----------------|-------|
| BoW       | ❌ No             | ❌ No          | Classical ML |
| TF-IDF    | ⚠️ Slightly       | ❌ No          | Classical ML |
| Word2Vec  | ✅ Yes            | ❌ No          | Lightweight DL |
| GloVe     | ✅ Yes            | ❌ No          | Fast embeddings |
| ELMo/BERT | ✅✅ Yes          | ✅ Yes         | Modern NLP / LLMs |

---

🧭 Next: **Section C – Deep Learning in NLP**  
(Learn about sequences, attention, and transformers — the core of LLMs)
