# 🧠 NLP - Section C: Deep Learning in NLP

---

## 10. Sequence Modeling

Text is naturally **sequential** — each word depends on the previous one.

### 🧠 Why Special Handling?
- Order matters: “Dog bites man” ≠ “Man bites dog”
- Models must **remember what came before** to predict what's next

### Traditional Models:
- **RNNs (Recurrent Neural Networks)** process inputs one-by-one with memory
- **LSTMs / GRUs** improve RNNs by handling long-term dependencies better

💡 Sequence models laid the foundation for attention and Transformers.

---

## 11. Attention Mechanism

Attention allows models to **focus on specific parts** of the input when producing an output.

### 🧠 First Principles:
- Instead of treating all input words equally, attention learns **where to “look”**
- Inspired by how humans read: we focus on keywords

### Formula (conceptual):
- Each word attends to **every other word**, assigning weights

📌 Example: In translation, “bank” may attend more to “money” than “river”

💡 Attention improves **context handling, speed, and interpretability**

---

## 12. Transformers

Transformers are the architecture that **replaced RNNs and LSTMs**.

### 🔧 Core Components:
- **Self-Attention**: Every word attends to every other word in a sentence
- **Positional Encoding**: Adds word order (since attention doesn’t track order)
- **Feedforward Layers**: Adds non-linearity
- **Stacked Layers**: Multiple attention blocks stacked together

### Why Powerful?
- Handles **long sequences efficiently**
- Trains in **parallel** (unlike RNNs)
- Captures **rich contextual relationships**

💡 All modern LLMs (GPT, BERT, Claude) are **based on Transformers**

---

## 13. Pretraining vs Fine-tuning

### Pretraining:
- Model is trained on **massive unlabeled data**
- Learns grammar, facts, semantics, reasoning patterns

### Fine-tuning:
- Model is then adapted on **task-specific** data (e.g., sentiment analysis, Q&A)

### Why Important?
- Pretraining gives the model **language understanding**
- Fine-tuning makes it perform **specific jobs**

💡 GPT-3, ChatGPT, and open-source LLMs use this two-step process.

---

## 14. Transfer Learning in NLP

Same principle as DL:
- Reuse a pretrained NLP model (like BERT, GPT)
- Fine-tune or prompt it for specific tasks

### Benefits:
- Saves compute
- Needs less labeled data
- Faster convergence

💡 Transfer learning is **how agents and custom GenAI tools are built** on top of base models.

---

## 15. Masked vs Causal Language Modeling

### 🔹 Masked Language Modeling (MLM)
- Used by BERT
- Predict missing words in the middle of a sentence

📌 Example:
- Input: "The dog [MASK] over the fence"
- Target: "jumped"

### 🔹 Causal Language Modeling (CLM)
- Used by GPT
- Predicts the **next word only**, based on previous ones
- No peeking ahead!

📌 Example:
- Input: "The dog jumped"
- Predict: "over"

💡 Causal modeling is used in **text generation** — making it essential for LLMs and agents.

---

# ✅ Summary

| Concept | What It Enables | Used In |
|--------|------------------|---------|
| Sequence Modeling | Understands word order | RNNs, LSTMs |
| Attention | Focus on relevant context | All modern models |
| Transformers | Efficient, parallel, contextual | BERT, GPT, Claude |
| Pretraining | General knowledge | All LLMs |
| Fine-tuning | Task-specific adaptation | Sentiment, NER, Q&A |
| MLM vs CLM | Understanding vs generation | BERT (MLM), GPT (CLM) |

---

🧭 Next: **Section D – LLM-Specific NLP Concepts**  
(Tokenization, embeddings, chunking, prompts, retrieval, LangChain-style chaining)
