# 🧪 2.3 Evaluation & Safety in LLMs

Understanding how to **evaluate**, **align**, and **control** large language models (LLMs) is critical for deploying them in real-world use cases responsibly.

---

## 🧠 Hallucination

### 🔹 What is Hallucination?
When an LLM generates **factually incorrect or made-up information** with high confidence.

### 🔹 Why It Happens
- Models **predict next tokens** based on patterns, not truth.
- Lack of access to real-time or verified data.
- Prompt ambiguity or limited context.

### 🔹 Real-World Example

Prompt: "Who was the first woman to walk on the moon?"
Hallucinated Answer: "Sally Ride in 1983" ❌
Reality: No woman has walked on the moon as of 2025.


🔹 Mitigation

    - Use retrieval-augmented generation (RAG).
    - Apply system prompts for factuality.
    - Post-processing with verifiers or human-in-the-loop.


## 🛡️ Safety Alignment & System Prompts
🔹 What is Safety Alignment?
- Ensuring the model's behavior is helpful, harmless, and honest — aligned with human intent and values.

🔹 How?

- System prompts: Special hidden instructions that shape model behavior.

- Fine-tuning: Training on filtered, safe, aligned data.

- Reinforcement Learning from Human Feedback (RLHF): Optimizing outputs based on human preferences.

🔹 Example: System Prompt
    You are a helpful, respectful assistant. Do not provide harmful or unsafe information.

This reduces risk of:

    Toxic language

    Biased or offensive responses

    Dangerous instructions (e.g., how to make explosives)

📏 Evaluation Metrics
🔹 1. Perplexity (High-Level)

    - Measures how "surprised" the model is by the test data.

    - Lower perplexity = better fit to data (but not always better quality).

    - Often used in language modeling pretraining.

    ⚠️ Perplexity is not useful for generation quality.

🔹 2. BLEU & ROUGE (Automated)

    | Metric    | Purpose                            | Best For      |
    | --------- | ---------------------------------- | ------------- |
    | **BLEU**  | Overlap of n-grams with references | Translation   |
    | **ROUGE** | Recall-based metric for overlap    | Summarization |


    Limitations:

        Rigid word-overlap based.

        Cannot capture creativity or coherence.

        Miss context nuances (e.g., paraphrasing).

🔹 3. Human Evaluation (Gold Standard)
- Humans rate generated outputs based on criteria:

    - Relevance
    - Factuality
    - Helpfulness
    - Coherence

- Time-consuming but most accurate.

Example Use:

    Comparing outputs of GPT-3.5 vs GPT-4

    Evaluating factual summaries from news articles


## 🎲 Deterministic vs Stochastic Outputs
| Type              | Description                                      | Use Case Example                |
| ----------------- | ------------------------------------------------ | ------------------------------- |
| **Deterministic** | Same input = same output (e.g., `temperature=0`) | Legal documents, factual Q&A    |
| **Stochastic**    | Adds randomness (e.g., `temperature > 0`)        | Creative writing, brainstorming |



### 🔹 Real-World Example:
    Prompt: "Write a startup idea involving cats and AI."

    - Deterministic: "A smart litter box that detects health issues."
    - Stochastic: "An AI therapist for lonely cats that plays mood-based music."


### ✅ TL;DR
| Concept            | Summary                                                     |
| ------------------ | ----------------------------------------------------------- |
| Hallucination      | Outputs may sound real but are factually false              |
| Safety Alignment   | Ensures outputs are safe, respectful, and honest            |
| System Prompts     | Hidden instructions to guide model behavior                 |
| Evaluation Metrics | Automated (BLEU, ROUGE), statistical (perplexity), or human |
| Determinism        | Controls randomness for creative vs consistent output       |
