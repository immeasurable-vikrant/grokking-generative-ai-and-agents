# ⚙️ Key Operational Concepts in Large Language Models (LLMs)

Understanding how LLMs (like GPT, Claude, etc.) operate under the hood helps in using them effectively and cost-efficiently.

---

## 🧱 Tokens, Context Windows, and Token Limits

### 🔹 What is a Token?
A **token** is a chunk of text — it can be a word, part of a word, punctuation, or even whitespace.

| Text | Tokens |
|------|--------|
| "ChatGPT is smart." | ["Chat", "G", "PT", " is", " smart", "."] |

- **English estimate**: ~1 token = ¾ of a word

### 🔹 Context Window
- The **context window** is the amount of text (tokens) an LLM can "see" at once.
- **Example**: GPT-4 has 8k and 32k token variants; Claude 2.1 supports up to 200k tokens.

### 🔹 Why It Matters
- If input + output > context window → earlier content is truncated or forgotten.

### 🔹 Real-World Impact
- In a long chat, model might **"forget" earlier messages** (context overflow).
- Using large context (e.g., full documents) increases **latency and cost**.

---

## 🎛️ Controlling LLM Output

| Parameter     | What It Does | Real-World Example |
|---------------|--------------|--------------------|
| `temperature` | Controls randomness (0 = deterministic, 1 = creative) | Lower for math/code, higher for story writing |
| `top_k`       | Picks from top *k* options | `top_k=5` → picks from top 5 most likely tokens |
| `top_p`       | Nucleus sampling: picks from top tokens with cumulative probability *p* | `top_p=0.9` → picks from tokens covering 90% likelihood |
| `max_tokens`  | Max number of tokens in the response | Prevents runaway generations |

### 🧠 Example
{
  "prompt": "Write a startup idea.",
  "temperature": 0.8,
  "top_p": 0.9,
  "max_tokens": 100
}


## ✍️ Prompting Types
🔹 Basic Prompting
    Translate to French: "Good morning"

🔹 Structured Prompting (Templates)
    You are a helpful assistant. When given a product, describe its benefits.
    Product: "Noise Cancelling Headphones"
    Benefits:

🔹 Zero-shot Prompting
No examples provided; relies on general knowledge.

    Classify: "This movie was boring and too long." → Negative

🔹 Few-shot Prompting

Gives a few examples to guide the model.

    Translate:
    English: Hello → Spanish: Hola  
    English: Thank you → Spanish: Gracias  
    English: Good night → Spanish:


## 🔌 Function / Tool Calling

- Function calling allows models to call external tools or APIs.
- LLM returns structured data for external use.

🔹 Example (OpenAI Function Call)
    {
      "function": "get_weather",
      "arguments": { "location": "Delhi", "unit": "celsius" }
    }
Real use: Chatbot fetching live weather, database queries, triggering scripts.


## 💰 Rate Limits & Token Costs
| Provider  | Rate Limit            | Cost Per Token / 1K Tokens            |
| --------- | --------------------- | ------------------------------------- |
| OpenAI    | Depends on model/tier | ~$0.0015–$0.12                        |
| Anthropic | Context-based limits  | Priced per input/output token         |
| Cohere    | Based on tier/model   | Typically cheaper for open-source use |

- Exceeding limits → throttling or blocked requests.

- Higher models (e.g., GPT-4, Claude 2.1) → slower + more expensive.


## 🧠 Context Restrictions

LLMs are stateless:
They don’t “remember” previous chats unless you send them in the context window again.

🔹 Why?

    - LLMs do not have persistent memory (unless implemented externally).
    - Older parts of a long conversation may be dropped (context truncation).

## ⚖️ Latency vs Cost vs Quality Trade-Offs

| Factor       | Higher Value Means              | Trade-Off                             |
| ------------ | ------------------------------- | ------------------------------------- |
| Model Size   | Better output, deeper reasoning | More latency, higher cost             |
| Context Size | More data at once               | Slower inference, higher cost         |
| Temperature  | More creative outputs           | Less reliable/factual answers         |
| Zero-shot    | Faster + cheaper prompting      | May reduce accuracy for complex tasks |

🔹 Real Example
    - Customer chatbot: Use GPT-3.5 for speed and cost
    - Medical reasoning tool: Use GPT-4 or Claude 2.1 for depth despite cost

### TL;DR

    - Tokens are the currency of LLMs — control your cost and context size.
    - Use temperature, top_k, and top_p to balance creativity vs control.
    - Prompt smartly: choose zero-shot for simplicity, few-shot for guidance.
    - Function calling makes LLMs interactive with tools and APIs.
    - Trade-off between speed, accuracy, and cost is crucial in real-world LLM deployment.