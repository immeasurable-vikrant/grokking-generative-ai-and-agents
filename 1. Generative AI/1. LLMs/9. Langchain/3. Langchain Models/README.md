# LangChain Models: In-Depth Tutorial with Code Demonstrations

## 🧠 Introduction to LangChain Models

Welcome to this detailed exploration of the **model** component in LangChain. Having already discussed the foundational concepts—such as what LangChain is, its use cases, and core components (models, prompts, chains, agents)—we now dive into hands-on, code-driven tutorials.

This guide will walk you through how to:

- Understand the role of models in LangChain
- Interact with both closed-source and open-source models
- Use language models and embedding models effectively
- Build practical applications like chatbots and document similarity tools

By the end, you’ll be confident in implementing the **models** component in real-world projects.

---

## 🧾 Recap of Key Concepts

Before implementation, a brief recap:

- **LangChain Overview**: A modular framework for LLM-based applications.
- **Applications**: From chatbots to RAG systems.
- **Components**: Models (AI interface), Prompts (inputs), Chains (pipelines), Agents (decision-making).

Now we zoom in on the **Models** component — the foundation of LLM interaction.

---

## 🔍 Understanding the Model Component

LangChain provides a **unified interface** to communicate with multiple AI models across different providers.

There are two major types:

- **Language Models**  
  → Input: Text | Output: Text  
  → Use: Chatbots, Q&A, summarization

- **Embedding Models**  
  → Input: Text | Output: Vector  
  → Use: Semantic search, similarity matching

### 📊 Diagram Overview (Conceptual)
    Models
    ├── Language Models → "What is the capital of India?" → "New Delhi"
    └── Embedding Models → "What is the capital of India?" → [0.123, -0.456, ...]


This structure allows for flexible and consistent interaction with diverse AI capabilities.

---

## 🧠 Types of AI Models in LangChain

| Type            | Input   | Output  | Example Use Case                  |
|-----------------|---------|---------|-----------------------------------|
| Language Models | Text    | Text    | Chatbots, summarization, QA       |
| Embedding Models| Text    | Vectors | Semantic search, RAG systems      |

---

## 🗺️ Plan of Action

We will implement both types of models with hands-on examples.

### Language Models
- Closed-source: OpenAI (GPT), Claude (Anthropic), Gemini (Google)
- Open-source: Hugging Face (TinyLlama)
- Build: A simple chatbot

### Embedding Models
- Closed-source: OpenAI
- Open-source: Hugging Face (Sentence Transformers)
- Build: Document similarity app

---

## 💬 Deep Dive into Language Models

LangChain separates LLMs into:

- **LLMs (Legacy)**: Input/output as raw text
- **Chat Models (Preferred)**: Optimized for multi-turn conversations

### 🆚 LLMs vs Chat Models

| Aspect            | LLMs                          | Chat Models                         |
|-------------------|-------------------------------|-------------------------------------|
| Purpose           | Free-form NLP tasks           | Multi-turn conversations            |
| Training Data     | Books, articles                | Plus chats, conversations           |
| Memory Support    | ❌ Stateless                   | ✅ Context-aware                     |
| Role Awareness    | ❌ None                        | ✅ Supports roles (system/user)      |
| Examples          | GPT-2, BLOOM                  | GPT-4o, Claude, Gemini               |
| Recommended For   | Legacy NLP tasks               | Chatbots, agents, assistants        |

---

## 🛠️ Setup for Implementation

### 1. Create Project Folder

    mkdir langchain_models && cd langchain_models

### 2. Create and Activate Virtual Environment
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate

### 3. requirements.txt
    langchain
    langchain-openai
    langchain-anthropic
    langchain-google-genai
    langchain-huggingface
    openai
    anthropic
    google-generativeai
    python-dotenv
    scikit-learn
    numpy
    huggingface-hub

pip install -r requirements.txt

### 4. Create File Structure
    mkdir llms chat_models embeddings
    touch test.py

Test installation:
# test.py
    import langchain
    print(langchain.__version__)


## 🔐 Working with Closed-Source Language Models

    These require API keys from providers (e.g., OpenAI, Google, Anthropic).    
    ✅ OpenAI (LLM Demo)

    File: llms/llm_demo.py

        from langchain_openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        llm = OpenAI(model="gpt-3.5-turbo-instruct")
        result = llm.invoke("What is the capital of India?")
        print(result)
    📝 Note: This uses legacy LLM interface. Prefer chat models for new projects.


## 💬 Chat Models (Modern)
🔹 OpenAI (Chat Model)

    File: chat_models/chat_model_openai.py

        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        load_dotenv()
        model = ChatOpenAI(model="gpt-4o")
        result = model.invoke("What is the capital of India?")
        print(result.content)
    Parameters: temperature, max_tokens, etc.

🔹 Anthropic (Claude)
    Use: ChatAnthropic with model "claude-3-5-sonnet-20240620"

🔹 Google (Gemini)
    Use: ChatGoogleGenerativeAI with model "gemini-1.5-pro"

LangChain ensures a consistent interface across providers.


## 🧪 Open-Source Language Models

Open-source models offer:

    ✅ Zero cost
    ✅ Full control
    ✅ Offline access

    Popular Models:

    LLaMA (Meta)

    Mistral

    Falcon

    BLOOM

    TinyLlama

### Using Hugging Face Inference API

    File: chat_models/chat_model_huggingface_api.py

        from langchain_huggingface import ChatHuggingFace
        model = ChatHuggingFace(
            repo_id="chinu/tinyllama-1.1b-chat-v1.0", 
            task="text-generation"
        )
        result = model.invoke("What's the capital of India?")
        print(result.content)

    - repo_id is the unique identifier of a model hosted on Hugging Face Hub
    - task parameter tells the Hugging Face inference endpoint what kind of operation
    the model is built for.
👉 Inference = Using a trained model to get predictions or answers.    
On Hugging Face, “inference” refers to:
    - The API service that runs the model (e.g., “Inference API” or “Inference Endpoints”)

### Local Download and Execution:
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline

    pipe = pipeline("text-generation", model="chinu/tinyllama-1.1b-chat-v1.0")
    model = HuggingFacePipeline(pipeline=pipe)
    result = model.invoke("Who is Virat Kohli?")
    print(result)


## 📈 Embedding Models

Embeddings = Convert text → vectors (semantic search, similarity).

    🔒 OpenAI Embeddings

    File: embeddings/embedding_openai_query.py
        from langchain_openai import OpenAIEmbeddings
        from dotenv import load_dotenv

        load_dotenv()
        embed = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
        vector = embed.embed_query("Delhi is the capital of India")
        print(vector)

    Use embed_documents([...]) for multiple texts.

    🌐 Hugging Face (Local)

    File: embeddings/embedding_hf_local.py

        from langchain_huggingface import HuggingFaceEmbeddings

        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector = embed.embed_query("Tell me about Sachin Tendulkar")
        print(vector)

## 🔍 Document Similarity App

File: embeddings/document_similarity.py

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from langchain_huggingface import HuggingFaceEmbeddings

    docs = [
        "Virat Kohli is a modern-day cricket legend...",
        "MS Dhoni led India to World Cup glory...",
        "Rohit Sharma is known for his double centuries...",
        "Kapil Dev was the first Indian WC-winning captain...",
        "Sachin Tendulkar is called the God of Cricket..."
    ]

    query = "Tell me about Virat Kohli"
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    doc_vecs = embed.embed_documents(docs)
    query_vec = embed.embed_query(query)

    scores = cosine_similarity([query_vec], doc_vecs)[0]
    best_index = np.argmax(scores)

    print(f"Query: {query}")
    print(f"Most Relevant Doc: {docs[best_index]}")
    print(f"Score: {scores[best_index]}")
    
Foundation for building RAG pipelines and smart search engines.


#### FYI (Embedding Vectors):

    - Text embeddings convert sentences into vectors (lists of numbers) so computers can “understand” meaning.

    - Each vector lives in a meaning space, where similar meanings are close together and unrelated meanings are far apart.

    - Dimensions = number of features describing the text (e.g., 32D means 32 numbers per vector).

    - Floating-point numbers allow fine-grained meaning representation; negatives indicate opposition or absence along a feature.

    - The model learns embeddings by reading massive text, adjusting numbers so words/sentences with similar context cluster together.

    - Distance or angle between vectors measures semantic similarity, enabling search, clustering, or reasoning in NLP.