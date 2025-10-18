# 🎓 Prompts in LangChain | Generative AI using LangChain 

## 🧠 Introduction

> ⚡ This is **Video 4** — focused on the **second major LangChain component: Prompts**.  
The video explains why prompts are needed, how they work, and how to use them effectively in LangChain.

---

## 🔁 Recap of Previous Videos

| Video | Topic | Summary |
|--------|--------|----------|
| 1 | Introduction to LangChain | Why we need LangChain |
| 2 | 6 Components of LangChain | Real-life example showing how components work together |
| 3 | Models | Deep dive into LLMs and their parameters |
| 4 | Prompts | Current video – understanding and implementing prompts |

---

## ⚙️ Correction from Last Video (Models)
### 🔸 Temperature Parameter Clarification
- **Temperature** controls the **randomness** of the LLM output.

| Temperature | Output Type | Description |
|--------------|--------------|--------------|
| 0 | Deterministic | Same input → same output |
| 0.5–2 | Creative | More varied and creative outputs |

**Example:**
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)
response = model.invoke("Write a 5-line poem on cricket.")
print(response.content)


✅ Use low temperature for factual/consistent tasks.
🎨 Use high temperature for creative generation.


## 💡 What are Prompts?
🧩 Definition

Prompts are input instructions or queries given to a model to guide its output.

Any message you send to an LLM is a prompt.
Example: "Write a 5-line poem on cricket."

## 🧱 Types of Prompts
| Type            | Description                       | Example                     |
| --------------- | --------------------------------- | --------------------------- |
| **Text-based**  | Most common (99%)                 | Interact with LLMs via text |
| **Multi-modal** | Combine text + images/audio/video | “Describe this image”       |


🎯 Focus: Text-based prompts for now.


## 🔑 Importance of Prompts

    LLM outputs depend heavily on prompt design.

    Small wording changes → big differences in responses.

    Led to a new career: Prompt Engineering.

    ✍️ Prompt Engineering = Crafting inputs for desired outputs
    Examples: Zero-shot, Few-shot, Chain-of-Thought prompting.

    - Zero-shot → The model is asked to perform a task without any examples.
    - Few-shot → The model is given a few examples along with the task to guide its output.
    - Chain-of-Thought (CoT) → The model is prompted to reason step by step before giving the final answer.

## 🧍‍♂️ Static vs Dynamic Prompts
    🧊 Static Prompts

    Hardcoded in code — not suitable for real-world apps.

    response = model.invoke("Write a 5-line poem on cricket.")

#### Problems:

    - No personalization
    - Inconsistent outputs
    - User typos and variations

## ⚙️ Dynamic Prompts (Solution)

Use a Prompt Template with placeholders filled at runtime.


### Please summarize the research paper titled {paper_input} with the following 
### specifications:
- Explanation style: {style_input}
- Explanation length: {length_input}
Include equations if available. Use analogies or code where helpful.

### User provides:

    Paper → “BERT”

    Style → “Simple”

    Length → “Medium”

### ✅ Benefits:

    Consistency

    Reusability

    Reduced user error

## 🖥️ Building a UI for Dynamic Prompts (Streamlit)

    pip install streamlit

#### Code (prompt_ui.py):

    import streamlit as st
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    from langchain_core.prompts import PromptTemplate

    load_dotenv()
    st.header("Research Assistant Tool")

    paper = st.selectbox("Select Research Paper", ["Attention is All You Need", "Word2Vec", "BERT", "GPT"])
    style = st.selectbox("Explanation Style", ["Simple", "Math-Heavy", "Code-Heavy"])
    length = st.selectbox("Explanation Length", ["Short", "Medium", "Long"])

    if st.button("Summarize"):
    template = PromptTemplate.from_file("template.json")
    prompt = template.invoke({"paper_input": paper, "style_input": style, "length_input": length})

    model = ChatOpenAI()
    result = model.invoke(prompt)
    st.write(result.content)

Run:

    streamlit run prompt_ui.py

### 🧩 PromptTemplate Class
### ✅ Usage

    from langchain_core.prompts import PromptTemplate

    template = PromptTemplate(
        template="Summarize {paper} in {style} style, {length} length.",
        input_variables=["paper", "style", "length"],
        validate_template=True
    )

    prompt = template.invoke({
        "paper": "Attention is All You Need",
        "style": "simple",
        "length": "short"
    })

### 🛠️ Benefits Over f-strings

| Feature             | f-String | PromptTemplate |
| ------------------- | -------- | -------------- |
| Validation          | ❌        | ✅              |
| Reusable            | ❌        | ✅              |
| Serializable (JSON) | ❌        | ✅              |
| Partial Filling     | ❌        | ✅              |


    template.save("template.json")
    loaded_template = load_prompt("template.json")

Integration:
    chain = template | model
    result = chain.invoke({"paper": "BERT"})



## 💬 Building a Simple Chatbot

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from dotenv import load_dotenv

    load_dotenv()
    model = ChatOpenAI()

    chat_history = [SystemMessage(content="You are a helpful assistant.")]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        chat_history.append(HumanMessage(content=user_input))
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI:", result.content)

    print("Chat History:", chat_history)


Concepts:
    - SystemMessage: Defines role/behavior.

    - HumanMessage: User input.

    - AIMessage: Model response.


### 🧱 Message Types in LangChain
    | Message Type    | Purpose             | Example                        |
    | --------------- | ------------------- | ------------------------------ |
    | `SystemMessage` | Sets behavior/rules | “You are a helpful assistant.” |
    | `HumanMessage`  | User input          | “Tell me about LangChain.”     |
    | `AIMessage`     | Model output        | “LangChain is a framework…”    |


## 💬 ChatPromptTemplate

### For multi-turn conversations with variables.

    from langchain_core.prompts import ChatPromptTemplate

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain in simple terms what is {topic}.")
    ])

    prompt = chat_template.invoke({"domain": "cricket", "topic": "offside"})
    print(prompt)


## 🧱 MessagesPlaceholder
#### Use for inserting past chat history dynamically.

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a customer support agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

    chat_history = [
        HumanMessage(content="Hi, I want a refund."),
        AIMessage(content="Sure, can you provide your order ID?")
    ]

    prompt = chat_template.invoke({
        "chat_history": chat_history,
        "query": "Where is my refund?"
    })
    print(prompt)

💾 In production, use Redis/MongoDB to persist chat history between sessions.



## 🧩 Overall Recap
| Scenario       | Component                                    | Description                  |
| -------------- | -------------------------------------------- | ---------------------------- |
| Single prompt  | `PromptTemplate`                             | For single-turn queries      |
| Chat history   | `ChatPromptTemplate` + `MessagesPlaceholder` | For multi-turn conversations |
| UI integration | Streamlit / Web frameworks                   | For real-world applications  |


## 🧭 Summary Diagram
    User Input
       ↓
    PromptTemplate / ChatPromptTemplate
       ↓
    Model (ChatOpenAI / Llama / etc.)
       ↓
    Response





FYI:
    | **Concept**        | **2-Line Summary**                                                                                                                        |
    | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
    | **Context**        | The full input the model uses (prompts, history, documents).<br>It determines what the model "remembers" during response generation.      |
    | **Context Window** | The max number of tokens the model can process at once (input + output).<br>If exceeded, older data is dropped or errors occur.           |
    | **Token**          | A chunk of text, roughly 4 characters or ¾ of a word.<br>Used to measure input/output length and billing.                                 |
    | **Why It Matters** | Token limits affect model memory, costs, and output quality.<br>Efficient token use = better, cheaper, smarter responses.                 |
    | **100 Tokens**     | ~75 words or ~400 characters, not 100 words.<br>You’re billed for both input and output tokens.                                           |
    | **Model Examples** | GPT-3.5 (16k tokens), GPT-4o (128k), Claude 3 (200k+).<br>Bigger window = better long-context understanding.                              |
    | **Cost Impact**    | More tokens = higher cost (for paid APIs).<br>Use concise prompts to save money.                                                          |
    | **When to Care**   | Important in long chats, RAG, large docs, summarization, and agent memory.<br>Optimizing tokens ensures functionality and budget control. |
