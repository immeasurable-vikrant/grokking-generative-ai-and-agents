Prompts in LangChain | Generative AI using LangChain | Video 4 | CampusX
Introduction
Hello everyone! My name is Nitish, and welcome to my YouTube channel. In this video, we continue our LangChain playlist. I apologize for the slight delay due to travel, but I'm back now and will speed up the remaining videos. Today's topic is Prompts, the second important component in LangChain after Models (which we covered in the last video).
The video is kept simple and lucid to avoid confusion. We'll explain why prompts are needed and cover them in detail.
Recap of Previous Videos
Before starting, let's quickly recap what we've covered so far in this playlist:

Video 1: Detailed introduction to LangChain and why we need this framework.
Video 2: Explained the 6 most important components of LangChain with a real-life example to show how they work together.
Video 3: Deep dive into the first component - Models. We covered interacting with different LLMs in detail.

This is the fourth video, focusing on the second component: Prompts. We'll cover end-to-end how prompts work in LangChain, important classes, and scenarios to use them.

Correction from Last Video (Models)
In the last video on Models, there was a small mistake in explaining the temperature parameter for LLMs. I said temperature ranges from 0 to 2, where 0 gives deterministic outputs and 2 gives creative ones. That's partially correct, but let's clarify properly (thanks to a student who pointed it out in comments).

Temperature controls the randomness of the LLM's output.
At temperature = 0: For the same input, you always get the exact same output (deterministic).
As temperature increases (e.g., 0.5, 1.5, up to 2): Outputs become more varied and creative for the same input.

Example Code (from last video):

from langchain_openai import ChatOpenAI
model = ChatOpenAI(temperature=0)  # Deterministic output
response = model.invoke("Write a 5-line poem on cricket.")
print(response.content)

Run multiple times at temp=0: Same poem every time.
At higher temp (e.g., 1.5): Different, more creative poems each time.

Use low temperature for consistent applications (e.g., factual summaries) and high for creative ones (e.g., storytelling). Thanks to the student for the correction—YouTube feedback helps!
What are Prompts?
What are Prompts?
Definition: Prompts are input instructions or queries given to a model to guide its output.

Basically, any message you send to an LLM is a prompt.
We've used prompts in the last video without calling them that (e.g., "Write a 5-line poem on cricket").

Types of Prompts

Text-based Prompts: Most common (99% of cases today). Interact via text instructions.
Multi-modal Prompts: Combine text with other modes like images, audio, or video (e.g., upload an image to ChatGPT and ask questions about it, or analyze a song/video).

Focus: We'll cover text-based prompts, as they're dominant now. Multi-modal may become more prominent in the future.
Importance of Prompts

LLM outputs heavily depend on prompts. Small changes can lead to big differences in responses.
Prompts are so crucial that a new job profile has emerged: Prompt Engineering.
Planned future playlist: Dedicated to Prompt Engineering techniques (e.g., Chain of Thought, Few-Shot Prompting).
In this video: Focus on how to design prompts in LangChain.

Elaboration: Prompt engineering involves crafting inputs to get desired outputs. Techniques include zero-shot (no examples), few-shot (few examples), and chain-of-thought (step-by-step reasoning). While not covered deeply here, remember: Good prompts reduce hallucinations and improve accuracy.

Static vs Dynamic Prompts
Before diving in, understand how we've been writing prompts so far.
Example from last video:

response = model.invoke("Write a 5-line poem on cricket.")

This is a static prompt: Hardcoded by the programmer. Not ideal for real-world apps where users provide inputs.

Problem with Static Prompts

In real apps (e.g., research assistant tool), users type prompts via UI.
Giving users full control over prompts can lead to inconsistencies (e.g., typos in paper names, varying styles).
Outputs become unpredictable, as LLMs are sensitive to prompt wording.
Hard to ensure consistent user experience (e.g., always include analogies or math details).

Solution: Dynamic Prompts

Use a prompt template with placeholders filled at runtime based on user inputs.
Example Template:

Please summarize the research paper titled {paper_input} with the following specifications:
- Explanation style: {style_input}
- Explanation length: {length_input}
Include relevant mathematical equations if present in the paper. Explain concepts using simple analogies or code snippets where applicable. If information is unavailable, say "Insufficient information available." Ensure the summary is clear, accurate, and aligned with the provided style and length.


User provides: Paper name, style (e.g., simple, math-heavy, code-heavy), length (short, medium, long).
Benefits: Consistency, reduced errors, reusable.

Building a UI for Dynamic Prompts (Using Streamlit)
We'll create a simple web app for a research assistant.
Install: pip install streamlit
Code (prompt_ui.py):

import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.header("Research Assistant Tool")

# Dropdowns for user inputs
paper = st.selectbox("Select Research Paper", ["Attention is All You Need", "Word2Vec", "BERT", "GPT"])
style = st.selectbox("Explanation Style", ["Simple", "Math-Heavy", "Code-Heavy"])
length = st.selectbox("Explanation Length", ["Short", "Medium", "Long"])

if st.button("Summarize"):
    # Load template (or define inline)
    template = PromptTemplate.from_file("template.json")  # Or define string template here
    prompt = template.invoke({"paper_input": paper, "style_input": style, "length_input": length})
    
    model = ChatOpenAI()
    result = model.invoke(prompt)
    st.write(result.content)


Run: streamlit run prompt_ui.py
User selects options, clicks "Summarize" → Dynamic prompt generated and sent to LLM.

Elaboration: Streamlit makes quick UIs easy. In production, use databases for templates or integrate with Flask/Django for more complex apps.
PromptTemplate Class

Used for creating dynamic single messages.
Placeholders: Use {variable}.
Input Variables: List them for validation.


template = PromptTemplate(
    template="Summarize {paper} in {style} style, {length} length.",
    input_variables=["paper", "style", "length"],
    validate_template=True  # Enables validation
)
prompt = template.invoke({"paper": "Attention is All You Need", "style": "simple", "length": "short"})


Benefits Over f-strings

Built-in Validation: Checks if all placeholders match input variables (throws error if mismatch).
Reusability: Save as JSON for loading in multiple files.


template.save("template.json")
loaded_template = load_prompt("template.json")

Integration with LangChain Ecosystem: Easily chain with models.

chain = template | model  # Pipe operator for chaining
result = chain.invoke({"paper": "..."})


Missing Topic Addition: f-strings are Python-specific and lack serialization. PromptTemplate supports partial filling (e.g., fix some variables upfront) for more flexibility.
Building a Simple Chatbot
Let's build a console-based chatbot to demonstrate prompts in conversations.
Code (chatbot.py):


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


Infinite loop for conversation.
Maintain chat history as list of messages.
Problem: Without labels, context can get confusing in long chats.
Solution: Use message types (System, Human, AI) for labeling.

Message Types in LangChain
LangChain supports three message types for conversations:

SystemMessage: Top-level instructions (e.g., "You are a helpful assistant. Answer patiently.").
HumanMessage: User inputs.
AIMessage: LLM responses.

Example:


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain.")
]
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)


Elaboration: These ensure context clarity. System messages set behavior; human/AI alternate in history.
ChatPromptTemplate

For dynamic lists of messages (multi-turn conversations).
Similar to PromptTemplate but for chat scenarios.

Example:




from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms what is {topic}.")
])
prompt = chat_template.invoke({"domain": "cricket", "topic": "offside"})
print(prompt)


Note: Use tuples for roles if classes don't work directly (LangChain quirk in older versions).
MessagePlaceholder

Special placeholder in ChatPromptTemplate for inserting dynamic chat history at runtime.
Useful for loading past conversations from databases.

Example Scenario: Customer support chatbot for refunds.

Save history to file/database.
Load and insert via placeholder.

Code:

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Template with placeholder
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Load history from file (simulate)
with open("chat_history.txt", "r") as f:
    lines = f.readlines()
chat_history = [HumanMessage(content=lines[0].strip()), AIMessage(content=lines[1].strip())]

prompt = chat_template.invoke({"chat_history": chat_history, "query": "Where is my refund?"})
print(prompt)


Elaboration: In real apps, use Redis/MongoDB for history. This prevents context loss in multi-session chats.
Overall Recap (Logical Diagram)

Invoke Model:

Single Message: For standalone queries.

Static: Hardcoded.
Dynamic: Use PromptTemplate.


List of Messages: For multi-turn conversations.

Static History.
Dynamic: Use ChatPromptTemplate + MessagePlaceholder.





This covers prompts comprehensively for LangChain. Future: Prompt Engineering playlist (e.g., Chain of Thought, Few-Shot).



















# 🎓 Prompts in LangChain | Generative AI using LangChain | Video 4 | CampusX

## 🧠 Introduction
Hello everyone! My name is **Nitish**, and welcome to my YouTube channel.  
In this video, we continue our **LangChain playlist**.

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

## 🧍‍♂️ Static vs Dynamic Prompts
    🧊 Static Prompts

Hardcoded in code — not suitable for real-world apps.

response = model.invoke("Write a 5-line poem on cricket.")
    Problems:

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

Code (prompt_ui.py):
python 
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


💬 ChatPromptTemplate

    For multi-turn conversations with variables.

    from langchain_core.prompts import ChatPromptTemplate

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain in simple terms what is {topic}.")
    ])

    prompt = chat_template.invoke({"domain": "cricket", "topic": "offside"})
    print(prompt)


## 🧱 MessagesPlaceholder
Use for inserting past chat history dynamically.

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
