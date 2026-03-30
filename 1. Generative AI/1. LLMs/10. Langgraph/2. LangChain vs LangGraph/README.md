# Agentic AI using LangGraph - Video 3 Notes

## Overview
**Video Title**: Introduction to LangGraph – Why LangGraph Exists & LangChain vs LangGraph Comparison  

**Playlist Position**: Third Video in "Agentic AI using LangGraph" Playlist  

**Previous Videos**:
- Video 1: Difference between Generative AI and Agentic AI
- Video 2: What is Agentic AI? (Definition, Characteristics, Components + HR Recruiter Example)

**Goal of This Video**:
- Understand **why LangGraph exists** and what problem it solves that LangChain cannot handle easily.
- Get a technical overview of LangGraph.
- Clear comparison: **LangChain vs LangGraph** – When to use which?

**Prerequisites**: Basic knowledge of LangChain (Models, Prompts, Retrievers, Chains, Tools, basic Agents).

---

### 1. Quick Recap of LangChain

**LangChain** is an open-source library designed to **simplify building LLM-based applications**.

**Core Building Blocks (Modular Components)**:
- **Models**: Unified interface to work with any LLM (OpenAI, Anthropic, Hugging Face, Ollama, etc.)
- **Prompts**: Helps in prompt engineering and templating
- **Retrievers**: Fetch relevant documents from vector stores (used in RAG)
- **Chains**: The most powerful feature – connect multiple components so output of one becomes input of next

**What can you build easily with LangChain?**
- Simple conversational workflows (Chatbots, Text Summarizers)
- Multi-step linear workflows (e.g., Topic → Detailed Report → Summary)
- RAG-based applications (Chat with your documents)
- Basic agents (LLM + Tools)

**Limitation**: LangChain works great for **linear / simple chains**, but struggles with **complex, non-linear, long-running workflows**.

---

### 2. The Complex Workflow Example – Automated Hiring System

We took the **same HR Recruiter Agentic AI example** from Video 2 and created a detailed **flowchart**.

**Key Steps in the Hiring Workflow**:
1. Receive hiring request (Backend Engineer, Remote, 2-4 years exp)
2. Create Job Description (JD) using LLM
3. Human approval of JD (Human-in-the-Loop)
4. Post JD on LinkedIn / Naukri using APIs
5. Wait 7 days → Monitor applications
6. If applications < threshold (e.g., 20):
   - Modify JD (e.g., change to Full Stack, lower experience)
   - Wait 48 hours → Monitor again (Loop)
7. If enough applications:
   - Parse & Shortlist resumes using LLM + Resume Parser tool
   - Schedule interviews (Calendar + Email APIs)
8. Conduct interviews → Decision (Select / Reject)
9. Send Offer Letter → Negotiation if needed
10. Onboarding (Welcome email, IT access, Laptop, KT sessions)

**Important Note**:  
This flowchart is a **predefined workflow** (created by developer), **not** a true Agent.  
- In a real **Agentic AI**, the LLM would dynamically decide the steps and order.  
- Here the flow is **static** and fixed by the developer → This is called a **Workflow**, not a full Agent.

---

### 3. Challenges of Building This Complex Workflow with LangChain

The speaker highlights **8 major challenges** when trying to implement the above complex hiring workflow using only LangChain:

1. **Control Flow Complexity** (Non-linear flow)
   - Conditional branches (if/else)
   - Loops (keep modifying JD until enough applications)
   - Jumps (go back to previous steps)
   - LangChain chains are mostly **linear**. You end up writing a lot of custom Python "glue code" (while loops, if-else, manual stitching).

2. **State Management**
   - Complex workflows need to track many data points (JD text, approval status, application count, shortlisted candidates, offer status, etc.).
   - LangChain is mostly **stateless**. You have to manually manage state using dictionaries → very error-prone and hard to maintain.

3. **Event-Driven Execution**
   - Need to **pause** workflow for days (wait 7 days after posting JD, wait for candidate to accept offer, etc.).
   - LangChain is designed for **sequential execution** (runs from start to finish without pausing). Not suitable for long-running, event-driven flows.

4. **Fault Tolerance**
   - Long-running workflows can fail (API down, server crash, etc.).
   - LangChain has no built-in retry or recovery mechanism. If it fails midway, you usually have to start from the beginning.

5. **Human-in-the-Loop**
   - Need human approval at multiple points (JD approval, etc.).
   - LangChain doesn't support long pauses (hours/days) for human input gracefully. Script would hang or require complex splitting of chains.

6. **Nested Workflows (Subgraphs)**
   - Some steps (e.g., "Conduct Interview") are themselves complex workflows.
   - LangChain doesn't support nesting workflows easily.

7. **Observability / Debugging**
   - Hard to monitor and debug because of heavy glue code.
   - LangSmith can track LangChain parts but not custom Python glue code.

8. **Maintainability**
   - Too much glue code → difficult to maintain, debug, and work in teams.

---

### 4. How LangGraph Solves These Problems

**What is LangGraph?**  
LangGraph is an orchestration framework (built on top of LangChain) that lets you build **stateful, multi-step, event-driven workflows** using graphs.

**Core Idea**:
- Represent the entire workflow as a **Graph**.
- Each task/step = **Node** (simple Python function).
- Connections & control flow = **Edges**.
- Conditional logic, loops, branching = **Conditional Edges**.

**Key Advantages of LangGraph**:

- **Graph Representation**: Naturally handles non-linear, complex flows.
- **Stateful Execution**: Built-in state object (Pydantic or TypedDict) accessible and mutable by every node.
- **Event-Driven & Checkpoints**: Can pause indefinitely and resume later (perfect for waiting 7 days or human approval).
- **Built-in Fault Tolerance**: Retry logic for small failures + Recovery using checkpoints for big failures (server crash).
- **Human-in-the-Loop**: First-class support – pause for human input (minutes to days) using checkpoints.
- **Nested Workflows (Subgraphs)**: One node can be an entire subgraph → supports multi-agent systems and reusability.
- **Excellent Observability**: Tight integration with LangSmith – full timeline of every node, state changes, decisions.

**When to use LangChain vs LangGraph**

| Use Case                              | Recommended Tool     | Reason |
|---------------------------------------|----------------------|--------|
| Simple linear chains                  | LangChain            | Easy and sufficient |
| Basic RAG, summarization, chatbots    | LangChain            | Straightforward |
| Complex non-linear workflows          | LangGraph            | Handles branches, loops, state |
| Long-running / event-driven flows     | LangGraph            | Checkpoints + pause/resume |
| Human-in-the-loop with long waits     | LangGraph            | Native support |
| Multi-agent systems                   | LangGraph            | Subgraphs |
| Need high maintainability & debugging | LangGraph            | Graph + LangSmith |

**Important Clarification**:
- **LangGraph does NOT replace LangChain**.
- LangGraph is **built on top of LangChain**.
- You still use LangChain components (Models, Prompts, Retrievers, Tools, etc.).
- LangGraph is used for **orchestrating** / connecting them in complex ways.

---

### Final Summary

- **LangChain** → Great for simple, linear LLM applications and basic chains.
- **LangGraph** → Designed for building **robust, stateful, complex, production-grade Agentic AI workflows**.
- LangGraph gives you: Graph structure, Stateful execution, Checkpoints, Conditional edges, Subgraphs, Fault tolerance, Human-in-the-loop, and excellent observability.

**Recommendation from Video**:  
Master both. Use LangChain components + LangGraph for orchestration when building real Agentic AI systems.

---

**Notes prepared from Video Transcript**  
**Playlist**: Agentic AI using LangGraph  
**Video 3**: Introduction to LangGraph & LangChain vs LangGraph  
*March 2026*

---

You can now copy the entire content above and save it as `README.md`. It is clean, well-organized, and ready for GitHub or any Markdown viewer.

Would you like me to create a combined README that merges Video 2 + Video 3 notes into one file? Just say the word!