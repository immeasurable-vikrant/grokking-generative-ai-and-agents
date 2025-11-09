# Output Parsers in LangChain  
**Generative AI using LangChain | Video 6 | CampusX**  
*📝 Comprehensive Notes with Examples & Explanations* 

---

## 1. Introduction to Output Parsers 🔑

- **What are Output Parsers?**  
  Output parsers are classes in LangChain that process LLM responses, extracting structured data from unstructured text. They ensure consistency, validation, and ease of use in applications.  
  > **Elaboration**: LLMs output raw text (unstructured). Parsers "parse" this into usable formats (e.g., strings, JSON). They're essential for integrating LLMs with systems like databases/APIs. Analogy: Like a translator converting messy speech into a formatted report.

- **Why Important?**  
  - Handles models that don't natively support structured output (e.g., open-source LLMs).  
  - Builds on "Structured Output" from Video 5 (recap below).  
  - Enables chaining (pipelines) for complex workflows.  
  - Works with **any LLM** (e.g., OpenAI, Hugging Face, local models).

- **Disclaimer**: This video builds heavily on Video 5 (Structured Output). Watch it first for context.

---

## 2. Recap from Video 5 (Structured Output) 🔄

| Concept | Description |
|---------|-------------|
| **Unstructured Output** | Default LLM responses: Textual, hard to integrate (e.g., metadata + content). |
| **Structured Output** | Forces LLM to return formatted data (e.g., JSON) via schemas. Benefits: Machine-readable for databases/APIs. |
| **Model Types** | - **Can**: Natively support (e.g., OpenAI GPT – use `with_structured_output`).<br>- **Can't**: Don't support (e.g., open-source like TinyLlama – need parsers). |

> **Elaboration**: Structured output lets LLMs "talk" to machines. Without it, responses are like free-form essays—useful for humans but not automation. Pitfall: Free APIs (e.g., Hugging Face) can timeout; use reliable ones like OpenAI.

---

## 3. Types of Output Parsers 📊

LangChain has many parsers (e.g., CSV, List, XML, Datetime), but focus on the 4 most common:  
- **String Output Parser**: Simplest – extracts plain text.  
- **JSON Output Parser**: Forces JSON format (no schema enforcement).  
- **Structured Output Parser**: Enforces a predefined schema in JSON.  
- **Pydantic Output Parser**: Enforces schema + data validation (most powerful).  

> **Table: Quick Comparison**  

| Parser                  | Key Feature                          | Schema Enforcement | Data Validation | Best For |
|-------------------------|--------------------------------------|--------------------|-----------------|----------|
| **String** 📄          | Extracts text from response.        | No                 | No              | Simple chains/pipelines. |
| **JSON** 🔗             | Forces JSON output.                 | No                 | No              | Basic JSON needs. |
| **Structured** 🛠️      | Enforces custom schema in JSON.     | Yes                | No              | Structured data without validation. |
| **Pydantic** ✅         | Schema + validation/coercion.       | Yes                | Yes             | Robust apps (e.g., age > 18 as int). |

> **Elaboration**: Choose based on needs—start simple (String/JSON), escalate for control (Structured/Pydantic). All work with chains for pipelines.

---

## 4. String Output Parser 📄

- **Purpose**: Converts LLM response (with metadata) to a plain string. Ideal for chains where you need clean text for next steps.  
  > **Elaboration**: Avoids manual `result.content`. Great for multi-step flows (e.g., generate report → summarize). Pitfall: Free APIs may timeout—test with OpenAI.

- **Use Case**: Generate detailed report on topic (e.g., black hole), then summarize to 5 lines.  

**Example Code** (with Chain – cleaner for pipelines):  
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()

template1 = PromptTemplate.from_template("Write a detailed report on {topic}")
template2 = PromptTemplate.from_template("Write a five-line summary on the following text:\n{text}")

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "black hole"})

print(result)  # Outputs: 5-line summary as string
```

> **Elaboration**: Chain (`|`) pipes steps: Prompt1 → Model → Parse (text) → Prompt2 → Model → Parse. Extension: Use with Hugging Face (e.g., Gemma model) for open-source.

## 5. JSON Output Parser 🔗

- **Purpose**: Forces LLM to return JSON. No schema enforcement—LLM decides structure.  
  > **Elaboration**: Simple for JSON needs, but unpredictable (e.g., facts as list vs. keys). Analogy: Asking for a box (JSON) without specifying compartments (schema).

- **Use Case**: Get name, age, city of fictional person.  

**Example Code** (with Chain):  
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOpenAI()
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})

print(result)  # Outputs: {'name': '...', 'age': 30, 'city': '...'} as dict
```

> **Elaboration**: `get_format_instructions()` injects JSON prompt. Pitfall: No schema—e.g., facts might come as array, not keyed objects. Extension: Use for APIs expecting JSON.

---

## 6. Structured Output Parser 🛠️

- **Purpose**: Enforces a predefined schema in JSON (no validation).  
  > **Elaboration**: Builds on JSON parser but adds structure. Good for consistent formats. Pitfall: No type checks (e.g., age as string "thirty" accepted).

- **Use Case**: Get 3 facts about a topic (e.g., black hole) as keyed JSON.  

**Example Code** (with Chain):  
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

model = ChatOpenAI()

response_schemas = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = PromptTemplate(
    template="Give 3 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "black hole"})

print(result)  # Outputs: {'fact_1': '...', 'fact_2': '...', 'fact_3': '...'}
```

> **Elaboration**: `ResponseSchema` defines structure. Extension: Combine with agents for tool inputs.

---

## 7. Pydantic Output Parser ✅

- **Purpose**: Enforces schema + validates data (e.g., types, constraints). Uses Pydantic for safety.  
  > **Elaboration**: Most robust—coerces types (e.g., "32" → 32), enforces rules (e.g., age > 18). Analogy: A form with required fields and checks.

- **Use Case**: Get name, age (int > 18), city for fictional person.  

**Example Code** (with Chain):  
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City the person belongs to")

model = ChatOpenAI()
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person.\n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"place": "Sri Lankan"})

print(result)  # Outputs: Pydantic object (e.g., name='...', age=25, city='...')
```

> **Elaboration**: Validates/errors on invalid data. Extension: Add more fields (e.g., email with `EmailStr`).

---

## 8. Other Parsers & Tips 🔍

- **Others**: CSV, List, Markdown List, XML, Datetime, Output Fixing (retries failed parses).  
- **Tips**:  
  - Use chains for cleaner code.  
  - `get_format_instructions()` auto-injects prompts.  
  - Test with different LLMs (e.g., Hugging Face for open-source).  
  - Study docs for niche parsers.

---

## Conclusion 🎯

> **Output Parsers = Key to Usable LLM Outputs**  

You've learned:  
- Fundamentals & types.  
- Code for each with chains.  
- When to use (e.g., Pydantic for validation).  

Next: Dive into chains/agents. Practice on your machine!  

---
