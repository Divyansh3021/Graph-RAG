import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict
from llama_index.llms.langchain import LangChainLLM

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
from typing import List, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import OpenAI

# Define the entity and relation data models
class Entity(BaseModel):
    """Entity extracted from the text."""
    name: str = Field(description="The name of the entity")

class Relation(BaseModel):
    """Relation between entities extracted from the text."""
    entity1: str = Field(description="The first entity in the relationship")
    relationship: str = Field(description="The type of relationship")
    entity2: str = Field(description="The second entity in the relationship")

class ExtractedData(BaseModel):
    """Structured output containing entities and relations."""
    entities: List[Entity] = Field(description="List of entities")
    relations: List[Relation] = Field(description="List of relations")


# Configure the LLM to produce structured output
structured_llm = llm.with_structured_output(ExtractedData)

# Define the extraction prompts
extraction_prompt = """
Extract entities and relationships from the following text. Return the entities and relationships in a structured format.

Text: {text}

Entities and Relationships:
"""

# Define the function to invoke the structured LLM
def extract_entities_and_relations(text, structured_llm):
    prompt_text = extraction_prompt.format(text=text)
    response = structured_llm.invoke(prompt_text)
    return response

# Example text content
text_content = """
Alice is a software engineer at Google. She works with Bob, who is a data scientist. They both live in San Francisco.
"""

# Extract entities and relations
extracted_data = extract_entities_and_relations(text_content, llm)


# Print the extracted entities and relations
# print("Entities:")
# for entity in extracted_data.entities:
#     print(entity.name)

# print("\nRelations:")
# for relation in extracted_data.relations:
#     print(f"{relation.entity1} {relation.relationship} {relation.entity2}")
