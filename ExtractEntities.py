import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class Graph(BaseModel):
    """Representation of the graph as an adjacency matrix."""
    matrix: str = Field(description="The created adjacency matrix")

structured_llm = llm.with_structured_output(Graph)


prompt = """
You are an expert system designed to extract relationships between entities in text and represent them as an adjacency matrix. Your task is to:

1. Identify key entities (nodes) in the given text.
2. Determine relationships between these entities.
3. Create an adjacency matrix where:
   - Rows and columns represent the entities (nodes).
   - Matrix elements represent the relationships between entities.
   - Use 0 if there's no direct relationship.
   - Use a word or short phrase to describe the relationship if one exists.

Format your response as a string representing the adjacency matrix, with entities as labels and relationships in the cells. Each row should be on a new line, and cells should be separated by tabs.

Now, analyze the following text and provide the adjacency matrix:

Content:
Dragon Ball (Japanese: ドラゴンボール, Hepburn: Doragon Bōru) is a Japanese media franchise created by Akira Toriyama in 1984. The initial manga, written and illustrated by Toriyama, was serialized in Weekly Shōnen Jump from 1984 to 1995, with the 519 individual chapters collected in 42 tankōbon volumes by its publisher Shueisha. Dragon Ball was originally inspired by the classical 16th-century Chinese novel Journey to the West, combined with elements of Hong Kong martial arts films. Dragon Ball characters also use a variety of East Asian martial arts styles, including karate[1][2][3] and Wing Chun (kung fu).[2][3][4] The series follows the adventures of protagonist Son Goku from his childhood through adulthood as he trains in martial arts. He spends his childhood far from civilization until he meets a teen girl named Bulma, who encourages him to join her quest in exploring the world in search of the seven orbs known as the Dragon Balls, which summon a wish-granting dragon when gathered. Along his journey, Goku makes several other friends, becomes a family man, discovers his alien heritage, and battles a wide variety of villains, many of whom also seek the Dragon Balls.

Remember to use actual entity names and relationship descriptions in your matrix, not just letters and numbers.
"""

res = llm.invoke(prompt)
print(res)