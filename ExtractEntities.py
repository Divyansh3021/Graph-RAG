from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    tiemout = 15
)

nodes_schema = ResponseSchema(name="nodes",
                                   description="List of nodes",
                                   type="List")

edges_schema = ResponseSchema(name="edges",
                                   description="List of edges",
                                   type="List")

response_schemas = [nodes_schema, edges_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)

def get_nodes_edge(code):

    prompt = """
        Extract the details from the following code according to the specified code language and format them as a JSON object. Each class should be a node with its methods as nodes connected by edges. Each method should have nodes for its attributes and return values. Each edge should describe the relationship between nodes.

        For Example:

        Code :
        ``` 
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from KnowledgeGraph import kg

        class GraphEmbedding:
            def __init__(self, model_name='all-MiniLM-L6-v2'):
                self.model = SentenceTransformer(model_name)

            def embed_node(self, node):
                return self.model.encode(node)

            def embed_graph(self, graph):
                embeddings = {}
                for node in graph.nodes():
                    embeddings[node] = self.embed_node(node)
                return embeddings

        # Usage
        ge = GraphEmbedding()
        node_embeddings = ge.embed_graph(kg.graph)
        ```

        Output:
        ```json
        {
        "nodes": [
            {"id": "1", "label": "GraphEmbedding"},
            {"id": "2", "label": "__init__"},
            {"id": "3", "label": "model_name"},
            {"id": "4", "label": "model"},
            {"id": "5", "label": "embed_node"},
            {"id": "6", "label": "node"},
            {"id": "7", "label": "model.encode(node)"},
            {"id": "8", "label": "embed_graph"},
            {"id": "9", "label": "graph"},
            {"id": "10", "label": "embeddings"},
            {"id": "11", "label": "graph.nodes()"},
            {"id": "12", "label": "embed_node(node)"}
        ],
        "edges": [
            {"source": "1", "target": "2", "label": "has method"},
            {"source": "2", "target": "3", "label": "has parameter"},
            {"source": "2", "target": "4", "label": "has attribute"},
            {"source": "1", "target": "5", "label": "has method"},
            {"source": "5", "target": "6", "label": "has parameter"},
            {"source": "5", "target": "7", "label": "returns"},
            {"source": "1", "target": "8", "label": "has method"},
            {"source": "8", "target": "9", "label": "has parameter"},
            {"source": "8", "target": "10", "label": "has attribute"},
            {"source": "8", "target": "11", "label": "has attribute"},
            {"source": "8", "target": "12", "label": "returns"}
        ]
        }
        ```

        Given Code:

        Code Language: Typescript
        """
    
    prompt += "```\n" + code + "\n```"

    new_res = llm.invoke(prompt)
    return output_parser.parse(new_res.content)

code = """

import prisma from "@calcom/prisma";

import type { UserList } from "../types/user";

/*
 * Extracts usernames (@Example) and emails (hi@example.com) from a string
 */
export const extractUsers = async (text: string) => {
  const usernames = text
    .match(/(?<![a-zA-Z0-9_.])@[a-zA-Z0-9_]+/g)
    ?.map((username) => username.slice(1).toLowerCase());
  const emails = text
    .match(/[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+/g)
    ?.map((email) => email.toLowerCase());

  const dbUsersFromUsernames = usernames
    ? await prisma.user.findMany({
        select: {
          id: true,
          username: true,
          email: true,
        },
        where: {
          username: {
            in: usernames,
          },
        },
      })
    : [];

  const usersFromUsernames = usernames
    ? usernames.map((username) => {
        const user = dbUsersFromUsernames.find((u) => u.username === username);
        return user
          ? {
              username,
              id: user.id,
              email: user.email,
              type: "fromUsername",
            }
          : {
              username,
              id: null,
              email: null,
              type: "fromUsername",
            };
      })
    : [];

  const dbUsersFromEmails = emails
    ? await prisma.user.findMany({
        select: {
          id: true,
          email: true,
          username: true,
        },
        where: {
          email: {
            in: emails,
          },
        },
      })
    : [];

  const usersFromEmails = emails
    ? emails.map((email) => {
        const user = dbUsersFromEmails.find((u) => u.email === email);
        return user
          ? {
              email,
              id: user.id,
              username: user.username,
              type: "fromEmail",
            }
          : {
              email,
              id: null,
              username: null,
              type: "fromEmail",
            };
      })
    : [];

  return [...usersFromUsernames, ...usersFromEmails] as UserList;
};

"""
node_edge = get_nodes_edge(code=code)

print(node_edge+"\n\n\n\n")