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
    try:
        res_dict = output_parser.parse(new_res.content)
    
    except AttributeError or Exception:
        print("Exception occured!!")
        res = llm.invoke("""
            this JSON data is generating error while parsing, format it according as this example:

            Example:       
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
                              
            JSON data:
            

        """+new_res.content)
        print(res)
        res_dict = output_parser.parse(res)
    return res_dict
