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
        Extract the details from the following code and format them as a JSON object. Each class should be a node with its methods as nodes connected by edges. Each method should have nodes for its attributes and return values. Each edge should describe the relationship between nodes.

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


        """
    
    prompt += "```\n" + code + "\n```"

    new_res = llm.invoke(prompt)
    return output_parser.parse(new_res.content)

code = """
def validate_environment(cls, values: Dict) -> Dict:
        '''Validates params and passes them to google-generativeai package.'''
        if values.get("credentials"):
            genai.configure(
                credentials=values.get("credentials"),
                transport=values.get("transport"),
                client_options=values.get("client_options"),
            )
        else:
            google_api_key = get_from_dict_or_env(
                values, "google_api_key", "GOOGLE_API_KEY"
            )
            if isinstance(google_api_key, SecretStr):
                google_api_key = google_api_key.get_secret_value()
            genai.configure(
                api_key=google_api_key,
                transport=values.get("transport"),
                client_options=values.get("client_options"),
            )

        model_name = values["model"]

        safety_settings = values["safety_settings"]

        if safety_settings and (
            not GoogleModelFamily(model_name) == GoogleModelFamily.GEMINI
        ):
            raise ValueError("Safety settings are only supported for Gemini models")

        if GoogleModelFamily(model_name) == GoogleModelFamily.GEMINI:
            values["client"] = genai.GenerativeModel(
                model_name=model_name, safety_settings=safety_settings
            )
        else:
            values["client"] = genai

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if values["max_output_tokens"] is not None and values["max_output_tokens"] <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        if values["timeout"] is not None and values["timeout"] <= 0:
            raise ValueError("timeout must be greater than zero")

        return values
"""
node_edge = get_nodes_edge(code=code)

print(node_edge)