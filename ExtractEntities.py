from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0,
    max_tokens=None,
    timeout=15,
    max_retries=2
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

        For each node, include the file name and line numbers where it appears in the code. For edges, use the actual names of nodes as source and target, not IDs.

        For Example:

        Code:

        from sentence_transformers import SentenceTransformer
        import numpy as np
        from KnowledgeGraph import kg

        class GraphEmbedding:
        
            def init(self, model_name='all-MiniLM-L6-v2'):
                self.model = SentenceTransformer(model_name)

            def embed_node(self, node):
                return self.model.encode(node)

            def embed_graph(self, graph):
                embeddings = {}
                for node in graph.nodes():
                    embeddings[node] = self.embed_node(node)
                return embeddings
        
        ge = GraphEmbedding()
        node_embeddings = ge.embed_graph(kg.graph)

        Output:
        ```json
        {
        "nodes": [
            {"label": "GraphEmbedding", "file_name": "example.py", "line_numbers": "4-14"},
            {"label": "__init__", "file_name": "example.py", "line_numbers": "5-6"},
            {"label": "model_name", "file_name": "example.py", "line_numbers": "5"},
            {"label": "model", "file_name": "example.py", "line_numbers": "6"},
            {"label": "embed_node", "file_name": "example.py", "line_numbers": "8-9"},
            {"label": "node", "file_name": "example.py", "line_numbers": "8"},
            {"label": "model.encode(node)", "file_name": "example.py", "line_numbers": "9"},
            {"label": "embed_graph", "file_name": "example.py", "line_numbers": "11-14"},
            {"label": "graph", "file_name": "example.py", "line_numbers": "11"},
            {"label": "embeddings", "file_name": "example.py", "line_numbers": "12"},
            {"label": "graph.nodes()", "file_name": "example.py", "line_numbers": "13"},
            {"label": "embed_node(node)", "file_name": "example.py", "line_numbers": "13"}
        ],
        "edges": [
            {"source": "GraphEmbedding", "target": "__init__", "label": "has method"},
            {"source": "__init__", "target": "model_name", "label": "has parameter"},
            {"source": "__init__", "target": "model", "label": "has attribute"},
            {"source": "GraphEmbedding", "target": "embed_node", "label": "has method"},
            {"source": "embed_node", "target": "node", "label": "has parameter"},
            {"source": "embed_node", "target": "model.encode(node)", "label": "returns"},
            {"source": "GraphEmbedding", "target": "embed_graph", "label": "has method"},
            {"source": "embed_graph", "target": "graph", "label": "has parameter"},
            {"source": "embed_graph", "target": "embeddings", "label": "has attribute"},
            {"source": "embed_graph", "target": "graph.nodes()", "label": "uses"},
            {"source": "embed_graph", "target": "embed_node(node)", "label": "calls"}
        ]
        }
        ```

        Important Notes:

        - For each node, include "file_name" and "line_numbers" attributes.
        - In the "edges" array, use the actual node labels for "source" and "target", not IDs.
        - Ensure that the relationships described in the edges accurately reflect the code structure.
        - Be as detailed and accurate as possible in identifying nodes and their relationships.


        Code Language: Python

        Source File: Raptor_index.py
        
        Given Code:

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
                {"label": "GraphEmbedding", "file_name": "example.py", "line_numbers": "4-14"},
                {"label": "__init__", "file_name": "example.py", "line_numbers": "5-6"},
                {"label": "model_name", "file_name": "example.py", "line_numbers": "5"},
                {"label": "model", "file_name": "example.py", "line_numbers": "6"},
                {"label": "embed_node", "file_name": "example.py", "line_numbers": "8-9"},
                {"label": "node", "file_name": "example.py", "line_numbers": "8"},
                {"label": "model.encode(node)", "file_name": "example.py", "line_numbers": "9"},
                {"label": "embed_graph", "file_name": "example.py", "line_numbers": "11-14"},
                {"label": "graph", "file_name": "example.py", "line_numbers": "11"},
                {"label": "embeddings", "file_name": "example.py", "line_numbers": "12"},
                {"label": "graph.nodes()", "file_name": "example.py", "line_numbers": "13"},
                {"label": "embed_node(node)", "file_name": "example.py", "line_numbers": "13"}
            ],
            "edges": [
                {"source": "GraphEmbedding", "target": "__init__", "label": "has method"},
                {"source": "__init__", "target": "model_name", "label": "has parameter"},
                {"source": "__init__", "target": "model", "label": "has attribute"},
                {"source": "GraphEmbedding", "target": "embed_node", "label": "has method"},
                {"source": "embed_node", "target": "node", "label": "has parameter"},
                {"source": "embed_node", "target": "model.encode(node)", "label": "returns"},
                {"source": "GraphEmbedding", "target": "embed_graph", "label": "has method"},
                {"source": "embed_graph", "target": "graph", "label": "has parameter"},
                {"source": "embed_graph", "target": "embeddings", "label": "has attribute"},
                {"source": "embed_graph", "target": "graph.nodes()", "label": "uses"},
                {"source": "embed_graph", "target": "embed_node(node)", "label": "calls"}
            ]
            }
                              
            JSON data:
            

        """+new_res.content)
        print(res)
        res_dict = output_parser.parse(res)
    return res_dict

code = """

def recursive_raptor(embeddings, summaries, model, level=1, max_levels=3):
    if level > max_levels or len(embeddings) <= 1:
        return summaries

    n_clusters = min(5, len(embeddings))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(embeddings)
    probabilities = gmm.predict_proba(embeddings)

    cluster_texts = [[] for _ in range(n_clusters)]
    for i, embedding in enumerate(embeddings):
        cluster_idx = np.argmax(probabilities[i])
        cluster_texts[cluster_idx].append(i)

    new_embeddings = []
    new_summaries = []
    for indices in cluster_texts:
        if indices:
            texts = [summaries[i] for i in indices]
            summary = summarize(texts)
            new_summaries.append(summary)
            new_embedding = model.encode([summary])[0]
            new_embeddings.append(new_embedding)

    return new_summaries

hierarchical_summaries = recursive_raptor(summary_embeddings, summaries, model)
print(hierarchical_summaries)

"""

graph_dict = get_nodes_edge(code)
print(graph_dict)