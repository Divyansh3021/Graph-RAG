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