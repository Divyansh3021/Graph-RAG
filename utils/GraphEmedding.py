import json
import numpy as np
from sentence_transformers import SentenceTransformer

class GraphEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_node(self, node):
        return self.model.encode(node)

    def embed_graph(self, graph):
        embeddings = {}
        for node in graph.nodes():
            embeddings[node] = self.embed_node(node).tolist()  # Convert to list for JSON serialization
        return embeddings

    def save_embeddings(self, embeddings, filename='embeddings.json'):
        with open(filename, 'w') as f:
            json.dump(embeddings, f)

    def load_embeddings(self, filename='embeddings.json'):
        with open(filename, 'r') as f:
            embeddings = json.load(f)
        # Convert lists back to numpy arrays
        return {k: np.array(v) for k, v in embeddings.items()}

    def embed_and_save_graph(self, graph, filename='embeddings.json'):
        embeddings = self.embed_graph(graph)
        self.save_embeddings(embeddings, filename)
        return embeddings