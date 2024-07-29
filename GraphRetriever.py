import faiss
from GraphEmedding import node_embeddings, ge
import numpy as np

class GraphRetriever:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.index = None
        self.nodes = list(embeddings.keys())
        self.build_index()

    def build_index(self):
        vectors = np.array(list(self.embeddings.values())).astype('float32')
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    def retrieve(self, query, k=1):
        query_vector = ge.embed_node(query).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        return [self.nodes[i] for i in indices[0]]

# Usage
retriever = GraphRetriever(node_embeddings)
relevant_nodes = retriever.retrieve("What is the capital of France?")
print(relevant_nodes)