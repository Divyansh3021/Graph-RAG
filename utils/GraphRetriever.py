import faiss
from GraphEmedding import GraphEmbedding
import numpy as np

class GraphRetriever:
    def __init__(self, knowledge_graph, embeddings):
        self.kg = knowledge_graph
        self.embeddings = embeddings
        self.index = None
        self.nodes = list(embeddings.keys())
        self.build_index()

    def build_index(self):
        vectors = np.array(list(self.embeddings.values())).astype('float32')
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    def retrieve(self, query, k=1):
        ge = GraphEmbedding()
        query_vector = ge.embed_node(query).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        retrieved_nodes = [self.nodes[i] for i in indices[0]]
        
        retrieved_info = []
        for node in retrieved_nodes:
            # Get node attributes
            node_data = self.kg.graph.nodes[node]
            
            # Get outgoing edges
            outgoing = list(self.kg.graph.out_edges(node, data=True))
            outgoing_info = [f"{node} --({edge[2]['label']})-> {edge[1]}" for edge in outgoing]
            
            # Get incoming edges
            incoming = list(self.kg.graph.in_edges(node, data=True))
            incoming_info = [f"{edge[0]} --({edge[2]['label']})-> {node}" for edge in incoming]
            
            retrieved_info.append({
                'node': node,
                'file_name': node_data.get('file_name'),
                'line_numbers': node_data.get('line_numbers'),
                'outgoing_edges': outgoing_info,
                'incoming_edges': incoming_info
            })
        
        return retrieved_info

# Usage example:
