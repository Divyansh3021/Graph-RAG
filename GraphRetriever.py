import faiss
from GraphEmedding import node_embeddings, GraphEmbedding
from KnowledgeGraph import kg
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
            # Get outgoing edges
            outgoing = list(self.kg.graph.out_edges(node, data=True))
            outgoing_info = [f"{node} --({edge[2]['label']})-> {edge[1]}" for edge in outgoing]
            
            # Get incoming edges
            incoming = list(self.kg.graph.in_edges(node, data=True))
            incoming_info = [f"{edge[0]} --({edge[2]['label']})-> {node}" for edge in incoming]
            
            retrieved_info.append({
                'node': node,
                'outgoing_edges': outgoing_info,
                'incoming_edges': incoming_info
            })
        
        return retrieved_info


ge = GraphEmbedding()
node_embeddings = {node: ge.embed_node(node) for node in kg.graph.nodes()}

retriever = GraphRetriever(kg, node_embeddings)
result = retriever.retrieve("What is AgentExecutor?", k=10)  # Retrieve info for top 2 most relevant nodes

for info in result:
    print(f"Node: {info['node']}")
    print("Outgoing edges:")
    for edge in info['outgoing_edges']:
        print(f"  {edge}")
    print("Incoming edges:")
    for edge in info['incoming_edges']:
        print(f"  {edge}")
    print()