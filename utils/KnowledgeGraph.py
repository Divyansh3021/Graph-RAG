import networkx as nx
from pyvis.network import Network
import json


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Changed to DiGraph for directed graph

    def add_entity(self, label, file_name=None, line_numbers=None):
        self.graph.add_node(label, file_name=file_name, line_numbers=line_numbers)

    def add_relationship(self, entity1, entity2, relationship):
        self.graph.add_edge(entity1, entity2, label=relationship)  # Store label instead of relationship

    def visualize(self, filename='graph.html'):
        net = Network(notebook=True, directed=True)
        for node, data in self.graph.nodes(data=True):
            net.add_node(node, label=f"{node}")
        for edge in self.graph.edges(data=True):
            net.add_edge(edge[0], edge[1], label=edge[2]['label'])
        net.show(filename)

    def save(self, filename='knowledge_graph.json'):
        data = nx.node_link_data(self.graph)
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load(self, filename='knowledge_graph.json'):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True)  # Ensure loaded graph is directed

    def load_from_dict(self, data):
        
        # Add nodes
        for node_info in data['nodes']:
            label, file_name, line_numbers = node_info
            self.add_entity(label, file_name=file_name, line_numbers=line_numbers)
        
        # Add edges
        for edge_info in data['edges']:
            source, target, relationship = edge_info
            self.add_relationship(source, target, relationship)
