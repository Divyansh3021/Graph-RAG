import networkx as nx
from pyvis.network import Network
import json


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Changed to DiGraph for directed graph

    def add_entity(self, entity):
        self.graph.add_node(entity)

    def add_relationship(self, entity1, entity2, relationship):
        self.graph.add_edge(entity1, entity2, label=relationship)  # Store label instead of relationship

    def visualize(self, filename='graph.html'):
        net = Network(notebook=False, directed=True)  # Set directed=True
        net.from_nx(self.graph)
        for edge in net.edges:
            edge['label'] = edge['label']  # Ensure label is displayed
        net.show(filename, notebook=False)

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
        for node in data['nodes']:
            self.add_entity(node['label'])
        
        # Add edges
        for edge in data['edges']:
            source = next(node['label'] for node in data['nodes'] if node['id'] == edge['source'])
            target = next(node['label'] for node in data['nodes'] if node['id'] == edge['target'])
            self.add_relationship(source, target, edge['label'])

