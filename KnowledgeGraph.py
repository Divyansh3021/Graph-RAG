import networkx as nx
from pyvis.network import Network
import json

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_entity(self, entity):
        self.graph.add_node(entity)

    def add_relationship(self, entity1, entity2, relationship):
        self.graph.add_edge(entity1, entity2, relationship=relationship)

    def visualize(self, filename='graph.html'):
        net = Network(notebook=False)
        net.from_nx(self.graph)
        net.show(filename, notebook=False)

    def save(self, filename='knowledge_graph.json'):
        data = nx.node_link_data(self.graph)
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load(self, filename='knowledge_graph.json'):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

# Usage
kg = KnowledgeGraph()
kg.add_entity("Paris")
kg.add_entity("France")
kg.add_relationship("Paris", "France", "capital of")
kg.save()
