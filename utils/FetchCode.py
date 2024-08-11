import os

folders_to_remove = [
    '.git',
    'node_modules',
    'site-packages', 
    'target',
    'gems',
    'vendor',
    'bin',
    'obj',
    'migrations',
    'lib',
    '__pycache__'
]

def get_all_files_paths(directory = "./"):
        file_paths = []
        # print("file paths")
        for root, dirs, files in os.walk(directory):
            for folder in folders_to_remove:
                if folder in dirs:
                    dirs.remove(folder)
            for file_name in files:
                if not file_name.endswith(".ipynb"):
                    file_paths.append(os.path.join(root, file_name))
        return file_paths

file_paths = get_all_files_paths("C:/Code/folder2")

print(file_paths)

from ExtractEntities import get_nodes_edge

graph_dict = {'nodes': [], 'edges': []}

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        code = [f"{idx+1}: {line}" for idx, line in enumerate(f.readlines())]
        res_dict = get_nodes_edge(code="\n".join(code), source = file_path)
        print("processing: ", file_path)
        # print(res_dict, "\n\n\n")
        graph_dict['nodes'].extend(res_dict['nodes'])
        graph_dict['edges'].extend(res_dict['edges'])
        
from KnowledgeGraph import KnowledgeGraph

kg = KnowledgeGraph()
kg.load_from_dict(graph_dict)
kg.save()
kg.visualize()


from GraphRetriever import GraphRetriever, GraphEmbedding
from ReadLine import read_lines

ge = GraphEmbedding()
node_embeddings = ge.embed_and_save_graph(kg.graph)

retriever = GraphRetriever(kg, node_embeddings)
result = retriever.retrieve("What is the use of load_and_split function?", k=4)

for info in result:
    print(f"Node: {info['node']}")
    print(f"File: {info['file_name']}")
    print(f"Line numbers: {info['line_numbers']}")
    print("Outgoing edges:")
    for edge in info['outgoing_edges']:
        print(f"  {edge}")
    print("Incoming edges:")
    for edge in info['incoming_edges']:
        print(f"  {edge}")
    print()
    print("code lines: \n\n")
    print(read_lines(info['file_name'], info['line_numbers']))
    print()
#provide these files path to a function which will call extractEntities.py for each file.