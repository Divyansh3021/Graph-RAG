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
                file_paths.append(os.path.join(root, file_name))
        return file_paths

files = get_all_files_paths("C:/Code/Graph RAG")
print(files)