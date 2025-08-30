# Ex.No: 2  Implementation of Depth First Search
### DATE: 30/08/2025                                                                           
### REGISTER NUMBER : 212223060108
### AIM: 
To write a python program to implement Depth first Search. 
### Algorithm:
1. Start the program
2. Create the graph by using adjacency list representation
3. Define a function dfs and take the set “visited” is empty 
4. Search start with initial node. Check the node is not visited then print the node.
5. For each neighbor node, recursively invoke the dfs search.
6. Call the dfs function by passing arguments visited, graph and starting node.
7. Stop the program.
### Program:
```
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)

# Example Graph (Adjacency List)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Run DFS
print("DFS Traversal:", end=" ")
dfs(graph, 'A')
```

### Output:

<img width="823" height="157" alt="image" src="https://github.com/user-attachments/assets/da8ec995-d300-47a1-99ba-8a6106d572bc" />


### Result:
Thus the depth first search order was found sucessfully.
