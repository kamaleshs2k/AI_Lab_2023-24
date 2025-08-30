# Ex.No: 1  Implementation of Breadth First Search 
### DATE:                                                                            
### REGISTER NUMBER : 
### AIM: 
To write a python program to implement Breadth first Search. 
### Algorithm:
1. Start the program
2. Create the graph by using adjacency list representation
3. Define a function bfs and take the set “visited” is empty and “queue” is empty
4. Search start with initial node and add the node to visited and queue.
5. For each neighbor node, check node is not in visited then add node to visited and queue list.
6.  Creating loop to print the visited node.
7.   Call the bfs function by passing arguments visited, graph and starting node.
8.   Stop the program.
### Program:

```
from collections import deque

def bfs(graph, start):
    visited = set()            # to keep track of visited nodes
    queue = deque([start])     # initialize queue with start node
    visited.add(start)

    print("BFS Traversal:", end=" ")
    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example Graph (Adjacency List)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Run BFS
bfs(graph, 'A')
```

### Output:

<img width="828" height="144" alt="image" src="https://github.com/user-attachments/assets/8d754847-b17d-4b60-affe-2978ce6f9d7e" />


### Result:
Thus the breadth first search order was found sucessfully.
