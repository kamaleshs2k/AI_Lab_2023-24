# Ex.No: 3  Implementation of Minimax Search
### DATE:  30/08/2025                                                                          
### REGISTER NUMBER : 212223060108
### AIM: 
Write a mini-max search algorithm to find the optimal value of MAX Player from the given graph.
### Algorithm:
1. Start the program
2. import the math package
3. Specify the score value of leaf nodes and find the depth of binary tree from leaf nodes.
4. Define the minimax function
5. If maximum depth is reached then get the score value of leaf node.
6. Max player find the maximum value by calling the minmax function recursively.
7. Min player find the minimum value by calling the minmax function recursively.
8. Call the minimax function  and print the optimum value of Max player.
9. Stop the program. 

### Program:
```
# Minimax Algorithm

def minimax(node, depth, is_maximizing, graph):
    # If leaf node, return its value
    if node not in graph:  
        return node
    
    if is_maximizing:
        best = float('-inf')
        for child in graph[node]:
            val = minimax(child, depth+1, False, graph)
            best = max(best, val)
        return best
    else:
        best = float('inf')
        for child in graph[node]:
            val = minimax(child, depth+1, True, graph)
            best = min(best, val)
        return best


# Example Game Tree
# Structure: { Node: [Children] }
# Here, leaf nodes are integers (utilities)
graph = {
    "A": ["B", "C"],     # MAX at A
    "B": ["D", "E"],     # MIN at B
    "C": ["F", "G"],     # MIN at C
    "D": [3, 5],         # MAX at D
    "E": [6, 9],         # MAX at E
    "F": [1, 2],         # MAX at F
    "G": [0, -1]         # MAX at G
}

optimal_value = minimax("A", 0, True, graph)
print("Optimal Value for MAX Player:", optimal_value)
```

### Output:

<img width="824" height="184" alt="image" src="https://github.com/user-attachments/assets/f30cbfa1-6fe7-4873-a1a1-eb393344637b" />


### Result:
Thus the optimum value of max player was found using minimax search.
