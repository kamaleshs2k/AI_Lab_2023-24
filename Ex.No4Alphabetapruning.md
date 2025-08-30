# Ex.No: 4   Implementation of Alpha Beta Pruning 
### DATE: 30/08/2025                                                                           
### REGISTER NUMBER : 212223060108
### AIM: 
Write a Alpha beta pruning algorithm to find the optimal value of MAX Player from the given graph.
### Steps:
1. Start the program
2. Initially  assign MAX and MIN value as 1000 and -1000.
3.  Define the minimax function  using alpha beta pruning
4.  If maximum depth is reached then return the score value of leaf node. [depth taken as 3]
5.  In Max player turn, assign the alpha value by finding the maximum value by calling the minmax function recursively.
6.  In Min player turn, assign beta value by finding the minimum value by calling the minmax function recursively.
7.  Specify the score value of leaf nodes and Call the minimax function.
8.  Print the best value of Max player.
9.  Stop the program. 

### Program:
```
def alphabeta(node, depth, alpha, beta, maximizingPlayer, graph):
    # If leaf node, return its value
    if node not in graph:
        return node
    
    if maximizingPlayer:  # MAX player
        best = float('-inf')
        for child in graph[node]:
            val = alphabeta(child, depth+1, alpha, beta, False, graph)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:  # prune
                break
        return best
    else:  # MIN player
        best = float('inf')
        for child in graph[node]:
            val = alphabeta(child, depth+1, alpha, beta, True, graph)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:  # prune
                break
        return best


# Example Game Tree
# Structure: { Node: [Children] }
# Leaf nodes are integers
graph = {
    "A": ["B", "C"],     # MAX
    "B": ["D", "E"],     # MIN
    "C": ["F", "G"],     # MIN
    "D": [3, 5],         # MAX
    "E": [6, 9],         # MAX
    "F": [1, 2],         # MAX
    "G": [0, -1]         # MAX
}

optimal_value = alphabeta("A", 0, float('-inf'), float('inf'), True, graph)
print("Optimal Value for MAX Player:", optimal_value)
```

### Output:

<img width="828" height="180" alt="image" src="https://github.com/user-attachments/assets/f7d2081e-d91e-4f7e-a99d-37258ea03414" />

### Result:
Thus the best score of max player was found using Alpha Beta Pruning.
