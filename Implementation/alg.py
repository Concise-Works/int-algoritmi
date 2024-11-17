
from typing import List, Dict # Types for List
from collections import deque # deque allows us to treat an array as a queue

"""
Gale Shapely:
Matching problem between two sets A and B. A proposes to 'b' in B
from their most to least preferred. 'b' will initially accept
'a' until 'b' gets a better offer. Then 'a' will have to propose again.

Stable Matching:
is when no two pairs would rather switch partners.
If groups would rather switch partners it's an unstable matching.

Note:
- Proposers get their best choice, if there is no conflict
- If Proposers a,b both pick c, c chooses their preferred
"""

#P proposers and A acceptors both nxn in length
def gale_shapely(P: List[int], A: List[int]):
    n = len(P)
    matches = [None] * n # index: acceptor, value: proposer
    engaged = [False] * n # prospers
    all_matched = False
    while (not all_matched):
        all_matched = True
        # Loop through all proposers i
        for i in range(n):
            if engaged[i]:
                continue
            all_matched = False
            # Look at i's next preference
            # If i's pref isn't engaged they get them 
            next_pref = P[i][-1]
            if matches[next_pref] == None:
                matches[next_pref] = i
                engaged[i] = True
                P[i].pop() 
            # If i's pref is engaged, see if they
            # prefer i over their current match
            else:
                # going from least-to-most
                for j in range(n-1):
                    # if I see they see i first, ignore i
                    if A[next_pref][j] == i:
                        P[i].pop()
                        break
                    # if they see my current match, engage i
                    elif A[next_pref][j] == matches[next_pref]:
                        engaged[matches[next_pref]] = False
                        engaged[i] = True
                        matches[next_pref] = i
    return matches

"""
Time Complexity: O(n^2). Outer for-loop is O(n), if at least one
p in P isn't engaged we have to run the for-loop again. Hence, in 
worst case every p conflicts with each other. but a in A must prefer
one of them. So we are forced to run n^2 times to resolve n conflicts.

Space Complexity : O(nxn). Size of the pref lists (counting input).
"""

# set[i] is name, set[i][...] least to most preferred
hospitals: List[int] = [[1,2,0],[1,2,0],[0,2,1]] 
residents: List[int] = [[2,1,0],[0,2,1],[0,1,2],]

# residents 0,1,2 get their first choices
# print(gale_shapely(residents, hospitals)) # [0,1,2]
# hospitals get their first choice, however
# h0, h1 conflict both wanting r0, but r0 prefers h0
# print(gale_shapely(hospitals, residents)) # [0,2,1]
        
"""
Trees:
a graph G is a tree if any two statements are true:
- G is connected
- G has n-1 edges
- G has no cycles

Types of Edges in a graph:
1. Tree-edges: an edge in a BFS tree
2. Forward-edges: a non-tree-edge connecting an ancestor to a descendant
3. Backward-edges: descendant to ancestor
4. Cross-edge: connects two nodes that don't have ancestor or defendant
               connections (Does not create cycles).
"""

"""
Breadth-First Search (BFS)

Starting from a node 's', queue 's's children 'c' to visit. Once
a 'c' is visited queue their children. It's important to keep 
nodes visited to avoid adding back nodes we've already evaluated

Properties of graph 'T' produced by BFS:
- 'T' is a tree with a root node 's'
- 's' to any node 'n' in 'T' is the shortest path between them
- Any sub-paths between 's' and 'n' are also shortest paths
  eg. s-e-b-n, s-n is the shortest path from s-n, likewise with e-b
"""

# Dictionary of int keys containing a list of integers
def bfs(G: Dict[int, List[int]],s: int):
    q = deque() # queue object to dequeue with 'popleft()'
    q.append(s)
    T = {}
    visited = [False] * len(G) 
    visited[s] = True

    # Loop the queue until empty
    while q:
        p = q.popleft()
        T[p] = []
        # loop all connections
        for c in G[p]:
            # if already visited do not add to the queue or tree
            if visited[c]:
                continue
            visited[c] = True
            T[p].append(c)
            q.append(c)
    return T

"""
Time complexity: O(n+m), for 'n' nodes and 'm' edges. Each node 'n'
has 'd' connections m:= number of all 'd' connections in the table.

Space complexity: O(n+m) the input and output graph are same length.
"""

r"""
Example G:
     2   4
    / \ /
   0   3
    \ / \ 
     1   5
"""

G = {
    0: [2,1],
    1:[0,3],
    2:[0,3],
    3:[1,2,4,5],
    4:[3],
    5:[3]
    }

# print(bfs(G,0))
r"""
     2   4
    / \ /
   0   3
    \   \ 
     1   5
"""
# print(bfs(G,1))
r"""
     2   4
    /   /
   0   3
    \ / \ 
     1   5
"""
# print(bfs(G,3))
r"""
     2   4
      \ /
   0   3
    \ / \ 
     1   5
"""

"""
Depth-First-Search:
visit each branch fully (does not find longest path)
method (stack):
Process a node, then put it's children on a stack. Pop an item
from the stack then process that node. Continue till stack is empty
"""

def dfs(G: Dict[int,List[int]],s: int):
    n = len(G)
    stack = []
    T = {} # Parent table
    visited = [False] * n
    stack.append(s)
    last_node = None
    while stack:
        # print(stack)
        p = stack[-1]
        if visited[p]:
            stack.pop()
        else:
            visited[p] = True
            T[p] = last_node
            last_node = p
            for c in G[p]:
                if not visited[c]:
                    stack.append(c)
    return T
r"""
Time complexity: O(n+m), we traverse all nodes, every node we traverse we 
mark as visited, and is not added to the stack

Space complexity: O(n+m), for incoming Graph
Example G:
     2   4
    / \ /
   0   3
    \ / \ 
     1   5
"""
# print(dfs(G,0)) # 0-1-3-5-4-2

r"""
Stack:
            5  5v 
            4  4  4  4v
            2  2  2  2  2  2v
         3  3v 3v 3v 3v 3v 3v 3v 
      1  1v 1v 1v 1v 1v 1v 1v 1v 1v 
      2  2  2  2  2  2  2  2v 2v 2v 2v
    0 0v 0  0v 0v 0v 0v 0v 0v 0v 0v 0v 0v
    -------------------------------------
    0 1  2  3  4  5  6  7  8  9  10 11 12

Resulting traversal:
     2---4
         |
   0   3 |
    \ / \|  
     1   5
Which mimics the backtracking 0->1->3->5->3->4->3->2
This works because we place on the stack what is reachable from each node
essentially working as a bfs queue but we only explore one path at a time.
We keep track what path/nodes have already been visited

"""

H = {
    0: [2,1],
    1: [4,3],
    2: [5],
    3: [],
    4: [],
    5: []
}
r"""
      0
     / \
    1   2
   / \   \
  3   4   5
"""        
print(dfs(H,0)) # 0-1-3-4-2-5
"""
[0]
[0v, 2, 1]
[0v, 2, v1, 4, 3]
[0v, 2, v1, 4, 3v]
[0v, 2, v1, 4]
[0v, 2, v1, 4v]
[0v, 2, v1]
[0v, 2]
[0v, 2v, 5]
[0v, 2v, 5v]
[0v, 2v]
[0v]
"""





        


