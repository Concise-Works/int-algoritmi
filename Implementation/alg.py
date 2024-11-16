
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
    while(len(q) > 0):
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

"""
Example G:
     2   4
    / \ /
   0   3
    \ / \ 
     1   5
"""

G = {0: [2,1], 1:[0,3],2:[0,3],3:[1,2,4,5],4:[3],5:[3]}

# print(bfs(G,0))
"""
     2   4
    / \ /
   0   3
    \   \ 
     1   5
"""
# print(bfs(G,1))
"""
     2   4
    /   /
   0   3
    \ / \ 
     1   5
"""
# print(bfs(G,3))
"""
     2   4
      \ /
   0   3
    \ / \ 
     1   5
"""
