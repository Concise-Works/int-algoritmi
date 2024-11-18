
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
# print(dfs(H,0)) # 0-1-3-4-2-5
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

r"""
Topological Sort:

A topological sort can be given to DAGs, meaning there's an 
order to nodes. Like putting on clothes

underwear->pants--\
shirt->hoodie------> school
socks->shoes------/

possible orderings:
underwear->pants->shirt...
shirt->socks->hoodie...
underwear->shirt->socks...

using our DFS stack method we can obtain the topological sort 
tracing our backtracking. This tells us, what comes before what,
we should finish processing the final node at the end of our algorithm.
"""

# assuming connected graph and s is a valid starting node with no deps.
def dfs_top(G: Dict[int,List[int]],s: int):
    n = len(G)
    stack = []
    proc = [] # processed stack
    visited = [False] * n
    processed = [False] * n
    stack.append(s)
    while stack:
        # print(stack)
        p = stack[-1]
        if visited[p]:
            if not processed[p]:
                proc.append(p)
                processed[p] = True
            stack.pop()
        else:
            visited[p] = True
            for c in G[p]:
                if not visited[c]:
                    stack.append(c)
    reverse_stack = []
    for _ in range(n):
        reverse_stack.append(proc.pop())
    return reverse_stack
"""
time complexity: O(n+m). Still the same dfs, just that we are evaluating it
differently
space complexity: O(n+m)
"""

J = {
    0: [2,1],
    1: [3],
    2: [1,4],
    3: [],
    4: []
}
K = {
    0: [1,2],
    1: [3],
    2: [1,4],
    3: [],
    4: []
}

        
"""
left to right directed graph
  /-1-3
 / /
0-2--4

possible valid orderings 0-2-1-3-4 or 0-2-4-1-3
"""

# print(dfs_top(J,0)) # 0-2-4-1-3
# print(dfs_top(K,0)) # 0-2-1-3-4

"""
classes in python
__init__: is the constructor function, each function needs 
self as the first parameter to refer to the class itself. Similar to 
'this' is java and other languages.
__str__: is the toString for the class
"""

class Dog:
    # optional typing
    name: str
    age: int
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def bark(self):
        print(f"{self.name}, says a-woof-a-doodle-doo!")
    def __str__(self):
        return f"This dog's ol' name is {self.name}, {self.age}yrs of age~"

# cool_dog = Dog("Reinhardt", 5)
# cool_dog.bark()
# print(cool_dog)

r"""
Min-Max Heaps:
- complete binary tree (binary and balanced)
- Min heaps: nodes with lesser values live at the top
- Max heaps: nodes with higher values live at the top
Functions:
- PEEK: Return root       O(1)
- INSERT: Add new element O(log n)
- EXTRACT: Remove root    O(log n)
- UPDATE: Update a node   O(log n)

Heap Structure:
        10
       /  \
     15    20
    /  \
  17   25

Array Representation: [10, 15, 20, 17, 25]
for index i
'//':= integer division (floor division)
- Left Child:  (2*i)+1
- Right Child: (2*i)+2
- Parent:      (index-1) // 2

Pseudo Code min-heap:
H <- is our array
PEEK: return H[0]
INSERT: 
    - add to the end of the array
    - keep swapping with parent if parent it's smaller
DELETE: 
    - add to the end of the array
    - swap with H[0]
    - keep swapping with children who are smaller
EXTRACT:
    - return H[0] and DELETE it in H
UPDATE:
    - update element at index i
    - if the new value is smaller than the old value
      preform swaps with parents who might be larger
      else, preform swaps with children who might be smaller

This is all log n because the height of a balance tree is log n where
n is the number of nodes.
"""


class MinHeap:
    def __init__(self):
        """Initialize an empty heap."""
        self.heap = []

    def _parent(self, index):
        """Get the parent index."""
        return (index - 1) // 2

    def _left_child(self, index):
        """Get the left child index."""
        return 2 * index + 1

    def _right_child(self, index):
        """Get the right child index."""
        return 2 * index + 2

    def _swap(self, i,j):
        # This syntax avoids having to make a temp variable
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, index):
        """Maintain heap property after insertion."""
        while index > 0: # Stop if at root
            parent = self._parent(index)
            if self.heap[index] < self.heap[parent]:
                # Swap if current node is smaller than the parent
                self._swap(index, parent)
                index = parent
            else:
                break
      
    def _heapify_down(self, index):
        """Maintain heap property after deletion."""
        size = len(self.heap)
        while True:
            left = self._left_child(index)
            right = self._right_child(index)
            smallest = index # Assume the current node is the smallest

            if left < size and self.heap[left] < self.heap[smallest]: # Compare with left child
                smallest = left
            if right < size and self.heap[right] < self.heap[smallest]: # Compare with right child
                smallest = right

            if smallest != index:
                # Swap with the smaller child
                self._swap(index, smallest)
                index = smallest # repeat with the newly swapped position
            else:
                break

    def insert(self, value):
        """Insert a value into the heap."""
        self.heap.append(value)  # Add the value to the end
        self._heapify_up(len(self.heap) - 1)  # Restore heap property

    def update(self, index, new_value):
        """Update a value at a given index."""
        if 0 <= index < len(self.heap):
            old_value = self.heap[index]
            self.heap[index] = new_value
            # If the new value is smaller, heapify up
            if new_value < old_value:
                self._heapify_up(index)
            # If the new value is larger, heapify down
            else:
                self._heapify_down(index)
        else:
            raise IndexError("Index out of range.")

    def delete(self, index):
        """Delete a value at a given index."""
        if 0 <= index < len(self.heap):
            # Swap with the last element and remove it
            self._swap(index,-1)
            self.heap.pop()
            # Restore heap property
            if index < len(self.heap):
                self._heapify_down(index)
        else:
            raise IndexError("Index out of range.")

    def extract_min(self):
        """Extract the minimum value (root) from the heap."""
        if len(self.heap) == 0:
            raise IndexError("Heap is empty.")
        min_value = self.heap[0]
        self.delete(0)
        return min_value

    def __str__(self):
        """String representation of the heap."""
        return str(self.heap)

class MaxHeap:
    def __init__(self):
        """Initialize an empty heap."""
        self.heap = []

    def _parent(self, index):
        """Get the parent index."""
        return (index - 1) // 2

    def _left_child(self, index):
        """Get the left child index."""
        return 2 * index + 1

    def _right_child(self, index):
        """Get the right child index."""
        return 2 * index + 2

    def _swap(self, i,j):
        # This syntax avoids having to make a temp variable
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, index):
        """Maintain heap property after insertion."""
        while index > 0: # Stop if at root
            parent = self._parent(index)
            if self.heap[index] > self.heap[parent]:
                # Swap if current node is greater than the parent
                self._swap(index, parent)
                index = parent
            else:
                break
      
    def _heapify_down(self, index):
        """Maintain heap property after deletion."""
        size = len(self.heap)
        while True:
            left = self._left_child(index)
            right = self._right_child(index)
            largest = index # Assume the current node is the greatest

            if left < size and self.heap[left] > self.heap[largest]: # Compare with left child
                largest = left
            if right < size and self.heap[right] > self.heap[largest]: # Compare with right child
                largest = right

            if largest != index:
                # Swap with the smaller child
                self._swap(index, largest)
                index = largest # repeat with the newly swapped position
            else:
                break

    def insert(self, value):
        """Insert a value into the heap."""
        self.heap.append(value)  # Add the value to the end
        self._heapify_up(len(self.heap) - 1)  # Restore heap property

    def update(self, index, new_value):
        """Update a value at a given index."""
        if 0 <= index < len(self.heap):
            old_value = self.heap[index]
            self.heap[index] = new_value
            # If the new value is greater, heapify up
            if new_value > old_value:
                self._heapify_up(index)
            # If the new value is smaller, heapify down
            else:
                self._heapify_down(index)
        else:
            raise IndexError("Index out of range.")

    def delete(self, index):
        """Delete a value at a given index."""
        if 0 <= index < len(self.heap):
            # Swap with the last element and remove it
            self._swap(index,-1)
            self.heap.pop()
            # Restore heap property
            if index < len(self.heap):
                self._heapify_down(index)
        else:
            raise IndexError("Index out of range.")

    def extract_min(self):
        """Extract the minimum value (root) from the heap."""
        if len(self.heap) == 0:
            raise IndexError("Heap is empty.")
        min_value = self.heap[0]
        self.delete(0)
        return min_value

    def __str__(self):
        return str(self.heap)


"""
Interval scheduling:
When we have a list of jobs, with start and finish times. Our job 
is to find the largest subset of jobs without conflict

Our strategy (Earliest Finish Time [EFT]):
    - Sort schedules in ascending order  
    - Keep taking the next earliest finish time
EFT works as each interval there is some deadline. We want to start 
collecting as many jobs as possible, so we start with the one that 
finishes first. This will give us more opportunity to collect.

This is within the optimal solution. Say we have till 10am to clean our room
and we can either clean our bed or our desk first. They both take 
'b' and 'd' time respectively. It doesn't matter which one we choose first
it will take the same amount of time 'b+d'.

Using that analogy as intuition,

If job 'i' is what we pick and job 'j' is optimal 
and both are interchangeable within an interval-deadline. Then 'i' 
is also an optimal solution, as it doesn't matter.

Recursively do this for each interval and we achieve an optimal like solution.
"""

def insertion_sort(A):
    for i in range(1, len(A)):
        j = i
        while j >= 1:
            if A[j] < A[j-1]:
                A[j],A[j-1] = A[j-1],A[j]
            else:
                break
            j-=1

# arr = [2,4,1,10,5,3]
# print(arr)
# insertion_sort(arr)
# print(arr)

def interval_est_sort(A):
    """Sorting by earliest start time using insertion sort"""
    for i in range(1, len(A)):
        j=i
        while j >=1:
            # tuples (start, finish)
            if A[j][0] < A[j-1][0]:
                A[j], A[j-1] = A[j-1], A[j]
            elif A[j][0] == A[j-1][0] and A[j][1] < A[j-1][1]:
                A[j], A[j-1] = A[j-1], A[j]
            j-=1
def interval_eft_sort(A):
    """Sorting by earliest finish using insertion sort"""
    for i in range(1, len(A)):
        j=i
        while j >=1:
            # tuples (start, finish)
            if A[j][1] < A[j-1][1]:
                A[j], A[j-1] = A[j-1], A[j]
            elif A[j][1] == A[j-1][1] and A[j][0] < A[j-1][0]:
                A[j], A[j-1] = A[j-1], A[j]
            j-=1

# (start, finish)     
times = [(4,7),(3,8),(0,6),(8,11),(1,4),(6,10),(5,9),(3,5)]
# interval_est_sort(times)
# print(times)
# interval_eft_sort(times)
# print(times)
def isCompatible(i,j):
        comes_first, comes_second = (i, j) if i[0] < j[0] else (j, i)
        return comes_first[1] <= comes_second[0]

def interval_schedule(L):
    interval_eft_sort(L)
    sol = []
    sol.append(L[0])
    last_compatible = 0
    for i in range (1,len(L)):
        if isCompatible(L[last_compatible],L[i]):
            sol.append(L[i])
            last_compatible = i 
    return sol
"""
(For this Algorithm)
Time complexity: O(n^2). Bottle necked by choosing insertion sort for 
sorting. Would be O(n log n) if merge sort were used. The routine takes 
O(n), if we assumed the array is already sorted by earliest finish time

Space complexity: O(n). We store n tuples and return at most n of them.
"""

# print(interval_schedule(times)) #[(1, 4), (4, 7), (8, 11)]

"""
Interval Partitioning:
Say you have n classes and k classrooms. You want to find the the best
set such that we utilize the least amount of k classrooms to hold n classes.

Formally, given j jobs and k resources, run j jobs without conflict
allocating the minimum resources needed.

Strategy: Earliest Start Time First [EST]
- sort in EST
- if there's a conflict allocate a new resource
"""
        
def interval_partition_schedule(L):
    interval_est_sort(L)
    resources = [[]]
    resources[0].append(L[0])
    # Go through each class
    for i in range(1,len(L)):
        found_space = False
        # Compatible with any of the present resources?
        for r in resources:
            if isCompatible(r[-1], L[i]):
                r.append(L[i])
                found_space = True
                break
        if not found_space:
            resources.append([L[i]])
    return resources

"""
Time Complexity: O(n^2). Ignoring our sorting algorithm it's still O(n^2).
As in the worst case no job is compatible with each other, creating a new
resource. We iterate n jobs, then we follow the arithmetic sum, as we 
iterate over each r resource, which will increase by 1 each iteration.

Space Complexity: O(n). Input of n items and despite creating some 2D
structure, we never have to allocate any more space than n in either 
direction.
"""

# classes = [(4,10),(8,11),(10,15),(12,15),(0,7),(4,7),(12,15),(0,3),(8,11),(0,3)]
# schedule = interval_partition_schedule(classes)

# for i in range(len(schedule)):
#     print(f"{i}: {schedule[i]}")




            

