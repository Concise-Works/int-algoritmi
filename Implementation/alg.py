
from typing import List, Dict

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
# Hospitals & Residents are named 0,1,2, and prefer [*][0,1,2]
# from the other set, from least to most preferred, so that
# we can remove elements in O(1)
hospitals: List[int] = [[1,2,0],[1,2,0],[0,2,1]] 
residents: List[int] = [[0,2,1],[0,1,2],[2,1,0]]

#P proposers and A acceptors both nxn in length
def gale_shapely(P: List[int], A: List[int]):
    n = len(P)
    matches = [None] * n # index: acceptor, value: proposer
    engaged = [False] * n # prospers
    all_matched = False
    while (not all_matched):
        all_matched = True
        
        for i in range(n):
            if engaged[i]:
                continue
            all_matched = False

            next_pref = P[i][-1]
            if matches[next_pref] == None:
                matches[next_pref] = i
                engaged[i] = True
                P[i].pop()
            else:
                for j in range(n-1):
                    if A[next_pref][j] == i:
                        P[i].pop()
                        break
                    if A[next_pref][j == matches[next_pref]]:
                        engaged[matches[next_pref]] = False
                        engaged[i] = True
                        matches[next_pref] = i
    return matches

print(gale_shapely(residents, hospitals))
        

