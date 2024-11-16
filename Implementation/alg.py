
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

# set[i] is name, set[i][...] least to most preferred
hospitals: List[int] = [[1,2,0],[1,2,0],[0,2,1]] 
residents: List[int] = [[2,1,0],[0,2,1],[0,1,2],]

# residents 0,1,2 get their first choices
print(gale_shapely(residents, hospitals)) # [0,1,2]
# hospitals get their first choice, however
# h0, h1 conflict both wanting r0, but r0 prefers h0
print(gale_shapely(hospitals, residents)) # [0,2,1]
        

