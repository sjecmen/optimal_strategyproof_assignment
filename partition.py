import numpy as np
import random
from copy import copy
import networkx as nx
#from equitable_coloring import equitable_color

def random_partition(V):
    V = copy(V)
    random.shuffle(V)
    i = int(len(V)/2)
    V2 = V[:i]
    V1 = V[i:]
    return [V1, V2]

# assignment_list : optimal non-SP assignment
def k1_partition(V, assignment_list, S):
    reviewer_map = {v[0] : v for v in V}
    paper_map = {v[1] : v for v in V}
    digraph_map = {} # edge if v0 reviews v1
    weight_map = {} # weight of edge out of v
    for (r, p) in assignment_list:
        v0 = reviewer_map[r]
        v1 = paper_map[p]
        s = S[r, p]
        digraph_map[v0] = v1
        weight_map[v0] = s

    V = copy(V)
    random.shuffle(V)
    used = set()
    V1 = []
    V2 = []
    for v in V:
        if v in used:
            continue
        used.add(v)
        cycle = [v]
        edges = [weight_map[v]]
        while True:
            v = digraph_map[v]
            if v == cycle[0]:
                break
            used.add(v)
            cycle.append(v)
            s = weight_map[v]
            edges.append(s)
    
        i = np.argmin(edges)
        reorder_cycle = cycle[i+1:] + cycle[:i+1]
        A = []
        B = []
        for i, v in enumerate(reorder_cycle):
            if i % 2 == 0:
                A.append(v)
            else:
                B.append(v)
        if len(V1) <= len(V2):
            V1 += A
            V2 += B
        else:
            V1 += A
            V2 += B
    assert len(used) == len(V)
    return [V1, V2]

def k2_partition(V):
    pass

def multi_partition(V, assignment_list, k):
    graph = nx.DiGraph()
    for v in V:
        graph.add_node(v)

    reviewer_map = {v[0] : v for v in V}
    paper_map = {v[1] : v for v in V}
    for (r, p) in assignment_list:
        v0 = reviewer_map[r]
        v1 = paper_map[p]
        graph.add_edge(v0, v1)

    ncolor = (2*k) + 1
    d = nx.coloring.equitable_color(graph, ncolor)
    partition_map = {}
    for v, c in d.items():
        if c in partition_map:
            partition_map[c].append(v)
        else:
            partition_map[c] = [v]
    return [part for (_, part) in partition_map.items()] 
