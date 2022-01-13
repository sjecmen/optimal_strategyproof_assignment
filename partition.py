import numpy as np
import random
from copy import copy
import networkx as nx
from collections import defaultdict
import itertools

'''
Partitioning algorithms for the 1-to-1 authorship case.
'''

def construct_graph(V, assignment_list, S):
    graph = nx.DiGraph()
    for v in V:
        graph.add_node(v)

    reviewer_map = {v[0] : v for v in V}
    paper_map = {v[1] : v for v in V}
    for (r, p) in assignment_list:
        v0 = reviewer_map[r]
        v1 = paper_map[p]
        graph.add_edge(v0, v1, weight=S[r, p])
    return graph


def random_partition(V):
    V = copy(V)
    random.shuffle(V)
    i = int(len(V)/2)
    V2 = V[:i]
    V1 = V[i:]
    return [V1, V2]

# assignment_list : optimal non-SP assignment
def k1_partition(V, assignment_list, S, size_map={}):
    graph = construct_graph(V, assignment_list, S)

    # size functionality for the general authorship algo
    def size(l):
        return sum([size_map.get(v, 1) for v in l])

    used = set()
    V1 = []
    V2 = []
    node_order = reversed(sorted(V, key=lambda x: size_map.get(x, 1)))
    for v in node_order:
        if v in used:
            continue
        cycle = [v]
        weights = []
        while True:
            u = v
            v = next(graph.successors(u))
            weights.append(graph.get_edge_data(u, v)['weight'])
            if v == cycle[0]:
                break
            cycle.append(v)
        used.update(cycle)
    
        i = np.argmin(weights) # edge from i to i+1
        reorder_cycle = cycle[i+1:] + cycle[:i+1]
        A = []
        B = []
        for i, v in enumerate(reorder_cycle):
            if i % 2 == 0:
                A.append(v)
            else:
                B.append(v)
        if size(B) > size(A): # make sure A is larger
            temp = A
            A = B
            B = temp
        if size(V1) <= size(V2):
            V1 += A
            V2 += B
        else:
            V1 += B
            V2 += A
    assert len(used) == len(V)
    assert len(V1) + len(V2) == len(V)
    if size_map == {}:
        assert abs(len(V1) - len(V2)) <= 1
    return [V1, V2]

def coloring_partition(V, assignment_list, k, S):
    graph = construct_graph(V, assignment_list, S)
    ncolor = (2*k) + 2
    colors = nx.coloring.equitable_color(graph, ncolor) # map v -> color

    maxsim = 0
    for A in itertools.combinations(range(ncolor), k+1):
        sim = 0
        for (u, v, params) in graph.edges(data=True):
            if (colors[u] in A) != (colors[v] in A):
                assert (u[0], v[1]) in assignment_list
                sim += params['weight']
        if sim >= maxsim:
            maxsim = sim
            maxA = A
    V1 = [v for v in V if colors[v] in maxA]
    V2 = [v for v in V if colors[v] not in maxA]
    assert abs(len(V1) - len(V2)) <= k + 1
    return [V1, V2]

def multi_partition(V, assignment_list, k, S):
    graph = construct_graph(V, assignment_list, S)
    ncolor = (2*k) + 1
    d = nx.coloring.equitable_color(graph, ncolor)

    partition_map = defaultdict(list)
    for v, c in d.items():
        partition_map[c].append(v)
    return [part for (_, part) in partition_map.items()] 
