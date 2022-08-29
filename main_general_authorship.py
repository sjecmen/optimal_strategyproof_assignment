import numpy as np
from partition import k1_partition
from matching import assign, full_assign, full_assign_with_partition, matrix_to_list
import pickle
import networkx as nx
from collections import defaultdict
import itertools
import random
import time


'''
Runs the algorithm (for general-authorship) and saves the similarity and partitions.
'''


def get_connected_components(COI):
    nrev = COI.shape[0]
    npap = COI.shape[1]

    # find connected components
    author_graph = nx.Graph()
    for r in range(nrev):
        author_graph.add_node(r)
    for p in range(npap):
        author_graph.add_node(p + nrev)
    for r, p in np.argwhere(COI):
        author_graph.add_edge(r, p + nrev)
    comps = nx.connected_components(author_graph) # generator of sets of nodes
    rev_comps = {}
    pap_comps = {}
    comp_map = {}
    for i, comp in enumerate(comps):
        comp_map[i] = comp
        for node in comp:
            if node < nrev:
                rev_comps[node] = i
            else:
                pap_comps[node - nrev] = i
    return comp_map, rev_comps, pap_comps


def random_general_partition(S, COI):
    nrev = COI.shape[0]
    npap = COI.shape[1]

    comp_map, rev_comps, pap_comps = get_connected_components(COI)
    ncomp = len(comp_map)

    A = random.sample(range(ncomp), int(ncomp / 2))
    B = [x for x in range(ncomp) if x not in A]

    reviewer_partition = [[x for i in part for x in comp_map[i] if x < nrev] for part in [A, B]]
    paper_partition = [[x-nrev for i in part for x in comp_map[i] if x >= nrev] for part in [A, B]]

    return reviewer_partition, paper_partition
  

def heuristic_partition(S, COI, A_opt):
    nrev = COI.shape[0]
    npap = COI.shape[1]

    comp_map, rev_comps, pap_comps = get_connected_components(COI)
    ncomp = len(comp_map)

    S_comp = np.zeros((ncomp, ncomp))
    for r, p in itertools.product(range(nrev), range(npap)):
        if A_opt[r, p] and r in rev_comps:
            rc = rev_comps[r]
            pc = pap_comps[p]
            S_comp[rc, pc] += S[r, p]
            S_comp[pc, rc] += S[r, p]

    V_comp = [(i, i) for i in range(ncomp)]
    A_comp_opt = assign(S_comp, V_comp, 1)
    A_list_comp = matrix_to_list(A_comp_opt)
    sizes = {v : len([p for p in comp_map[v[0]] if p >= nrev]) for v in V_comp}
    partition_comp = k1_partition(V_comp, A_list_comp, S_comp, sizes)

    reviewer_partition = [[x for (i, _) in part for x in comp_map[i] if x < nrev] for part in partition_comp ]
    paper_partition = [[x-nrev for (i, _) in part for x in comp_map[i] if x >= nrev] for part in partition_comp]

    return reviewer_partition, paper_partition

def remove_high_authorships(S, COI, max_degree):
    reviewer_degrees = np.sum(COI, axis=1)
    mask = reviewer_degrees <= max_degree
    n_removed = np.sum(~mask)
    S_ = S[mask, :]
    COI_ = COI[mask, :]
    print(f'Removed {n_removed} reviewers with degree >{max_degree}', S_.shape)
    return S_, COI_, mask


if __name__ == '__main__':
    dataset = 'iclr2018'
    scores = np.load("data/" + dataset + ".npz", allow_pickle = True)
    S = scores["similarity_matrix"]
    COI = scores["mask_matrix"]
    revload = 6
    papload = 3

    A_opt = full_assign(S, COI, revload, papload)
    s_opt = np.sum(A_opt * S)
    with open(f'saved/{dataset}_opt_gen_rl{revload}pl{papload}.pkl', 'wb') as f:
        pickle.dump(s_opt, f)
    
    methods = ['heuristic_maxdegree3', 'heuristic', 'random']
    results = { m : [] for m in methods}
    ts = time.strftime('%m%d%H%M')
    failed = 0
    for method in methods:
        if method == 'random':
            T = 100
        else:
            T = 1

        for i in range(T):
            print(method, i)
            if method == 'heuristic':
                Rs, Ps = heuristic_partition(S, COI, A_opt)
                print('Size of partitions:', [len(R) for R in Rs], [len(P) for P in Ps])

                A = full_assign_with_partition(S, COI, Rs, Ps, revload, papload)
            elif method == 'heuristic_maxdegree3':
                S_reduced, COI_reduced, mask = remove_high_authorships(S, COI, max_degree=3)
                A_opt_reduced = full_assign(S_reduced, COI_reduced, revload, papload)
                Rs, Ps = heuristic_partition(S_reduced, COI_reduced, A_opt_reduced)
                print('Size of partitions:', [len(R) for R in Rs], [len(P) for P in Ps])

                A_reduced = full_assign_with_partition(S_reduced, COI_reduced, Rs, Ps, revload, papload)
                A = np.zeros_like(COI)
                A[mask, :] = A_reduced
            elif method == 'random':
                Rs, Ps = random_general_partition(S, COI)

                try:
                    A = full_assign_with_partition(S, COI, Rs, Ps, revload, papload)
                except RuntimeError: # if partition is too imbalanced
                    i -= 1
                    failed += 1
                    continue
            else:
                assert False
    
            assert np.all(A <= 1-COI)
            s = np.sum(A * S)
            p_opt = s / s_opt
            print(i, p_opt)
            results[method].append((Rs, Ps, s))

        fname = f'saved/{dataset}_gen_rl{revload}pl{papload}_{ts}.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(results, f)
    if failed > 0:
        print('Number of failures:', failed)
