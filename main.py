import numpy as np
from partition import random_partition, k1_partition, multi_partition
from matching import assign, assign_with_partition, matrix_to_list
import json

def get_outcomes(final_json, paper_idx):
    outcome_map = {}
    for elt in final_json:
        key = elt['url'].split('=')[-1] + '.txt'
        if key in paper_idx:
            idx = paper_idx[key]
            outcome_map[idx] = elt['decision']
    return outcome_map

def score(A, S):
    return np.sum(A * S)

def evaluate_split_fairness(partitions, outcome_map):
    oral_counts = [sum(['Accept (Oral)' == outcome_map[p] for (_, p) in partition]) for partition in partitions]
    poster_counts = [sum(['Accept (Poster)' == outcome_map[p] for (_, p) in partition]) for partition in partitions]
    accept_counts = [o + p for (o, p) in zip(oral_counts, poster_counts)]

    # 333 accepts, 23 orals 
    total_accepts = sum(accept_counts)

    accepts_per_group = np.mean(accept_counts)
    sp_accepts = sum([min(accepts_per_group, accepts) for accepts in accept_counts])
    sp_accepts_missed_frac = 1 - (sp_accepts / total_accepts)

    #sp_accepts_missed_frac = abs(accepts_v1 - accepts_v2) / (2 * total_accepts) # % of true accepts missed by SP assignment
    #sp_oral = abs(oral_v1 - oral_v2) / (2 * total_oral)
    return sp_accepts_missed_frac

    #s1 = np.mean(['Accept' in outcome_map[p] for (_, p) in V1])
    #s2 = np.mean(['Accept' in outcome_map[p] for (_, p) in V2])
    #mean_diff = abs(s1 - s2)

    


if __name__ == '__main__':
    dataset = 'query'

    scores = np.load("data/" + dataset + ".npz", allow_pickle = True)
    S = scores["similarity_matrix"]
    data = np.load('data/' + dataset + '_authorship.npz')
    author_matrix = data['single_author_matrix']
    V = matrix_to_list(author_matrix)

    k = 1

    if k == 1:
        data = np.load('data/' + dataset + '_assignment.npz')
        assignment_matrix = data['assignment_matrix_k1']
        assignment_list = matrix_to_list(assignment_matrix)
    else:
        assignment_matrix = assign(S, V, k)
        assignment_list = matrix_to_list(assignment_matrix)

    if dataset == 'iclr2018':
        paper_idx = scores["paper_idx"][()]
        f = open('data/iclr2018.json', 'r')
        final_json = json.loads(f.read())
        f.close()
        outcome_map = get_outcomes(final_json, paper_idx)

    print('opt', np.sum(S * assignment_matrix))
    methods = ['multi']
    results = { m : [] for m in methods}
    for method, T in zip(methods, [1]):
        print(method)
        for i in range(T):
            if method == 'random':
                partitions = random_partition(V)
            elif method == 'k1':
                partitions = k1_partition(V, assignment_list, S)
            elif method == 'multi':
                partitions = multi_partition(V, assignment_list, k)
            else:
                assert False
    
            A = assign_with_partition(S, partitions, k)
            s = score(A, S)
            if dataset == 'iclr2018':
                f = evaluate_split_fairness(partitions, outcome_map)
            else:
                f = 0
            print((s, f))
            results[method].append((s, f))
        np.savez(dataset + '_results2.npz', results=results)
