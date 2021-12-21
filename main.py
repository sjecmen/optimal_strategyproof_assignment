import numpy as np
from partition import random_partition, k1_partition
from matching import assign_with_partition, matrix_to_list
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

def evaluate_split_fairness(V1, V2, outcome_map):
    # 333 accepts, 23 orals 
    oral_v1 = sum(['Accept (Oral)' == outcome_map[p] for (_, p) in V1])
    oral_v2 = sum(['Accept (Oral)' == outcome_map[p] for (_, p) in V2])
    poster_v1 = sum(['Accept (Poster)' == outcome_map[p] for (_, p) in V1])
    poster_v2 = sum(['Accept (Poster)' == outcome_map[p] for (_, p) in V2])
    total_accepts = oral_v1 + oral_v2 + poster_v1 + poster_v2
    total_oral = oral_v1 + oral_v2
    accepts_v1 = oral_v1 + poster_v1
    accepts_v2 = oral_v2 + poster_v2
    print(total_accepts, total_oral)

    oral_per_group = int(total_oral / 2)
    sp_accepts = abs(accepts_v1 - accepts_v2) / (2 * total_accepts) # % of true accepts missed by SP assignment
    sp_oral = abs(oral_v1 - oral_v2) / (2 * total_oral)
    return sp_accepts

    #s1 = np.mean(['Accept' in outcome_map[p] for (_, p) in V1])
    #s2 = np.mean(['Accept' in outcome_map[p] for (_, p) in V2])
    #mean_diff = abs(s1 - s2)

    


if __name__ == '__main__':
    scores = np.load("data/iclr2018.npz", allow_pickle = True)
    S = scores["similarity_matrix"]
    paper_idx = scores["paper_idx"][()]
    data = np.load('data/iclr2018_authorship.npz')
    author_matrix = data['single_author_matrix']
    V = matrix_to_list(author_matrix)
    data = np.load('data/iclr2018_assignment.npz')
    assignment_matrix = data['assignment_matrix_k1']
    assignment_list = matrix_to_list(assignment_matrix)

    f = open('data/iclr2018.json', 'r')
    final_json = json.loads(f.read())
    f.close()
    outcome_map = get_outcomes(final_json, paper_idx)

    print('opt', np.sum(S * assignment_matrix))
    methods = ['random', 'k1']
    results = { m : [] for m in methods}
    for method, T in zip(methods, [100, 10]):
        print(method)
        for i in range(T):
            if method == 'random':
                V1, V2 = random_partition(V)
            elif method == 'k1':
                V1, V2 = k1_partition(V, assignment_list, S)
            else:
                assert False
    
            A = assign_with_partition(S, V1, V2, 1)
            s = score(A, S)
            f = evaluate_split_fairness(V1, V2, outcome_map)
            print((s, f))
            results[method].append((s, f))
        np.savez('iclr_results.npz', results=results)
