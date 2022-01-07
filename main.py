import numpy as np
from partition import random_partition, k1_partition, multi_partition, coloring_partition
from matching import assign, assign_with_partition, matrix_to_list
import pickle
import time

'''
Results to report:
    % of opt similarity achieved 
    Compare number of each outcome in each category and compare (differences, histogram)
    KS test on scores in each partition - report p-value and effect size (?)

    --> just save partition and % of opt similarity, and evaluate partition later
'''

dataset = 'iclr2018'
k = 2
print(dataset, k)

'''
scores = np.load("data/" + dataset + ".npz", allow_pickle = True)
S = scores["similarity_matrix"]
data = np.load('data/' + dataset + '_authorship.npz')
author_matrix = data['single_author_matrix']
V = matrix_to_list(author_matrix)

res = {}
for k in [1,2,3]:
    if k == 1:
        data = np.load('data/' + dataset + '_assignment.npz')
        assignment_matrix = data['assignment_matrix_k1']
        assignment_list = matrix_to_list(assignment_matrix)
    else:
        data = np.load('data/' + dataset + '_assignment.npz')
        assignment_matrix_k1 = data['assignment_matrix_k1']
        assignment_list_k1 = matrix_to_list(assignment_matrix_k1)
    
        assignment_matrix = assign(S, V, k)
        assignment_list = matrix_to_list(assignment_matrix)
    
    opt = np.sum(S * assignment_matrix)
    res[k] = opt

with open('saved/iclr2018_opt.pkl', 'wb') as f:
    pickle.dump(res, f)
exit()
'''

scores = np.load("data/" + dataset + ".npz", allow_pickle = True)
S = scores["similarity_matrix"]
data = np.load('data/' + dataset + '_authorship.npz')
author_matrix = data['single_author_matrix']
V = matrix_to_list(author_matrix)

if k == 1:
    data = np.load('data/' + dataset + '_assignment.npz')
    assignment_matrix = data['assignment_matrix_k1']
    assignment_list = matrix_to_list(assignment_matrix)
else:
    data = np.load('data/' + dataset + '_assignment.npz')
    assignment_matrix_k1 = data['assignment_matrix_k1']
    assignment_list_k1 = matrix_to_list(assignment_matrix_k1)

    assignment_matrix = assign(S, V, k)
    assignment_list = matrix_to_list(assignment_matrix)

opt = np.sum(S * assignment_matrix)


print('opt', opt)
methods = ['color', 'k1', 'multi', 'random']
results = { m : [] for m in methods}
ts = time.strftime('%m%d%H%M')
for method in methods:
    print(method)
    T = 1
    if method == 'random':
        T = 100
    score_list = []
    partition_list = []
    for i in range(T):
        if method == 'random':
            partitions = random_partition(V)
        elif method == 'k1':
            if k == 1:
                partitions = k1_partition(V, assignment_list, S)
            else:
                partitions = k1_partition(V, assignment_list_k1, S)
        elif method == 'multi':
            partitions = multi_partition(V, assignment_list, k, S)
        elif method == 'color':
            partitions = coloring_partition(V, assignment_list, k, S)
        else:
            assert False
        partition_list.append(partitions)

        A = assign_with_partition(S, partitions, k)
        s = np.sum(A * S)
        p_opt = s / opt
        print(i, p_opt)
        '''
        if dataset == 'iclr2018':
            f = evaluate_split_fairness(partitions, outcome_map), ks_test(partitions, score_map)
        else:
            f = (0, 0)
        print((s, f))
        '''
        results[method].append((partitions, s))
    fname = f'saved/{dataset}_k{k}_{ts}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
    #np.savez(f'{dataset}_results_k{k}.npz', results=results)
