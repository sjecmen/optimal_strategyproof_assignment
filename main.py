import numpy as np
from partition import random_partition, k1_partition, multi_partition, coloring_partition
from matching import assign, assign_with_partition, matrix_to_list
from authorship import single_authorship_assign
import pickle
import time

'''
Runs the main algorithms (for 1-to-1-authorship) and saves the similarity and partitions.
'''

# Parameters
dataset = 'iclr2018'
k = 3
print(dataset, k)



scores = np.load("data/" + dataset + ".npz", allow_pickle = True)
S = scores["similarity_matrix"]

try:
    data = np.load('data/' + dataset + '_authorship.npz')
    author_matrix = data['single_author_matrix']
except FileNotFoundError:
    M = scores["mask_matrix"]
    A = single_authorship_assign(S, M)
    np.savez('data/' + dataset + '_authorship.npz', single_author_matrix=A)
V = matrix_to_list(author_matrix)

try:
    data = np.load('data/' + dataset + '_assignment.npz')
    assignment_matrix_k1 = data['assignment_matrix_k1']
except FileNotFoundError:
    assignment_matrix_k1 = assign(S, V, 1)
    np.savez('data/' + dataset + '_assignment.npz', assignment_matrix_k1=assignment_matrix_k1)
assignment_list_k1 = matrix_to_list(assignment_matrix_k1)

if k > 1:
    assignment_matrix = assign(S, V, k)
    assignment_list = matrix_to_list(assignment_matrix)
else:
    assignment_matrix = assignment_matrix_k1
    assignment_list = assignment_list_k1



opt = np.sum(S * assignment_matrix)
with open(f'saved/{dataset}_opt_k{k}.pkl', 'wb') as f:
    pickle.dump(opt, f)

print('opt', opt)
methods = ['color', 'k1', 'multi', 'random']
results = { m : [] for m in methods}
ts = time.strftime('%m%d%H%M')
for method in methods:
    print(method)
    T = 1
    if method == 'random':
        T = 100
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

        A = assign_with_partition(S, partitions, k)
        s = np.sum(A * S)
        p_opt = s / opt
        print(i, p_opt)
        results[method].append((partitions, s))
    fname = f'saved/{dataset}_k{k}_{ts}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
