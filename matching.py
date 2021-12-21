import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools
import json

def matrix_to_list(M):
    return [tuple(c) for c in np.argwhere(M)]

def assign(S, V, k):
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)

    R, P = zip(*V)

    assign_vars = {}
    obj = 0
    for r, p in itertools.product(R, P):
        v = m.addVar(lb=0, ub=1, name=f'{r},{p}')
        obj += v * S[r, p]
        assign_vars[r, p] = v

    m.setObjective(obj, GRB.MAXIMIZE)

    for (r, p) in V:
        m.addConstr(assign_vars[r, p] == 0)
    for p in P:
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in R) == k) # perfect k-matching only atm
    for r in R:
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in P) == k)

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros_like(S)
    #assignment_list = []
    for idx, v in assign_vars.items():
        F[idx] = v.x
        #if v.x == 1:
        #    assignment_list.append(idx)
        #else:
        #    assert v.x == 0
    return F


def assign_with_partition(S, partitions, k):
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)

    reviewer_partition = {r : i for (i, part) in enumerate(partitions) for (r, _) in part}
    paper_partition = {p : i for (i, part) in enumerate(partitions) for (_, p) in part}

    assign_vars = {}
    obj = 0
    for r, p in itertools.product(reviewer_partition, paper_partition):
        ub = 1
        if reviewer_partition[r] == paper_partition[p]:
            ub = 0
        v = m.addVar(lb=0, ub=ub, name=f'{r},{p}')
        obj += v * (1 + S[r, p]) # just so that everyone is assigned
        assign_vars[r, p] = v

    m.setObjective(obj, GRB.MAXIMIZE)

    for p in paper_partition:
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in reviewer_partition) <= k)
    for r in reviewer_partition:
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in paper_partition) <= k)

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros_like(S)
    for idx, v in assign_vars.items():
        F[idx] = v.x
    assert np.sum(F) >= len(partitions) * min([len(part) for part in partitions])
    return F
 
if __name__ == '__main__':
    dataset = "DA1"
    scores = np.load('data/' + dataset + '.npz')
    S = scores["similarity_matrix"]
    M = scores["mask_matrix"]
    data = np.load('data/' + dataset + '_authorship.npz')
    author_matrix = data['single_author_matrix']
    V = matrix_to_list(author_matrix)

    assignment_matrix = assign(S, V, 1)
    np.savez('data/' + dataset + '_assignment.npz', assignment_matrix_k1=assignment_matrix)

