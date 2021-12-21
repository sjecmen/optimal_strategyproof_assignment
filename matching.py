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


def assign_with_partition(S, V1, V2, k):
    # COI should be just main author, to keep consistency -- should be incorporated in V1/V2
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)

    assert len(V1) >= len(V2)
    R1, P1 = zip(*V1)
    R2, P2 = zip(*V2)

    assign_vars = {}
    obj = 0
    for r, p in itertools.chain(itertools.product(R1, P2), itertools.product(R2, P1)):
        v = m.addVar(lb=0, ub=1, name=f'{r},{p}')
        obj += v * S[r, p]
        assign_vars[r, p] = v

    m.setObjective(obj, GRB.MAXIMIZE)

    for p in P1:
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in R2) <= k)
    for p in P2:
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in R1) <= k)
    for r in R1:
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in P2) <= k)
    for r in R2:
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in P1) <= k)

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros_like(S)
    for idx, v in assign_vars.items():
        F[idx] = v.x
    assert np.sum(F) == 2 * len(V2)
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

