import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools
import json

'''
Various functions for finding assignments.
'''

def matrix_to_list(M):
    return [tuple(c) for c in np.argwhere(M)]

# One-to-one authorship
# V : list of (rev idx, pap idx) authored pairs
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
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in R) == k)
    for r in R:
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in P) == k)

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros_like(S)
    for idx, v in assign_vars.items():
        F[idx] = v.x
    return F

# partitions : list of subsets, each of which is a list of (rev idx, pap idx) authored pairs
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
        obj += v * S[r, p]
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


# Assignment with arbitrary authorship
def full_assign(S, COI, revload, papload):
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)

    R = range(S.shape[0])
    P = range(S.shape[1])

    assign_vars = {}
    obj = 0
    for r, p in itertools.product(R, P):
        v = m.addVar(lb=0, ub=1-COI[r, p], name=f'{r},{p}')
        obj += v * S[r, p]
        assign_vars[r, p] = v

    m.setObjective(obj, GRB.MAXIMIZE)

    for p in P:
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in R) == papload)
    for r in R:
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in P) <= revload)

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros_like(S)
    for idx, v in assign_vars.items():
        F[idx] = v.x
    return F


def full_assign_with_partition(S, reviewer_partitions_list, paper_partitions_list, revload, papload):
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)

    reviewer_partition = {r : i for (i, part) in enumerate(reviewer_partitions_list) for r in part}
    paper_partition = {p : i for (i, part) in enumerate(paper_partitions_list) for p in part}

    assign_vars = {}
    obj = 0
    for r, p in itertools.product(reviewer_partition, paper_partition):
        ub = 1
        if reviewer_partition[r] == paper_partition[p]:
            ub = 0
        v = m.addVar(lb=0, ub=ub, name=f'{r},{p}')
        obj += v * S[r, p]
        assign_vars[r, p] = v

    m.setObjective(obj, GRB.MAXIMIZE)

    for p in paper_partition:
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in reviewer_partition) == papload)
    for r in reviewer_partition:
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in paper_partition) <= revload)

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros_like(S)
    for idx, v in assign_vars.items():
        F[idx] = v.x
    return F

