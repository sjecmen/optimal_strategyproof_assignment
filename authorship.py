import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import product


# Get single-authorship graph
# just doing max-similarity doesn't get the most matched -- just find a maximum matching unweighted
# if missing COI data, just do max-similarity over all
def single_authorship_assign(S, M, conflict_only=False):
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)

    assign_vars = {}
    obj = 0

    nrev, npap = S.shape
    for r, p in product(range(nrev), range(npap)):
        ub = M[r, p] if conflict_only else 1 # all assigned are authors
        v = m.addVar(lb=0, ub=ub, name=f'{r},{p}')
        w = 1 if conflict_only else (S.shape[1] * M[r,p]) + S[r, p] + 1
        obj += v * w
        assign_vars[r, p] = v
    m.setObjective(obj, GRB.MAXIMIZE)

    # match as many as possible without requiring perfect matching
    for p in range(npap):
        m.addConstr(gp.quicksum(assign_vars[r, p] for r in range(nrev)) <= 1)
    for r in range(nrev):
        m.addConstr(gp.quicksum(assign_vars[r, p] for p in range(npap)) <= 1)

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("Model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros((nrev, npap))
    for r, p in product(range(nrev), range(npap)):
        F[r, p] = assign_vars[r, p].x
    return F
 



if __name__ == '__main__':
    # iclr : 883
    # preflib3 : 146
    # DA1 : 400
    # query : 73
    dataset = "preflib3"
    scores = np.load('data/' + dataset + '.npz')
    S = scores["similarity_matrix"]
    M = scores["mask_matrix"]

    # TODO also add AAMAS datasets?

    A = single_authorship_assign(S, M)
    print(np.sum(A), A.shape)

    np.savez('data/' + dataset + '_authorship.npz', single_author_matrix=A)

