from genInputData import genData
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin,set_cover
from utils import isFairnessSatisfied,getProportions


from itertools import chain
import gurobipy as grb

import numpy as np
import math
from birkhoff import birkhoff_von_neumann_decomposition


def getItemProbExp4(dataDict,movieId,eligibleTopk,noUsers,groupIds):
    u = []
    G1 = []
    G2 = []
    count = 0
    idMap = {}
    for mid, (x, y) in dataDict.items():
        u.append(x[0])
        g = groupIds[mid]
        if g == 0:
            G1.append(count)
        else:
            G2.append(count)
        idMap[count] = mid
        count = count + 1

    l = len(u)
    v = [1 / i  for i in range(1, l + 1)]
    m = grb.Model()
    p = m.addVars(l, l, vtype=grb.GRB.CONTINUOUS, name='p')
    m.addConstrs(grb.quicksum(p[i, j] for i in range(l)) == 1 for j in range(l))
    m.addConstrs(grb.quicksum(p[i, j] for j in range(l)) == 1 for i in range(l))
    m.addConstrs(p[i, j] <= 1 for j in range(l) for i in range(l))
    m.addConstrs(p[i, j] >= 0 for j in range(l) for i in range(l))
    m.addConstrs(p[i, j] >= 0 for j in range(l) for i in range(l))
    m.addConstr(grb.quicksum(p[i, j] * v[j] for j in range(l) for i in range(l) if i in G1) - grb.quicksum(
        p[i, j] * v[j] for j in range(l) for i in range(l) if i in G2) == 0)
    m.setObjective(grb.quicksum(u[i] * p[i, j] * v[j] for i in range(l) for j in range(l)), grb.GRB.MAXIMIZE)
    m.optimize()
    values = m.getAttr("X", m.getVars())
    P = np.array(values).reshape((l, l))

    groupExp1 = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            if i in G1:
                groupExp1 = groupExp1 + P[i, j] * v[j]
    groupExp2 = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            if i in G2:
                groupExp2 = groupExp2 + P[i, j] * v[j]

    print("group exposure 1 = ",groupExp1)
    print("group exposure 2 = ", groupExp2)

    result = birkhoff_von_neumann_decomposition(P)
    perset = []
    coset = []
    for coefficient, permutation_matrix in result:
        #print('coefficient:', coefficient)
        #print('permutation matrix:', permutation_matrix)
        perset.append(permutation_matrix)
        coset.append(coefficient)

    r = 0
    result = []
    for mat in perset:
        ranking = []
        print(mat)
        for i in range(len(mat)):
            for j in range(len(mat)):
                if mat[i][j] == 1:
                    ranking.append(j)
        result.append((ranking, coset[r]))
        r = r + 1


    k = 3
    item_probs_pre = getIndividualFairness(result, k)
    item_probs = {}
    for itemid,prob in item_probs_pre.items():
        item_probs[idMap[itemid]] = prob

    X = []
    Y = []
    for mid, (x, y) in dataDict.items():
        if mid in item_probs:
            u = x[0] + item_probs[mid]*noUsers
        # else:
        #     x_new = x[0]
        X.append([u, x[1], x[2]])
        Y.append(y)
    x = np.array(X)
    y = np.array(Y)

    return x, y, item_probs






def getIndividualFairness(rankings, k):
    indFairMap = {}
    for item in range(len(rankings[0][0])):
        indFairMap[item] = 0
        # print(item)
    for rank, coef in rankings:
        topk = rank[0:k]
        for item in topk:
            if item in indFairMap:
                indFairMap[item] = indFairMap[item] + coef
            else:
                indFairMap[item] = coef
    return indFairMap