from genInputData import genData
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin,set_cover
from utils import isFairnessSatisfied,getProportions





#
#
# def getItemProbExp1(dataDict,movieId,eligibleTopk,noUsers):
#     nonzero_prob, _ = leximin(eligibleTopk)
#     item_probs = {}
#     for mid in movieId:
#         item_probs[mid] = 0
#     for key,val in nonzero_prob.items():
#         for mid in key:
#             item_probs[mid] = item_probs[mid] + val
#     total = sum(item_probs.values())
#     for key,val in item_probs.items():
#         item_probs[key] = val/total
#     for mid in movieId:
#         item_probs[mid] = item_probs[mid]*noUsers
#     X = []
#     Y = []
#     for mid, (x, y) in dataDict.items():
#         if mid in item_probs:
#             x_new = x + item_probs[mid]
#         else:
#             x_new = x
#         X.append(x_new)
#         Y.append(y)
#     x = np.array(X).reshape((-1, 1))
#     y = np.array(Y)
#     return x,y,item_probs




def getItemProbExp1(dataDict,movieId,eligibleTopk,noUsers,groupIds):
    #group criteria start
    eligibleTopk_group = []
    proportion = getProportions(groupIds, movieId)
    for topk in eligibleTopk:
        if isFairnessSatisfied(groupIds, proportion, topk) == False:
            continue
        eligibleTopk_group.append(topk)
    # group criteria finish
    nonzero_prob, _ = leximin(eligibleTopk_group)
    #nonzero_prob, _ = set_cover(eligibleTopk)
    item_probs = {}
    for mid in movieId:
        item_probs[mid] = 0


    for key,val in nonzero_prob.items():
        for mid in key:
            item_probs[mid] = item_probs[mid] + val
    total = sum(item_probs.values())
    for key,val in item_probs.items():
        item_probs[key] = val/total
    for mid in movieId:
        item_probs[mid] =  item_probs[mid]*noUsers
    X = []
    Y = []
    for mid, (x, y) in dataDict.items():
        if mid in item_probs:
            u = x[0] + item_probs[mid]
        # else:
        #     x_new = x[0]
        X.append([u,x[1],x[2]])
        Y.append(y)
    x = np.array(X)
    y = np.array(Y)
    return x,y,item_probs



