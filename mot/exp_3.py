import numpy as np
from utils import isFairnessSatisfied,getProportions


import math
#
# def getItemProbExp3(dataDict,movieId,eligibleTopk,noUsers,groupIds):
#     item_probs = {}
#     for mid in movieId:
#         item_probs[mid] = 0
#     proportion = getProportions(groupIds)
#
#     for topk in eligibleTopk:
#         if isFairnessSatisfied(groupIds, proportion, topk) == False:
#             continue
#         for mid in topk:
#             item_probs[mid] =  item_probs[mid] + 1
#
#     total = sum(item_probs.values())
#     for key,val in item_probs.items():
#         item_probs[key] = val/total
#     for mid in movieId:
#         item_probs[mid] = item_probs[mid]*noUsers
#
#     X = []
#     Y = []
#     for mid, (x, y) in dataDict.items():
#         if mid in item_probs:
#             x_new = x + item_probs[mid]
#         else:
#             x_new = x
#         X.append(x_new)
#         Y.append(y)
#
#     x = np.array(X).reshape((-1, 1))
#     y = np.array(Y)
#     return x, y,item_probs
#


def getItemProbExp3(dataDict,movieId,eligibleTopk,noUsers,groupIds):
    item_probs = {}
    for mid in movieId:
        item_probs[mid] = 0

    proportion = getProportions(groupIds,movieId)
    for topk in eligibleTopk:
        if isFairnessSatisfied(groupIds, proportion, topk) == False:
            continue
        for mid in topk:
            item_probs[mid] =  item_probs[mid] + 1

    total = sum(item_probs.values())
    for key,val in item_probs.items():
        item_probs[key] = val/total
    for mid in movieId:
        item_probs[mid] = item_probs[mid]*noUsers

    X = []
    Y = []
    for mid, (x, y) in dataDict.items():
        if mid in item_probs:
            u = x[0] + item_probs[mid]
        # else:
        #     x_new = x
        X.append([u,x[1],x[2]])
        Y.append(y)

    x = np.array(X)
    y = np.array(Y)
    return x, y,item_probs
