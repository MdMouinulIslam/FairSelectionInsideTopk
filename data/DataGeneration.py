import numpy as np
import operator
from itertools import combinations
import gurobipy as grb
import timeit
from itertools import chain
from sklearn.datasets import make_blobs
from scipy.spatial import distance
#from math import comb
from scipy.special import comb
import pandas as pd
from sklearn.preprocessing import normalize
import heapq as hq
import pickle
from sklearn.preprocessing import MinMaxScaler



K = 10
print("k = ", K)

coef = 0.2
print("lambda coefficient =", coef)

sorted_rel = {}

'''
instance = ["i1","i2","i3","i4","i5"]
sorted_rel["i1"] = 8.6
sorted_rel["i2"] = 8.5
sorted_rel["i3"] = 8.3
sorted_rel["i4"] = 8.1
sorted_rel["i5"] = 7.9
'''

Q = (0, 0)
print("Query: ", Q)

numberofSample = 10000
print("dataset size:", numberofSample)

# makeblobs

dataset = pd.read_csv(r'2M2FMakeBlobs.csv', nrows=numberofSample)
D = dataset.iloc[:, :].values
instance = list([tuple(e) for e in D])
print("makeblobs dataset")

#print(instance)
# movielens
'''
dataset=pd.read_csv(r'ratings.csv', nrows=numberofSample)
D = dataset.iloc[:, [2, 3]].values
#normalized_D = normalize(D, axis=0, norm='l2')
#X = normalized_D
#instance = list([tuple(e) for e in X])
instance = list([tuple(e) for e in D])
print("movielens dataset")
'''

# yelp
'''
dataset=pd.read_csv(r'business.csv' , nrows=numberofSample)
D = dataset.iloc[:, [6,7,8]].values
#instance = list([tuple(e) for e in D])
normalized_D = normalize(D, axis=0, norm='l2')
X = normalized_D*100
instance = list([tuple(e) for e in X])
print("yelp dataset")
#print(instance)
'''

# airbnb

# dataset=pd.read_csv(r'listings.csv' , nrows=numberofSample)
# D = dataset.iloc[:, [6,7,8,9]].values
# instance = list([tuple(e) for e in D])
# print("airbnb dataset")


'''
X, Y = make_blobs(n_samples=numberofSample, centers=10, cluster_std=20, random_state=0)
instance = list([tuple(e) for e in X])
#print("Instance Dataset: ", instance)
dfdata = pd.DataFrame(instance)
dfdata.to_csv("MakeBlobs1000.csv")
'''

total_sets = comb(numberofSample, K)

rel = {}
Q = (0,0)
for i in range(0, len(instance)):
    sim = 1 / (1 + distance.euclidean(Q, instance[i]))
    sim = round(sim, 2)
    rel[instance[i]] = sim

# print("Relevance to the query: ", rel)

sorted_rel = dict(sorted(rel.items(), key=operator.itemgetter(1), reverse=True))

#print("Sorted Relevance to the query: ", sorted_rel.values())

#maxnorm = max(sorted_rel.values())
#minnorm = min(sorted_rel.values())
'''
for k,val in sorted_rel.items():
    val = (val-minnorm)/(maxnorm - minnorm)
    sorted_rel[k]= val
'''
#print('Normalized sorted relevance list')
#print(sorted_rel.values())

max_rel = list(sorted_rel.values())[0]
min_rel = list(sorted_rel.values())[len(sorted_rel) - 1]

print("Maximum Relevance score: ", max_rel)
print("Minimum Relevance score: ", min_rel)


# pairwise_diversity = {}
#
# for i in range(0, len(instance)):
#     for j in range(i + 1, len(instance)):
#         if i != j:
#             dist = distance.euclidean(instance[i], instance[j])
#             dist = round(dist, 2)
#             pairs = [instance[i], instance[j]]
#             pairs.sort()
#             tup = tuple(pairs)
#             pairwise_diversity[tup] = dist


'''
pairwise_diversity[("i2","i3")] = 5
pairwise_diversity[("i3","i5")] = 5
pairwise_diversity[("i1","i3")] = 4
pairwise_diversity[("i3","i4")] = 4
pairwise_diversity[("i1","i4")] = 2
pairwise_diversity[("i4","i5")] = 2
pairwise_diversity[("i1","i2")] = 2
pairwise_diversity[("i2","i4")] = 2
pairwise_diversity[("i2","i5")] = 1
pairwise_diversity[("i1","i5")] = 1
'''

#sorted_pairdiv = pairwise_diversity
#
# sorted_pairdiv = dict(sorted(pairwise_diversity.items(), key=operator.itemgetter(1), reverse=True))
#
# #print("Sorted diversity:", sorted_pairdiv.values())
#
# maxdivnorm = max(sorted_pairdiv.values())
# mindivnorm = min(sorted_pairdiv.values())

'''
for k,val in sorted_pairdiv.items():
    val = (val-mindivnorm)/(maxdivnorm-mindivnorm)
    sorted_pairdiv[k]= val
'''

# print("diversity list sorted")
# #print("Sorted normalized diversity:", sorted_pairdiv.values())
# max_div = list(sorted_pairdiv.values())[0]
# min_div = list(sorted_pairdiv.values())[len(sorted_pairdiv) - 1]
#
#
# print("Maximum Diversity score: ", max_div)
# print("Minimum Diversity score: ", min_div)



file_to_write = open("sortedRel10k-makeblob.pickle", "wb")

pickle.dump(sorted_rel, file_to_write)

# file_to_write2 = open("sortedDiv20makeblobs.pickle", "wb")
#
# pickle.dump(sorted_pairdiv, file_to_write2)
