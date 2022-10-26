import numpy as np
import operator
from itertools import combinations
import gurobipy as grb
import timeit
from itertools import chain
from sklearn.datasets import make_blobs
from scipy.spatial import distance
from scipy.special import comb
import pandas as pd
from sklearn.preprocessing import normalize
import heapq as hq
import pickle
import networkx as nx
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

numberofSample = 1000
print("dataset size:", numberofSample)

# makeblobs
'''
dataset = pd.read_csv(r'MakeBlobs1000.csv', nrows=numberofSample)
D = dataset.iloc[:, :].values
instance = list([tuple(e) for e in D])
#print("makeblobs dataset")
'''
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
'''
dataset=pd.read_csv(r'listings.csv' , nrows=numberofSample)
D = dataset.iloc[:, [6,7,8,9]].values
instance = list([tuple(e) for e in D]) 
print("airbnb dataset")
'''

'''
X, Y = make_blobs(n_samples=numberofSample, centers=10, cluster_std=20, random_state=0)
instance = list([tuple(e) for e in X])
#print("Instance Dataset: ", instance)
dfdata = pd.DataFrame(instance)
dfdata.to_csv("MakeBlobs1000.csv")
'''

total_sets = comb(numberofSample, K)


'''
rel = {}
for i in range(0, len(instance)):
    sim = 1 / (1 + distance.euclidean(Q, instance[i]))
    sim = round(sim, 2)
    rel[instance[i]] = sim

# print("Relevance to the query: ", rel)

sorted_rel = dict(sorted(rel.items(), key=operator.itemgetter(1), reverse=True))

#print("Sorted Relevance to the query: ", sorted_rel.values())

maxnorm = max(sorted_rel.values())
minnorm = min(sorted_rel.values())

for k,val in sorted_rel.items():
    val = (val-minnorm)/(maxnorm - minnorm)
    sorted_rel[k]= val

print('Normalized sorted relevance list')
print(sorted_rel.values())

max_rel = list(sorted_rel.values())[0]
min_rel = list(sorted_rel.values())[len(sorted_rel) - 1]

print("Maximum Relevance score: ", max_rel)
print("Minimum Relevance score: ", min_rel)


pairwise_diversity = {}

for i in range(0, len(instance)):
    for j in range(i + 1, len(instance)):
        if i != j:
            dist = distance.euclidean(instance[i], instance[j])
            dist = round(dist, 2)
            pairs = [instance[i], instance[j]]
            pairs.sort()
            tup = tuple(pairs)
            pairwise_diversity[tup] = dist



# sorted_pairdiv = pairwise_diversity

sorted_pairdiv = dict(sorted(pairwise_diversity.items(), key=operator.itemgetter(1), reverse=True))

maxdivnorm = max(sorted_pairdiv.values())
mindivnorm = min(sorted_pairdiv.values())
for k,val in sorted_pairdiv.items():
    val = (val-mindivnorm)/(maxdivnorm-mindivnorm)
    sorted_pairdiv[k]= val


# print("diversity list sorted")
print("Sorted diversity:", sorted_pairdiv.values())
max_div = list(sorted_pairdiv.values())[0]
min_div = list(sorted_pairdiv.values())[len(sorted_pairdiv) - 1]


print("Maximum Diversity score: ", max_div)
print("Minimum Diversity score: ", min_div)


'''

with open('sortedRel20makeblobs.pickle', 'rb') as f:
    sorted_rel = pickle.load(f)
f.close()

with open('sortedDiv20makeblobs.pickle', 'rb') as f:
    sorted_pairdiv  = pickle.load(f)
f.close()
#####leximin

print("sorted rel list", sorted_rel.values())
print("sorted diversity list", sorted_pairdiv.values())

max_div = list(sorted_pairdiv.values())[0]
min_div = list(sorted_pairdiv.values())[len(sorted_pairdiv) - 1]

max_rel = list(sorted_rel.values())[0]
min_rel = list(sorted_rel.values())[len(sorted_rel) - 1]

def leximin(panel_items):
    print("number of panel items:", len(panel_items))
    if len(panel_items) == 12:
        print("deb")
    m = grb.Model()
    # Variables for the output probabilities of the different panels
    lambda_p = [m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.) for _ in panel_items]
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    x = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)
    m.addConstr(grb.quicksum(lambda_p) == 1)  # Probabilities add up to 1
    itemSet = set(list(chain.from_iterable([list(itm) for itm in panel_items])))
    for item in itemSet:
        item_probability = grb.quicksum(comm_var for committee, comm_var in zip(panel_items, lambda_p)
                                        if item in committee)
        m.addConstr(item_probability >= x)

    m.setObjective(x, grb.GRB.MAXIMIZE)
    m.optimize()

    probabilities = np.array([comm_var.x for comm_var in lambda_p]).clip(0, 1)
    probabilities = list(probabilities / sum(probabilities))

    finalsetprobs = {}

    for i in range(len(panel_items)):
        for p in range(len(probabilities)):
            if i == p:
                finalsetprobs[panel_items[i]] = probabilities[p]

    print("final panels probabilities: ", finalsetprobs)

    nonzero_prob = {}
    for k, v in finalsetprobs.items():
        if v != 0:
            nonzero_prob[k] = v

    print("non zero panels probabilities:", nonzero_prob)

    print("Size of non zero probability list:", len(nonzero_prob))

    prob = 0
    for i, j in nonzero_prob.items():
        prob = prob + j

    print("total prob:", prob)

    item_probs = {}
    for i in itemSet:
        p = 0
        for item, k in nonzero_prob.items():
            if i in item:
                p = p + k

        item_probs[i] = p

    print("item probs:", item_probs)
    print("Minimum probability of items: ", min(list(item_probs.values())))

    for v in m.getVars():
        print(v.varName, v.x)

    return nonzero_prob, item_probs


#######################
# heuristic leximin
def heuristic_leximin(panel_items):
    instance = set(list(chain.from_iterable([list(itm) for itm in panel_items])))
    itemCountDic = {}
    P = []
    m = len(panel_items)

    for item in instance:
        itemCountDic[item] = 0

    i = 0
    for i in range(m):
        for item in panel_items[i]:
            itemCountDic[item] = itemCountDic[item] + 1
        P.append(1 / m)

    reduced_m = m
    for i in range(m):
        prbZero = True
        for item in panel_items[i]:
            if itemCountDic[item] < 2:
                prbZero = False
                break

        if prbZero:
            reduced_m = reduced_m - 1
            for item in panel_items[i]:
                itemCountDic[item] = itemCountDic[item] - 1
            for j in range(m):
                if i != j and P[j] != 0:
                    P[j] = P[j] + P[i] / reduced_m
            P[i] = 0
            # print("now sum = ", sum(P))

    item_probs = {}

    nonzero_prob = {}

    for i in range(m):
        if P[i] != 0:
            nonzero_prob[panel_items[i]] = P[i]

    for i in instance:
        p = 0
        for item, k in nonzero_prob.items():
            if i in item:
                p = p + k

        item_probs[i] = p

    print("panel prob = ", nonzero_prob)
    print("item prob = ", item_probs)
    print("sum = ", sum(P), "min of item prob = ", min(item_probs.values()))

    return nonzero_prob, item_probs


###########greedy leximin

def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    # if elements != universe:
    #    return None
    covered = set()
    cover = []
    probs = {}
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(set(s) - covered))
        cover.append(subset)
        covered |= set(subset)
    l = len(cover)
    setProb = 1 / l
    for sett in subsets:
        if sett in cover:
            probs[tuple(sett)] = setProb
        else:
            probs[tuple(sett)] = 0
    # print(cover)

    item_probs = {}
    for i in universe:
        p = 0
        for item, val in probs.items():
            if i in list(item):
                p = p + val

        item_probs[i] = p
    print("item probs:")
    print(item_probs)
    nonzeroprobs = {}
    for i, v in probs.items():
        if v != 0:
            nonzeroprobs[i] = v
    print("number of non zero sets:")
    print(len(nonzeroprobs))
    print("non zero set probs:")
    print(nonzeroprobs)
    return nonzeroprobs, item_probs


columns = {}
panel = []
topkSets = {}
seen_relist = []
seen_divpairlist = []

minrel = float('inf')

'''
for i in instance:
    rel = sorted_rel[i]
    if rel < minrel:
        minrel = rel
'''

# print("Least relevant item in the top-k set:", list(sorted_rel.keys())[list(sorted_rel.values()).index(minrel)])


def getNextRel(position):
    return (list(sorted_rel.keys())[position], list(sorted_rel.values())[position])


# print(getNextRel(0))

def getNextDiv(position):
    return (list(sorted_pairdiv.keys())[position], list(sorted_pairdiv.values())[position])


##############################

def exactMMR(s):
    s = sorted(s)
    s = tuple(s)
    MMR_score = 0
    for item in s:

        maxsim = 0
        for elems in s:
            if item != elems:
                pairs = [item, elems]
                pairs.sort()
                tup = tuple(pairs)
                sim = sorted_pairdiv[tup]
                if maxsim < sim:
                    maxsim = sim
        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
        MMR_score = MMR_i + MMR_score

    return MMR_score


############################

topkSets = {}
topkSets[0] = float('inf')
init_j = 1
i = 2
count = 0

threshold = []
Lbounds = {}
Ubounds = {}

seen_items = []
seen_rellist = []
seen_divpairlist = []


flag = False







################################### lattice formation ##############################################
traversedList = []
nodeIdCounter = 0

def isTraversed(id):
    global traversedList
    if id in traversedList:
        return True
    else:
        return False

def addToTraversedNode(id):
    global traversedList
    traversedList.append(id)

def clearAllTraversed():
    global traversedList
    traversedList = []

def getNodeIdCounter():
    global nodeIdCounter
    nodeIdCounter = nodeIdCounter + 1
    return nodeIdCounter



class node:
    def __init__(self, value, height,nodeid):
        self.value = value
        self.height = height
        self.low_mmr = 0
        self.low_div = {}
        self.low_rel = {}
        self.up_mmr = 0
        self.up_div = {}
        self.up_rel = {}
        self.nodeId = nodeid


def reset(G):
    clearAllTraversed()


def traverse(G, node):
    for n in G.adj[node]:
        addToTraversedNode(n.nodeId)
        traverse(G, n)
        print(n.value)  # ," ",n.score)


#

def add(G, root, newItem, k):
    for n in G.adj[root]:
        if isTraversed(n.nodeId) == False:
            addToTraversedNode(n.nodeId)
            # print("root val = ",root.value)
            # print("height = ",root.height)
            if (root.height < k - 1):
                add(G, n, newItem, k)
                # print(n.value)
                newVal = n.value + newItem.value
                newNode = node(newVal, n.height + 1,getNodeIdCounter())
                #####################################

                if newNode.height == k:
                    set1 = newNode.value
                    MMR_score = 0
                    for item in set1:

                        maxsim = 0
                        for elem in set1:
                            if item != elem:

                                if (item, elem) in seen_divpairlist:
                                    sim = sorted_pairdiv[(item, elem)]
                                    if maxsim < sim:
                                        maxsim = sim
                                elif (elem, item) in seen_divpairlist:
                                    sim = sorted_pairdiv[(elem, item)]
                                    if maxsim < sim:
                                        maxsim = sim
                                else:
                                    sim = nextDiv_score
                                    if maxsim < sim:
                                        maxsim = sim
                                        # get rel and div from dictionary

                        if item in seen_rellist:
                            MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
                        else:
                            MMR_i = coef * (nextRel_score) + (1 - coef) * (maxsim)

                        MMR_score = MMR_i + MMR_score


                    Ubounds[tuple(newNode.value)] = MMR_score


                if newNode.height == k:
                    MMR_score = 0
                    sets = newNode.value
                    for item in sets:

                        maxsim = 0
                        for elem in sets:
                            if item != elem:

                                if (item, elem) in seen_divpairlist:
                                    sim = sorted_pairdiv[(item, elem)]
                                    if maxsim < sim:
                                        maxsim = sim
                                elif (elem, item) in seen_divpairlist:
                                    sim = sorted_pairdiv[(elem, item)]
                                    if maxsim < sim:
                                        maxsim = sim
                                else:
                                    sim = min_div
                                    if maxsim < sim:
                                        maxsim = sim
                                        # get rel and div from dictionary

                        if item in seen_rellist:
                            MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
                        else:
                            MMR_i = coef * (min_rel) + (1 - coef) * (maxsim)

                        MMR_score = MMR_i + MMR_score

                    Lbounds[tuple(newNode.value)] = MMR_score

                ################### ##################
                G.add_node(newNode)
                G.add_edge(n, newNode)
                G.add_edge(newItem, newNode)



def changeScore(G, root,k):
    for n in G.adj[root]:
        if isTraversed(n.nodeId) == False:
            addToTraversedNode(n.nodeId)

            ####################################
            if n.height == k:
                set1 = n.value
                MMR_score = 0
                for item in set1:

                    maxsim = 0
                    for elem in set1:
                        if item != elem:

                            if (item, elem) in seen_divpairlist:
                                sim = sorted_pairdiv[(item, elem)]
                                if maxsim < sim:
                                    maxsim = sim
                            elif (elem, item) in seen_divpairlist:
                                sim = sorted_pairdiv[(elem, item)]
                                if maxsim < sim:
                                    maxsim = sim
                            else:
                                sim = nextDiv_score
                                if maxsim < sim:
                                    maxsim = sim
                                    # get rel and div from dictionary

                    if item in seen_rellist:
                        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
                    else:
                        MMR_i = coef * (nextRel_score) + (1 - coef) * (maxsim)

                    MMR_score = MMR_i + MMR_score

                n.up_mmr = MMR_score
                Ubounds[tuple(n.value)] = MMR_score

            if n.height == k:
                MMR_score = 0
                sets = n.value
                for item in sets:

                    maxsim = 0
                    for elem in sets:
                        if item != elem:

                            if (item, elem) in seen_divpairlist:
                                sim = sorted_pairdiv[(item, elem)]
                                if maxsim < sim:
                                    maxsim = sim
                            elif (elem, item) in seen_divpairlist:
                                sim = sorted_pairdiv[(elem, item)]
                                if maxsim < sim:
                                    maxsim = sim
                            else:
                                sim = min_div
                                if maxsim < sim:
                                    maxsim = sim
                                    # get rel and div from dictionary

                    if item in seen_rellist:
                        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
                    else:
                        MMR_i = coef * (min_rel) + (1 - coef) * (maxsim)

                    MMR_score = MMR_i + MMR_score
                n.low_mmr = MMR_score
                Lbounds[tuple(n.value)] = MMR_score

            ####################################
            changeScore(G, n,k)
            #print(n.value)


def printLattice(G,root):
    traverse(G,root)
    reset(G)




def createLattice():
    G = nx.DiGraph()
    root = node('0',0,0)
    G.add_node(root)
    return G, root



def createNode(v):
    return node(v,1,getNodeIdCounter())


def addNode(G,root,n,k):
    add(G,root,n,k)
    G.add_edge(root,n)
    reset(G)



def updateNodes(G,n,k):
    changeScore(G,n,k)
    reset(G)

#################################################################################################################################################

itemTonodeMap = {}
nextRel_score = 0
nextDiv_score = 0
lattice, root = createLattice()

########################### NRA algorithm

def generateTopkSet(i, lastMax):
    global init_j
    global seen_items
    global seen_rellist
    global seen_divpairlist
    global count
    global K
    global totalComb
    global Ubounds
    global Lbounds
    global highestMMR
    global delta
    global theta
    global itemTonodeMap
    global nextDiv_score
    global nextRel_score

    global flag
    flag = False

    for j in range(init_j, len(sorted_rel) + 1):
        print("j = ", j)

        nextRel_score = getNextRel(j - 1)[1]
        nextDiv_score = getNextDiv(j - 1)[1]

        nextRel_item = getNextRel(j - 1)[0]
        nextDiv_items = getNextDiv(j - 1)[0]

        seen_items.append(nextRel_item)
        seen_rellist.append(nextRel_item)

        seen_divpairlist.append(nextDiv_items)

        # items = []
        thisiteritems = []

        thisiteritems.append(nextRel_item)

        for elem in nextDiv_items:
            seen_items.append(elem)
            thisiteritems.append(elem)

        seen_items = list(set(seen_items))

        thisiteritems = set(thisiteritems)
        thisiteritems = list(thisiteritems)

######################################################################################
        #itemlist = list(set([nextRel_item, nextDiv_items[0], nextDiv_items[1]]))
        for item in thisiteritems:
            if item in itemTonodeMap:
                node = itemTonodeMap[item]
                updateNodes(lattice, node, K)
            else:
                node = createNode([item])
                itemTonodeMap[item] = node
                addNode(lattice, root, node, K)
        print("bounds")

        start = timeit.default_timer()
        if i > 1:
            for selected in topkSets.values():
                if selected in Ubounds.keys():
                    Ubounds.pop(selected)
                    Lbounds.pop(selected)
        end = timeit.default_timer()
        print("time to remove prev selected items = ",end-start)
#####################################################################################
        if len(seen_items) < K:
            continue

        candidate_sets = Ubounds.keys()

        start = timeit.default_timer()

        all_values = Ubounds.values()
        max_MMR = min(all_values)


        MMR_threshold = max_MMR
        threshold.append(MMR_threshold)

        if i > 1 and MMR_threshold < theta:
            print("Breaking threshold", MMR_threshold)
            if len(candidate_sets) != 0:
                flag = True

            break

        print("Threshold = ", MMR_threshold)

        print("max upper bound = ",max(Ubounds.values()))
        if len(Ubounds) > 1:
            maxlb = max(Lbounds.values())
            settt = [k for k, v in Lbounds.items() if v == maxlb][0]
            maxub = 0
            ubsettt = Ubounds[settt]
            Ubounds.pop(settt)

            maxub = max(Ubounds.values())
            Ubounds[settt] = ubsettt
            # Ubounds
            if maxlb > maxub:
                init_j = j + 1
                print(init_j)
                # Ubounds[settt] = ubsettt
                return settt  # we found the next best set

        stop = timeit.default_timer()
        prsp_time = stop - start
        # print('Time for prune and stop: ', prsp_time)

        # print("size of candidate_sets = ", len(candidate_sets))
        # print("size of Lbounds = ", len(Lbounds.keys()))
        ############################ pruning condition
        start = timeit.default_timer()

        if len(Ubounds) > 1:
            newCandSet = []
            Ubounds = list(Ubounds.items())
            hq.heapify(Ubounds)
            for le in range(len(Ubounds)):
                # minub = hq.heappop(Ubounds)
                minub = Ubounds[0]
                # print(minub)
                if minub[1] >= maxlb:
                    # hq.heappush(Ubounds, minub)
                    break

            Ubounds = dict(Ubounds)
            candidate_sets = Ubounds.keys()

            '''
            for setlb in candidate_sets:
                #heap
                if Ubounds[setlb] >= maxlb:
                    newCandSet.append(setlb)
                else:
                    count = count + 1
            candidate_sets = newCandSet
            '''

            # for setlb in Lbounds.keys():
            #
            #     if Ubounds[setlb] < maxlb:
            #         count = count +1
            #         candidate_sets.remove(setlb)  # only for this iteration this set gets pruned

        stop = timeit.default_timer()
        pr_time = stop - start
        print('Time for prune: ', pr_time)

        # stopping condition
        start = timeit.default_timer()

        if i > 1:
            if max(Lbounds.values()) >= min(threshold[i], lastMax):

                init_j = j + 1

                mmr = 0
                for eachset in candidate_sets:
                    mmrset = exactMMR(eachset)
                    if mmr < mmrset:
                        mmr = mmrset
                        best = eachset
                return best

        stop = timeit.default_timer()
        sp_time = stop - start
        print('Time for stop: ', sp_time)


    if j == len(sorted_rel) or flag == True:
        flag = True
        mmr = 0
        for eachset in candidate_sets:
            mmrset = exactMMR(eachset)
            if mmr < mmrset:
                mmr = mmrset
                best = eachset
        return best


###########################################
#### main


topkSets[1] = generateTopkSet(1, topkSets[0])

start = timeit.default_timer()

highestMMR = exactMMR(topkSets[1])

print("exact mmr:", highestMMR)

delta = 0.01 * highestMMR

#delta = 289.18

#delta = 0
print("delta=", delta)

theta = highestMMR - delta
print("theta:", theta)

stop = timeit.default_timer()
exact_time = stop - start
print('Time for exact MMR: ', exact_time)

print("init_j", init_j)
print("Set pruning count for 1 top-k set:", int(count))
#print("total sets combination for first top-k:", totalComb)

i = 2
topkSets.pop(0)
countGenTopk = 0
while exactMMR(topkSets[i - 1]) > theta and init_j < numberofSample:
    lastMax = exactMMR(topkSets[i - 1])
    print("last max",lastMax)

    print("i = ",i)

    print("flag = ",flag)
    if flag == True:
        break


    topkSets[i] = generateTopkSet(i, lastMax)

    #print("mmr:", exactMMR(topkSets[i]))

    i = i + 1
    countGenTopk = countGenTopk + 1

start = timeit.default_timer()

finalres = {}
for k,v in topkSets.items():
    if v!= -1:
        finalres[k] = v

print("final top-k sets score", finalres)


print("number of topk set = ", len(finalres))
fileName = "exact_makeblob_1000_k=" + str(k)
with open(fileName, 'wb') as handle:
    pickle.dump(finalres, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("file saved in : ", fileName)

# set_prob, item_prob = leximin(list(topkSets.values()))
# set_prob, item_prob = set_cover(instance,list(topkSets.values()))
# set_prob, item_prob = heuristic_leximin(list(topkSets.values()))

stop = timeit.default_timer()
lex_time = stop - start
# print('Time for leximin: ', lex_time)
# print(set_prob, item_prob)
print("init_j", init_j)

myset = set(seen_items)
uniqueSeenItems = list(myset)

lenuniqueRecords = len(uniqueSeenItems)
print("number of seen records:", lenuniqueRecords)

print("number of final topk sets", len(finalres))

pruneRecord = (numberofSample - lenuniqueRecords) / numberofSample

print("record pruning percentage:", pruneRecord * 100)

print("num times genTopk clled = ", countGenTopk)
print(" set pruning condition count:", int(count))
print("total sets:", total_sets)

#print("total sets combination so far top-k:", totalComb)

#setPrune = (total_sets - totalComb) / total_sets

#print("prune percentage for sets:", setPrune * 100)