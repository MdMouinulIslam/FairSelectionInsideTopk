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
import networkx as nx






k = 5
print("k = ", k)

coef = 0.5
print("lambda coefficient =", coef)

# delta = 1
# print("delta=", delta)


sorted_rel = {}

'''
instance = ["i1","i2","i3","i4"]
sorted_rel["i3"] = 3
sorted_rel["i2"] = 2
sorted_rel["i1"] = 2
sorted_rel["i4"] = 1

'''

Q = (0,0)
print("Query: ", Q)

numberofSample = 1000

dataset=pd.read_csv(r'2M2FMakeBlobs.csv', nrows=numberofSample)
D = dataset.iloc[:, :].values
instance = list([tuple(e) for e in D])

#X, Y = make_blobs(n_samples=numberofSample, centers=20, cluster_std=0.5, random_state=0)
#instance = list([tuple(e) for e in X])
#print("Instance Dataset: ", instance)


total_sets = comb(numberofSample,k)


rel = {}
for i in range(0,len(instance)):
    sim = 1 / (1 + distance.euclidean(Q, instance[i]))
    rel[instance[i]] = sim

#print("Relevance to the query: ", rel)

sorted_rel = dict( sorted(rel.items(), key=operator.itemgetter(1),reverse=True))

#print("Sorted Relevance to the query: ", sorted_rel)


max_rel = list(sorted_rel.values())[0]
min_rel = list(sorted_rel.values())[len(sorted_rel)-1]

print("Maximum Relevance score: ", max_rel)
print("Minimum Relevance score: ", min_rel)



pairwise_diversity = {}

for i in range(0,len(instance)):
    for j in range(i+1, len(instance)):
        if i != j:
            dist = distance.euclidean(instance[i], instance[j])
            pairs = [instance[i], instance[j]]
            pairs.sort()
            tup = tuple(pairs)
            pairwise_diversity[tup] = dist



'''
pairwise_diversity[("i1","i3")] = 10
pairwise_diversity[("i1","i4")] = 5
pairwise_diversity[("i2","i3")] = 5
pairwise_diversity[("i2","i4")] = 5
pairwise_diversity[("i3","i4")] = 5
pairwise_diversity[("i1","i2")] = 4
'''

sorted_pairdiv = dict( sorted(pairwise_diversity.items(), key=operator.itemgetter(1),reverse=True))

max_div = list(sorted_pairdiv.values())[0]
min_div = list(sorted_pairdiv.values())[len(sorted_pairdiv)-1]

print("Maximum Diversity score: ", max_div)
print("Minimum Diversity score: ", min_div)




################################### lattice formation ##############################################

class node:
    def __init__(self, value, height):
        self.value = value
        self.height = height
        self.traversed = False
        self.low_mmr = 0
        self.low_div = {}
        self.low_rel = {}
        self.up_mmr = 0
        self.up_div = {}
        self.up_rel = {}


def reset(G):
    for n in list(G.nodes):
        n.traversed = False


def traverse(G, node):
    for n in G.adj[node]:
        if n.traversed == False:
            n.traversed = True
            traverse(G, n)
            print(n.value)  # ," ",n.score)


#
def add(G, root, newItem, k):
    for n in G.adj[root]:
        if n.traversed == False:
            n.traversed = True
            # print("root val = ",root.value)
            # print("height = ",root.height)
            if (root.height < k - 1):
                add(G, n, newItem, k)
                # print(n.value)
                newVal = n.value + newItem.value
                newNode = node(newVal, n.height + 1)
                #####################################
                sets = n.value
                elem = tuple(newItem.value)[0]
                maxsimLow = -1
                maxsimup = -1
                for intemindex in range(len(sets)):

                    # for elem in sets:
                    item = sets[intemindex]
                    if item != elem:

                        if (item, elem) in seen_divpairlist:
                            sim = sorted_pairdiv[(item, elem)]
                            if maxsimLow < sim:
                                maxsimLow = sim
                            if maxsimup < sim:
                                maxsimup = sim
                        elif (elem, item) in seen_divpairlist:
                            sim = sorted_pairdiv[(elem, item)]
                            if maxsimLow < sim:
                                maxsimLow = sim
                            if maxsimup < sim:
                                maxsimup = sim
                        else:
                            if maxsimLow < min_div:
                                maxsimLow = min_div
                            if maxsimup < max_div:
                                maxsimup = max_div

                        if n.low_div[item] < maxsimLow:
                            newNode.low_div[item] = maxsimLow
                        else:
                            newNode.low_div[item] = n.low_div[item]
                        newNode.low_rel[item] = n.low_rel[item]

                        if n.up_div[item] < maxsimup:
                            newNode.up_div[item] = maxsimup
                        else:
                            newNode.up_div[item] = n.up_div[item]
                        newNode.up_rel[item] = n.up_rel[item]

                newNode.low_div[elem] = maxsimLow
                newNode.up_div[elem] = maxsimup

                if elem in seen_rellist:
                    newNode.low_rel[elem] =  sorted_rel[elem]
                    newNode.up_rel[elem] = sorted_rel[elem]
                else:
                    newNode.low_rel[elem] =  min_rel
                    newNode.up_rel[elem] = max_rel

                newNode.low_mmr = coef * sum(newNode.low_rel.values()) + (1 - coef) * sum(newNode.low_div.values())
                newNode.up_mmr = coef * sum(newNode.up_rel.values()) + (1 - coef) * sum(newNode.up_div.values())

                if len(newNode.value) == k:
                    candidate_sets.append(newNode.value)
                    Lbounds[tuple(newNode.value)] = newNode.low_mmr
                    Ubounds[tuple(newNode.value)] = newNode.up_mmr

                ################### ##################
                G.add_node(newNode)
                G.add_edge(n, newNode)
                G.add_edge(newItem, newNode)


def changeScore(G, root,elem):
    for n in G.adj[root]:
        if n.traversed == False:
            n.traversed = True

            ####################################
            sets = n.value
            maxsimlow = -1
            maxsimup = -1
            for intemindex in range(len(sets)):

                # for elem in sets:
                item = sets[intemindex]
                if item != elem:

                    if (item, elem) in seen_divpairlist:
                        sim = sorted_pairdiv[(item, elem)]
                        if maxsimlow < sim:
                            maxsimlow = sim
                        if maxsimup < sim:
                            maxsimup = sim
                    elif (elem, item) in seen_divpairlist:
                        sim = sorted_pairdiv[(elem, item)]
                        if maxsimlow < sim:
                            maxsimlow = sim
                        if maxsimup<  sim:
                            maxsimup = sim
                    else:
                        if maxsimlow < min_div:
                            maxsimlow = min_div
                        if maxsimup < max_div:
                            maxsimup = max_div


                    if n.low_div[item] < maxsimlow:
                        n.low_div[item] = maxsimlow

                    if n.up_div[item] == max_div:
                        n.up_div[item] = maxsimup
                    if n.up_div[item] < maxsimup:
                        n.up_div[item] = maxsimup

                if item in seen_rellist:
                    n.low_rel[elem] = sorted_rel[elem]
                    n.up_div[elem] = sorted_rel[elem]
                else:
                    n.low_rel[elem] = min_rel
                    n.up_div[elem] = max_rel

            n.low_mmr = coef * sum(n.low_rel.values()) + (1 - coef) * sum(n.low_div.values())
            n.up_mmr = coef * sum(n.up_rel.values()) + (1 - coef) * sum(n.up_div.values())
            if len(n.value) == k:
                Lbounds[tuple(n.value)] = n.low_mmr
                Ubounds[tuple(n.value)] = n.up_mmr

            ####################################
            changeScore(G, n,item)
            #print(n.value)


def printLattice(G,root):
    traverse(G,root)
    reset(G)




def createLattice():
    G = nx.DiGraph()
    root = node('0',0)
    G.add_node(root)
    return G, root



def createNode(v):
    return node(v,1)


def addNode(G,root,n,k):
    add(G,root,n,k)
    G.add_edge(root,n)
    reset(G)



def updateNodes(G,n,item):
    changeScore(G,n,item)
    reset(G)

#################################################################################################################################################




#####leximin

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
#heuristic leximin
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
                itemCountDic[item] =  itemCountDic[item] - 1
            for j in range(m):
                if i != j and P[j] !=0:
                    P[j] = P[j] + P[i] / reduced_m
            P[i] = 0
            #print("now sum = ", sum(P))

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

    print("panel prob = ",nonzero_prob)
    print("item prob = ", item_probs)
    print("sum = ",sum(P), "min of item prob = ", min(item_probs.values()))

    return nonzero_prob, item_probs

###########greedy leximin

def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    #if elements != universe:
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
    setProb = 1/l
    for sett in subsets:
        if sett in cover:
            probs[tuple(sett)] = setProb
        else:
            probs[tuple(sett)] = 0
    #print(cover)

    item_probs = {}
    for i in universe:
        p = 0
        for item, val in probs.items():
            if i in list(item):
                p = p + val

        item_probs[i] = p
    #print("item probs:")
    #print(item_probs)
    nonzeroprobs = {}
    for i,v in probs.items():
        if v != 0:
            nonzeroprobs [i] = v
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

for i in instance:
    rel = sorted_rel[i]
    if rel < minrel:
        minrel = rel

print("Least relevant item in the top-k set:", list(sorted_rel.keys())[list(sorted_rel.values()).index(minrel)])



def getNextRel(position):
    return (list(sorted_rel.keys())[position],list(sorted_rel.values())[position])

#print(getNextRel(0))

def getNextDiv(position):
    return (list(sorted_pairdiv.keys())[position],list(sorted_pairdiv.values())[position])

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
########################### NRA algorithm

itemTonodeMap = {}
candidate_sets = []


def generateTopkSet(i, lastMax):
    global init_j
    global seen_items
    global seen_rellist
    global seen_divpairlist
    global count
    global k
    global delta
    global itemTonodeMap
    global candidate_sets

    lattice, root = createLattice()


    for j in range(init_j,len(sorted_rel)+1):



        nextRel_score = getNextRel(j-1)[1]
        nextDiv_score = getNextDiv(j-1)[1]

        nextRel_item = getNextRel(j-1)[0]
        nextDiv_items = getNextDiv(j-1)[0]

        seen_rellist.append(nextRel_item)

        seen_divpairlist.append(nextDiv_items)


        node1, node2, node3 = None, None, None


        itemlist = [nextRel_item,nextDiv_items[0],nextDiv_items[1]]
        for item in itemlist:
            if item in itemTonodeMap:

                node = itemTonodeMap[item]
                if item in seen_rellist:
                    node.low_rel[item] = sorted_rel[item]
                    node.up_rel[item] = sorted_rel[item]
                else:
                    node.low_rel[item] = min_rel
                    node.up_rel[item] = max_rel

                updateNodes(lattice,node,item)
            else:
                node = createNode([item])
                node.low_div[item] = 0
                node.up_div[item] = 0
                if item in seen_rellist:
                    node.low_rel[item] = sorted_rel[item]
                    node.up_rel[item] = sorted_rel[item]
                else:
                    node.low_rel[item] = min_rel
                    node.up_rel[item] = max_rel

                itemTonodeMap[item] = node
                addNode(lattice,root,node,k)


        seen_items.append(nextRel_item)




        # items = []
        thisiteritems = []

        thisiteritems.append(nextRel_item)

        for elem in nextDiv_items:
            seen_items.append(elem)
            thisiteritems.append(elem)


        seen_items = list(set(seen_items))

        thisiteritems = set(thisiteritems)
        thisiteritems = list(thisiteritems)

        if len(seen_items) < k:
            continue
        else:

            ###################### threshold and upperbound calculation

            all_values = Ubounds.values()
            max_MMR = max(all_values)

            MMR_threshold = max_MMR
            threshold.append(MMR_threshold)
            print("Threshold = ", MMR_threshold)

            #set delta 5% of the threshold value
            #delta = MMR_threshold * 0.05



            ############### pruning and stopping condition
            start = timeit.default_timer()

            if len(Ubounds) > 1:
                maxlb= max(Lbounds.values())
                settt = [k for k, v in Lbounds.items() if v == maxlb][0]
                maxub = 0
                for st in Ubounds.keys():
                    if st != settt:
                        if maxub < Ubounds[st]:
                            maxub = Ubounds[st]
                if  maxlb > maxub:
                    init_j = j + 1
                    print(init_j)
                    return settt  # we found the next best set

            stop = timeit.default_timer()
            prsp_time = stop - start
            print('Time for prune and stop: ', prsp_time)

            print("size of candidate_sets = ",len(candidate_sets))
            print("size of Lbounds = ",len(Lbounds.keys()))
            ############################ pruning condition
            start = timeit.default_timer()

            if len(Ubounds) > 1:
                newCandSet = []
                for setlbList in candidate_sets:
                    setlb = tuple(setlbList)
                    if Ubounds[setlb] >= maxlb:
                         newCandSet.append(setlb)
                    else:
                        count = count + 1
                candidate_sets = newCandSet



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


###########################################
#### main


topkSets[1] = generateTopkSet(1,minrel)

start = timeit.default_timer()

print("exact mmr:",exactMMR(topkSets[1]))
highestMMR = exactMMR(topkSets[1])
delta = 0.1*highestMMR
print("delta=", delta)

theta = highestMMR-delta

stop = timeit.default_timer()
exact_time = stop - start
print('Time for exact MMR: ', exact_time)

print("init_j", init_j)
print("average pruning count for 1 top-k set:", int(count/init_j))
print("total sets:", total_sets)


i =2
topkSets.pop(0)
countGenTopk = 0
while exactMMR(topkSets[i-1]) >= theta and init_j < numberofSample :
    lastMax = exactMMR(topkSets[i - 1])
    Ubounds.pop(topkSets[i-1])
    Lbounds.pop(topkSets[i-1])
    topkSets[i] = generateTopkSet(i,lastMax)

    start = timeit.default_timer()

    #set_prob, item_prob = leximin(list(topkSets.values()))
    #set_prob, item_prob = set_cover(instance,list(topkSets.values()))


    stop = timeit.default_timer()
    lex_time = stop - start
    print('Time for leximin: ', lex_time)
    i = i+1
    countGenTopk = countGenTopk + 1

set_prob, item_prob = heuristic_leximin(list(topkSets.values()))

#print(set_prob, item_prob)
print("init_j", init_j)
print("num times genTopk clled = ", countGenTopk)
print("average pruning count:", int(count/init_j))
print("total sets:", total_sets)