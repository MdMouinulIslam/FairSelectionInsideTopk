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
import pickle
import random as rnd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# filename = "makeblobs10k-topksets-delta0-01.pickle"
#
# with open(filename, 'rb') as f:
#     topk = pickle.load(f)
# f.close()
# print(topk)

#########################################################
def relevence(data,itemId,Q):
    sim = 1 / (1 + distance.euclidean(data[itemId], Q))
    sim = round(sim, 2)
    return sim
def diversity(data,itemId1,itemId2):
    return distance.euclidean(data[int(itemId1.replace('i',''))], data[int(itemId2.replace('i',''))])
##############################################################


##############################

def exactMMR(data,s):

    s = sorted(s)
    s = tuple(s)
    MMR_score = 0
    for item in s:

        maxsim = 0
        for elems in s:
            if item != elems:
                sim = diversity(data,item,elems)
                if maxsim < sim:
                    maxsim = sim
        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
        MMR_score = MMR_i + MMR_score

    return MMR_score


############################


def checkExist(resultSet, randTopK):
    for r in resultSet:
        if sorted(r) == sorted(randTopK):
            return True
    return False


##########################################
import random as rnd


def randomWalk(instance, theta, sorted_rel, sorted_pairdiv, k):
    resultSet = []
    topItemCount = {}
    for i in range(len(instance)):
        topItemCount[i] = 0
    maxmmr = 0
    while (min(topItemCount.values()) < 200):
        randTopK = []
        while (len(randTopK) < k):
            itemId = rnd.randint(0, len(instance) - 1)
            item = instance[itemId]
            if item not in randTopK:
                randTopK.append(item)
        randMMR = exactMMR(randTopK)
        if randMMR > maxmmr:
            maxmmr = randMMR
        if randMMR > theta:
            if checkExist(resultSet, randTopK) == False:
                resultSet.append(randTopK)
        topItemCount[itemId] = topItemCount[itemId] + 1
    print("max mmr ", maxmmr)
    return resultSet


###########################################
#### main


##########################################
import random as rnd


def randomWalkNew(instance, theta, sorted_rel, sorted_pairdiv, k, coeff):
    resultSet = []
    topItemCount = {}
    alpha = 4
    for i in sorted_rel.keys():
        topItemCount[i] = 0

    prob = []

    maxDivVal = {}
    for i in sorted_rel.keys():
        maxDivVal[i] = 0
    '''
    divItems = list(sorted_pairdiv.items())
    for key,val in divItems:
        item1 = key[0]
        item2 = key[1]
        if maxDivVal[item1] < val:
            maxDivVal[item1] = val
        if maxDivVal[item2] < val:
            maxDivVal[item2] = val
    '''
    for key, val in sorted_rel.items():
        obj = coeff * val + (1 - coeff) * maxDivVal[key]
        prob.append(pow(obj, alpha))

    probsum = sum(prob)

    prob = [i / probsum for i in prob]

    maxmmr = -1
    count = 0
    maxIt = 50000
    while (min(topItemCount.values()) < 1):
        randTopK = np.random.choice(list(sorted_rel.keys()), size=k, replace=False, p=prob)
        randMMR = exactMMR(randTopK)
        if randMMR > maxmmr:
            maxmmr = randMMR
        if randMMR > theta:
            if checkExist(resultSet, randTopK) == False:
                resultSet.append(randTopK)
        for item in randTopK:
            topItemCount[item] = topItemCount[item] + 1
        count = count + 1
        if count > maxIt:
            break
    print("max mmr ", maxmmr)
    return resultSet


###########################################
###########################################################################################


import random as rnd

def checkItemCount(itemCount,d):
    count = 0
    for val in itemCount.values():
        if val >=d:
            count = count + 1
    return count/len(itemCount)

def adaptiveRandomWalkNew(data,instance, theta, sorted_rel, sorted_pairdiv, k, coeff,maxIt):
    resultSet = []
    itemCount = {}
    topkItemCount = {}

    alpha = 3
    for i in sorted_rel.keys():
        itemCount[i] = 0
        topkItemCount[i] = 0

    prob = []

    maxDivVal = {}
    for i in sorted_rel.keys():
        maxDivVal[i] = 0
    '''
    divItems = list(sorted_pairdiv.items())
    for key,val in divItems:
        item1 = key[0]
        item2 = key[1]
        if maxDivVal[item1] < val:
            maxDivVal[item1] = val
        if maxDivVal[item2] < val:
            maxDivVal[item2] = val
    '''
    for key, val in sorted_rel.items():
        obj = coeff * val + (1 - coeff) * maxDivVal[key]
        prob.append(pow(obj, alpha))

    probsum = sum(prob)

    prob = [i / probsum for i in prob]



    maxmmr = -1
    count = 0
    while (min(itemCount.values()) < 1):
        randTopK = np.random.choice(list(sorted_rel.keys()), size=k, replace=False, p=prob)
        randMMR = exactMMR(data,randTopK)
        if randMMR > maxmmr:
            maxmmr = randMMR

        for item in randTopK:
            itemCount[item] = itemCount[item] + 1
            if randMMR > theta:
                if checkExist(resultSet, randTopK) == False:
                    topkItemCount[item] = topkItemCount[item] + 1

        if randMMR > theta:
            if checkExist(resultSet, randTopK) == False:
                resultSet.append(randTopK)
                newProb = []
                relKeys = list(sorted_rel.keys())
                for i in range(len(relKeys)):
                    item = relKeys[i]
                    newProb.append(prob[i] / (topkItemCount[item] + 1))
                sumNewProb = sum(newProb)
                prob = [i/sumNewProb for i in newProb]

        count = count + 1
        # r = checkItemCount(itemCount,1)
        # if r > 0.70:
        #    break
        if count > maxIt:
            print("satisfied: ",checkItemCount(itemCount,2))
            break
    print("max mmr ", maxmmr)
    return resultSet


###########################################
###################################################################################

def adaptiveRandomWalk(instance, theta, sorted_rel, sorted_pairdiv, k):
    resultSet = []
    topItemCount = {}
    itemScores = {}
    selectionProbability = []
    cumulitiveProbability = []
    itemIds = []
    for i in range(len(instance)):
        topItemCount[i] = 0
        itemScores[i] = 1
        selectionProbability.append(1 / len(instance))
        itemIds.append(i)
    cumSum = 0
    for i in selectionProbability:
        cumSum = cumSum + i
        cumulitiveProbability.append(cumSum)
    while (min(topItemCount.values()) < 1000):
        randTopK = []
        randTopKItems = []
        itemIndx = 0
        # while(True):
        # itemId = rnd.choices(itemIds, cum_weights=tuple(cumulitiveProbability), k=1)[0]
        # if itemId not in randTopK:
        #     item = instance[itemId]
        #     randTopK.append(itemId)
        #     randTopKItems.append(item)
        #     itemIndx = itemIndx + 1
        #     if itemIndx == k:
        #         break

        randTopKItems = rnd.choices(itemIds, cum_weights=tuple(cumulitiveProbability), k=k)[0]
        randMMR = exactMMR(randTopKItems)
        if randMMR > theta:
            resultSet.append(randTopKItems)
            for itemId in randTopK:
                topItemCount[itemId] = topItemCount[itemId] + 1
                itemScores[itemId] = itemScores[itemId] / (topItemCount[itemId] + 1)
            itemScoreValues = itemScores.values()

            cumSum = 0
            cumulitiveProbability = []
            for i in itemScoreValues:
                prob = i / sum(itemScores)
                selectionProbability.append(prob)
                cumSum = cumSum + prob
                cumulitiveProbability.append(cumSum)

    return resultSet


############################################################

def calculateRecall(exactTopK, randomTopK):
    exactTopKSorted = []
    for etopk in exactTopK.values():
        exactTopKSorted.append(list(sorted(etopk)))

    randomTopKSorted = []
    for rtopk in randomTopK:
        randomTopKSorted.append(sorted(rtopk))

    # randomTopKSorted.extend(exactTopKSorted)
    match = 0
    for pair in exactTopKSorted:
        if list(pair) in randomTopKSorted:
            match = match + 1
    print("number of match = ", match)
    print("number of exact sets = ", len(exactTopK))
    recall = match / len(exactTopK)
    return recall


#########################################################
# topkSets[1] = generateTopkSet(1,topkSets[0]) # oracle

# start = timeit.default_timer()



#yelp
# datasetNameList = ['yelp']
# dataset = pd.read_csv(r'business.csv', nrows=200000)
# D = dataset.iloc[:, [6,7,8]].values



#airbnb
#
# numberofSample = 50000
# datasetNameList = ['airbnb']
# dataset=pd.read_csv(r'listings.csv',  nrows=numberofSample)
# #D = dataset.iloc[:, 54].values
# D = dataset.iloc[:, [6,7,8,9]].values

#
#
# numberofSample = 50000
# datasetNameList = ['airbnb']
# dataset=pd.read_csv(r'listings1.csv',  nrows=numberofSample)
# #D = dataset.iloc[:, 54].values
# D = dataset.iloc[:,1].values

# IMDB
datasetNameList = ['imdb']
numberofSample = 10000
dataset = pd.read_csv(r'ImdbTitleRatings.csv', nrows=numberofSample)
#dataset = pd.read_csv(r'movies.csv', nrows=numberofSample)
#pd.to_numeric(dataset['Year'])
D = dataset.iloc[:, 2].values

#

# numberofSample = 100000
# print("dataset size:", numberofSample)
# dataset = pd.read_csv(r'2M2FMakeBlobs.csv', nrows=numberofSample)
# D = dataset.iloc[:, :].values
# datasetNameList =['makeblob']

nList = [numberofSample]
for datasetName in datasetNameList:
    for n in nList:
        # D = dataset.iloc[:, 54].values
        #D = dataset.iloc[:, 40].values
        # instance = list([tuple(e) for e in D])
        data = list(e for e in D)
        print("airbnb dataset")


        delta = 0.01
        k = 25
        coef = 0.99

        highestMMRdict = {'yelp': 31363.200569260214,'airbnb':368.6739215084142, 'imdb':439.7612455326149,'synthetic':4902.341114004959,'makeblob':100000}
        # highestMMR = 31363.200569260214 #yelp
        #highestMMR = 368.6739215084142  # airbnb
        # highestMMR = 439.7612455326149  #imdb
        # highestMMR = 4902.341114004959  #synthetic
        highestMMR = highestMMRdict[datasetName]


        relFileName = "sortedRel"+str(int(n/1000))+"k-" + datasetName + ".pickle"
        divFileName = "sortedDiv10k-" + datasetName + ".pickle"
        #exactTopkFileName = datasetName + str(n) +"k-topksets-delta0-"+str(delta)[2:]+".pickle"

        # exactTopk = [('i246', 'i987', 'i994', 'i996', 'i999'),  ('i246', 'i978', 'i987', 'i994', 'i999')]

        with open(relFileName, 'rb') as f:
            sorted_rel = pickle.load(f)
        f.close()

        sorted_pairdiv = []
        # with open(divFileName, 'rb') as f:
        #     sorted_pairdiv = pickle.load(f)
        # f.close()

        # with open(exactTopkFileName, 'rb') as f:
        #     exactTopk = pickle.load(f)
        # f.close()

        # hmmrset = ('i467', 'i715', 'i756', 'i912', 'i991')
        # hmmr = exactMMR(hmmrset)
        # print(hmmr)

        # exactMMR(topkSets[1])

        # print("exact mmr:",highestMMR)

        # mmr = exactMMR(('i5202', 'i6732', 'i9926', 'i9967', 'i9997'))
        # print(mmr)


        print("-------------------------------------------------------------------------------------------------------------")
        print("dataset = ", datasetName)
        print("delta=", delta)
        print("highest MMR = ", highestMMR)
        print("lambda coefficient =", coef)


        theta = highestMMR - delta * highestMMR

        instance = list(sorted_rel.keys())
        start = timeit.default_timer()
        resultSetRandomWalk = adaptiveRandomWalkNew(data,instance, theta, sorted_rel, sorted_pairdiv, k, coef,1000)
        end = timeit.default_timer()
        fileName = "Adaptive_Random_top-k_" + datasetName + "_n=" + str(n) + "_k=" + str(k) + "_delta=" + str(delta) + ".pickle"
        print("number of topk set random walk = ", len(resultSetRandomWalk))
        #print("topk sets = ", resultSetRandomWalk)
        with open(fileName, 'wb') as handle:
            pickle.dump(resultSetRandomWalk, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("file saved in : ", fileName)



        # fileName = "Random_top-k_"+datasetName + "_n=" + str(n)+"_k="+str(k)+"_delta="+str(delta)+".pickle"
        # with open(fileName, 'rb') as f:
        #     resultSetRandomWalk = pickle.load(f)

        fileName = "Adaptive_Random_top-k_" + datasetName + "_n=" + str(n) + "_k=" + str(k) + "_delta=" + str(delta) + ".pickle"
        with open(fileName, 'rb') as f:
            resultAdaptiveRandomWalk = pickle.load(f)
        # recall = calculateRecall(exactTopk, resultAdaptiveRandomWalk)
        # print("recall = ", recall)
        print("run time = ", end - start)

        # print("exact topk ", exactTopk)
        # print("adaptive topk ",resultAdaptiveRandomWalk)
        # print("random topk ",resultSetRandomWalk)

        # itemCounterExact = {}
        # for set in exactTopk.values():
        #     for item in set:
        #         if item in itemCounterExact:
        #             itemCounterExact[item] = itemCounterExact[item] + 1
        #         else:
        #             itemCounterExact[item] = 1
        #
        #
        # itemCounterRandom = {}
        # for set in resultSetRandomWalk:
        #     for item in set:
        #         if item in itemCounterRandom:
        #             itemCounterRandom[item] = itemCounterRandom[item] + 1
        #         else:
        #             itemCounterRandom[item] = 1
        #
        #
        #
        # itemCounterAdaptive = {}
        # for set in resultAdaptiveRandomWalk:
        #     for item in set:
        #         if item in itemCounterAdaptive:
        #             itemCounterAdaptive[item] = itemCounterAdaptive[item] + 1
        #         else:
        #             itemCounterAdaptive[item] = 1
        #
        #
        #
        # print("item count exact = ",itemCounterExact)
        # print("item count random = ",itemCounterRandom)
        # print("item count adaptive = ",itemCounterAdaptive)
