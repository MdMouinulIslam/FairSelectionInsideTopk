from genInputData import genData
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin
from sklearn import preprocessing
import math
import pandas as pd



def createDataPairs(X_test,Y_test,movieId_test):
    counter = len(X_test)
    dataPair = []
    while (counter):
        counter = counter - 1
        p = (Y_test[counter], X_test[counter], movieId_test[counter])
        dataPair.append(p)
    dataPair.sort(reverse=True)
    return dataPair

def findBestScore(dataPair,k):
    scoreBest = 0
    for i in range(0, k):
        scoreBest = scoreBest + dataPair[i][0]
    return scoreBest


def getCutOff(scoreBest,theta):
    cutOff = scoreBest - scoreBest*theta
    return cutOff


def findEligibleCandidates(dataPair,theta,n,k):
    scoreBest = findBestScore(dataPair, k)
    scoreBest_prime = scoreBest - dataPair[k - 1][0]


    cutOff = getCutOff(scoreBest,theta)
    movieEligible = []
    for i in range(0, n):
        if scoreBest_prime + dataPair[i][0] < cutOff:
            break
        #print(dataPair[i][2])
        movieEligible.append(dataPair[i][2])
    return movieEligible

def getScore(dataDict,topk):
    s = 0
    for mid in topk:
        key,val = dataDict[mid]
        s = s + val
    return s

def getEligibleTopK(dataDict,dataPair,movieEligible, theta,k):
    scoreBest = findBestScore(dataPair, k)
    cutOff = getCutOff(scoreBest, theta)
    allTopk = combinations(movieEligible, k)
    eligibleTopk = []
    for topk in allTopk:
        score = getScore(dataDict,topk)
        if score >= cutOff:
            eligibleTopk.append(topk)
        # else:
        #     break
    return eligibleTopk


def plot(x,y,y_pred,title):
    plt.subplot(1, 2, 1)
    plt.plot(x, y, ".")
    plt.subplot(1, 2, 2)
    plt.plot(x, y_pred)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(title)
    plt.show()


def normalize(Xin):
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_train_minmax = min_max_scaler.fit_transform(Xin)
    # return X_train_minmax
    #
    # X = [i[0] for i in Xin]
    # s = sum(X)
    # new_X = []
    # for x in X:
    #     new_x = x/s*100
    #     new_X.append([new_x])
    # return new_X


    X = [i[0] for i in Xin]
    minVal = min(X)
    maxVal = max(X)

    new_X = []
    for item in Xin:
        u = (item[0] - minVal)/ (maxVal - minVal)
        new_X.append([u,1,1])
    x = np.array(new_X)
    return x




def getProportions(groupIdsOrg,movieId):
    groupIds = {}
    for k,v in groupIdsOrg.items():
        if k in movieId:
            groupIds[k] = v
    allGroups = set(groupIds.values())
    proportion = {}
    for g in allGroups:
        proportion[g] = 0
    for g in groupIds.values():
        proportion[g] = proportion[g] + 1
    return proportion

def isFairnessSatisfied(groupIds,proportion,topk):

    k = len(topk)
    highs = {}
    lows = {}
    sum_portion = sum(proportion.values())
    for i in proportion:
        highs[i] = math.ceil(proportion[i]*k/sum_portion)
        lows[i] = math.floor(proportion[i]*k/sum_portion)

    topkProportion = {}
    allGroups = set(groupIds.values())
    for g in allGroups:
        topkProportion[g] = 0

    for i in topk:
        topkProportion[groupIds[i]]  = topkProportion[groupIds[i]]  + 1

    for g in allGroups:
        if topkProportion[g] < lows[g] or topkProportion[g] > highs[g]:
            return False
    return True

def saveToCsv(values, names, file):
    values = np.array(values)
    values = values.transpose()
    df = pd.DataFrame(values, columns=names)
    df.to_csv(file)
