
import timeit
import pickle
import numpy as np


def runImpThetaExp(datasetName,k):
    thetaList = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    resultList = []
    for theta in thetaList:
        relFileName = r'data/sortedRel1k-'+ datasetName + ".pickle"



        with open(relFileName, 'rb') as f:
            sorted_rel = pickle.load(f)
        f.close()



        print("dataset = ",datasetName)
        print("theta=", theta)


        n = len(sorted_rel.keys())

        sortedRelValues = list(sorted_rel.values())
        highestScore = 0
        for i in range(k):
            highestScore = highestScore + sortedRelValues[i]

        delta = highestScore - theta*highestScore

        topkMinusScore = 0
        for i in range(k-1):
            topkMinusScore = topkMinusScore + sortedRelValues[i]


        for i in range(k,n):
            topkScore = topkMinusScore + sortedRelValues[i]
            if topkScore < delta:
                break

        nPrime = i + k

        resultList.append(nPrime/n)

    print(resultList)

runImpThetaExp(datasetName = "imdb",k = 5)