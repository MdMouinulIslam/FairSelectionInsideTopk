import random as rnd
import numpy as np
from numpy.linalg import norm
import matplotlib as mpl
import matplotlib.mathtext as mathtext
import math
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = "26"
plt.rcParams.update({'mathtext.default':  'regular' })
#plt.rcParams['savefig.facecolor'] = "0.5"
plt.tight_layout()




def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def generateData(n,alpha):
    scoreDic = {}
    for x in range(1,n):
        scoreDic[x] = pow(x,(-1)*alpha)
    # scoreNormalized = scoreDic.values()
    # from sklearn import preprocessing
    return scoreDic

def runImpactDataDist(n,k,theta):
    scoreList = []
    alphaList = [0.001,0.005,0.01,0.05,0.1]
    result = []

    for alpha in alphaList:
        testcase = 1
        sum = 0
        tc = 0
        while (tc < testcase):

            #data = np.random.normal(1,sd,size = (n,2))

            # scoreDic = {}
            #
            # for i in range(n):
            #     score = np.dot(Q, data[i]) / (norm(Q) * norm(data[i]))
            #     scoreDic[i] = score

            scoreDic = generateData(n,alpha)

            # plt.plot(scoreDic.keys(),scoreDic.values())
            # plt.show()
            #fname = "result/alpha="+str(alpha)+".pdf"
           # plt.savefig(fname)



            sortedRelValues = list(scoreDic.values())

            scoreList.append(sortedRelValues)
            highestScore = 0
            for i in range(k):
                highestScore = highestScore + sortedRelValues[i]

            delta = highestScore - theta*highestScore

            topkMinusScore = 0
            for i in range(k-1):
                topkMinusScore = topkMinusScore + sortedRelValues[i]


            for i in range(k,n-1):
                topkScore = topkMinusScore + sortedRelValues[i]
                if topkScore < delta:
                    break

            nPrime = i

            numTopK = nCr(nPrime, k)
            totalPossibleTopK = nCr(n, k)
            sum = sum + numTopK / totalPossibleTopK * 100

            tc = tc + 1

        result.append((sum / testcase))

    plotFigure(scoreList,n)

    return result





def runImpactTheta(n,k,sd):
    Q = np.array([1,1])
    thetaList = [-1,-2,]
    thetaList = [i*0.1 for i in thetaList]
    result = []

    data = np.random.normal(1, sd, size=(n, 2))
    for theta in thetaList:
        testcase = 10
        sum = 0
        tc = 0

        while(tc < testcase):

            scoreDic = {}

            for i in range(n):
                score = np.dot(Q, data[i]) / (norm(Q) * norm(data[i]))
                scoreDic[i] = score



            sortedRelValues = list(scoreDic.values())
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

            numTopK = nCr(nPrime,k)
            totalPossibleTopK = nCr(n,k)
            sum = sum  + numTopK/totalPossibleTopK*1e10
            tc = tc + 1

        result.append(sum / testcase)

    return result

def plotFigure(scoreList, n):
    # import matplotlib.pyplot as plt
    # plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'mathtext.default': 'regular'})
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()

    x = [i for i in range (0,len(scoreList[0]))]
    y0 = scoreList[0]
    y1 = scoreList[1]
    y2 = scoreList[2]
    y3 = scoreList[3]
    y4 = scoreList[4]



    plt.tight_layout()
    # create an index for each tick position

    plt.ylabel('$x^{-α}$')
    plt.xlabel('$x$')

    plt.plot(x,y0,)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(x, y4)

    plt.yticks([0.4,0.6,0.8,1.0], ['0.4','0.6','0.8','1.0'],fontsize=16)
    plt.xticks([0,200,400,600,800,1000],['0','200','400','600','800','1000'],fontsize=16)
   # 0.001, 0.005, 0.01, 0.05, 0.1
    plt.ylim(0.4,1.6)
    plt.legend(['$α = 0.001$', "$α = 0.005$","$α = 0.01$","$α = 0.05$","$α = 0.1$"], fontsize="15",loc='upper left',ncol=2,handleheight=2.4,labelspacing=0.005)
    fig.savefig(r"result/power-law.pdf", dpi=2024, bbox_inches='tight')
    plt.show()



r = runImpactDataDist(n=1000,k=2,theta = 0.01)
print(r)


# r = runImpactTheta(n = 1000,k = 5,sd = 0.1)
# print(r)