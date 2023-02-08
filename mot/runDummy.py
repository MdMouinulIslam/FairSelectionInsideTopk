from genInputData import genData,readInput
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import plot
from utils import createDataPairs
from utils import findEligibleCandidates
from  utils import getEligibleTopK

from  exp_1 import  getItemProbExp1
from exp_2 import  getItemProbExp2
from exp_3 import getItemProbExp3

import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin


def runDummy():
        ############################## input param ################################
        n = 25
        theta = 0.05
        k = 3
        noUsers = 75
        g = 2

        ################################ gen Dummy Data ############################

        dataDict_train,movieId_train,group_train,X_train,Y_train = genData(n,g)
        dataDict_test,movieId_test,group_test,X_test,Y_test = genData(n,g)


        plt.bar(X_train, Y_train, color ='maroon',
                width = 0.4)
        plt.xlabel('#clicks')
        plt.ylabel('avg ratins')
        plt.title("training dataset")
        plt.show()

        # inputFile = 'data/movieLens_100.csv'
        # X_train,Y_train,dataDict_train,movieId_train = readInput(inputFile)["train"]
        # X_test,Y_test,dataDict_test,movieId_test = readInput(inputFile)["test"]
        ################################## train on train data ####################################

        model = LinearRegression()
        x_train = np.array(X_train).reshape((-1, 1))
        y_train = np.array(Y_train)
        model.fit(x_train, y_train)
        r_sq = model.score(x_train, y_train)
        print(f"coefficient of determination train: {r_sq*100:.2f}%")
        y_pred_train = model.predict(x_train)



        ################################# test data gen ###########################################



        x_test = np.array(X_test).reshape((-1, 1))
        y_test = np.array(Y_test)
        r_sq = model.score(x_test, y_test)
        print(f"coefficient of determination test: {r_sq*100:.2f}%")

        ############################### experiments ###########################################

        dataPair = createDataPairs(X_test,Y_test,movieId_test)
        movieEligible = findEligibleCandidates(dataPair,theta,n,k)
        eligibleTopk = getEligibleTopK(dataDict_test,dataPair,movieEligible, theta,k)


        ############################# exp 1 ##############################################

        x_exp1,y_exp1,item_probs1 = getItemProbExp1(dataDict_test,movieId_test,eligibleTopk,noUsers)

        r_sq = model.score(x_exp1,y_exp1)
        y_exp1_pred = model.predict(x_exp1)
        print(f"coefficient of determination our fairness: {r_sq*100:.2f}%")

        plt.subplot(1, 3, 1)
        plt.xlabel('movie id')
        plt.ylabel('#click addition')
        plt.title("our fairness")
        plt.bar(item_probs1.keys(),item_probs1.values(), color ='maroon',width = 0.4)

        ############################# exp 2 ##############################################

        x_exp2,y_exp2,item_probs2 = getItemProbExp2(dataDict_test,movieId_test,eligibleTopk,noUsers)


        r_sq = model.score(x_exp2,y_exp2 )
        y_exp2_pred = model.predict(x_exp2)
        print(f"coefficient of determination uniform: {r_sq*100:.2f}%")

        plt.subplot(1, 3, 2)
        plt.bar(item_probs2.keys(),item_probs2.values(), color ='maroon',width = 0.4)
        plt.xlabel('movie id')
        plt.ylabel('#click addition')
        plt.title("uniform")



        ############################# exp 2 ##############################################

        x_exp3,y_exp3,item_probs3 = getItemProbExp3(dataDict_test,movieId_test,eligibleTopk,noUsers,group_test)


        r_sq = model.score(x_exp3,y_exp3 )
        y_exp3_pred = model.predict(x_exp3)
        print(f"coefficient of determination uniform with group fairness: {r_sq*100:.2f}%")

        plt.subplot(1, 3, 3)
        plt.bar(item_probs3.keys(),item_probs3.values(), color ='maroon',width = 0.4)
        plt.xlabel('movie id')
        plt.ylabel('#click addition')
        plt.title("group fairness")
        plt.show()
        ################################## train datae plot ########################################



        plt.subplot(1, 4, 1)
        plt.plot(x_train, y_train, ".")
        plt.plot(x_train, y_pred_train)
        plt.xlabel('#click')
        plt.ylabel('avg ratings')
        plt.title("original")

        plt.subplot(1, 4, 2)
        plt.plot(x_exp1, y_exp1, ".")
        plt.plot(x_exp1, y_exp1_pred)
        plt.xlabel('#click')
        plt.ylabel('avg ratings')
        plt.title("our fairness")

        plt.subplot(1, 4, 3)
        plt.plot(x_exp2, y_exp2, ".")
        plt.plot(x_exp2, y_exp2_pred)
        plt.xlabel('#click')
        plt.ylabel('avg ratings')
        plt.title("uniform")



        plt.subplot(1, 4, 4)
        plt.plot(x_exp3, y_exp3, ".")
        plt.plot(x_exp3, y_exp3_pred)
        plt.xlabel('#click')
        plt.ylabel('avg ratings')
        plt.title("group fairness")


        plt.show()



        # plot(X_train,Y_train,y_pred_train,"train data")
        # plot(x_exp2,y_exp2,y_exp2_pred,"random")