from genInputData import genData, readInput, readInputLTR, readGroup
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from utils import plot
from utils import createDataPairs
from utils import findEligibleCandidates
from utils import getEligibleTopK

from exp_1 import getItemProbExp1
from exp_2 import getItemProbExp2, getItemFrequency
from exp_3 import getItemProbExp3
from utils import normalize
from sklearn import preprocessing
import pandas as pd

import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin, leximin
import lightgbm as lgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def runLTR():
    inputFile = 'data/movieLens_features_80.csv'
    X_train, Y_train, dataDict_train, movieId_train, group_train = readInputLTR(inputFile)["train"]
    X_test, Y_test, dataDict_test, movieId_test, group_test = readInputLTR(inputFile)["test"]
    n = len(Y_test)
    theta = 0.1
    k = 3
    noUsers = 100
    g = 2
    params = {
        'task': 'train',
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': 10,
        'learnnig_rage': 0.05,
        'metric': {'l2', 'l1'},
        'verbose': -1
    }
    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_test = np.array(X_test)
    y_test = np.array(Y_test)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    model = lgb.train(params,
                      train_set=lgb_train,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=30)

    # prediction
    y_pred = model.predict(x_test)

    # accuracy check
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** (0.5)
    mae = mean_absolute_error(y_test, y_pred)
    # print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)
    print("MAE: %.2f" % mae)


def runRealV1():
    ############################# input data ########################################

    # dataDict_train,movieId_train,group_train,X_train,Y_train = genData(n,g)
    # dataDict_test,movieId_test,group_test,X_test,Y_test = genData(n,g)
    noUsersList = [0, 10000, 20000, 30000]
    maeList_our_fairness = []
    maeList_our_uniform = []
    maeList_our_groupfair = []

    rsList_our_fairness = []
    rsList_our_uniform = []
    rsList_our_groupfair = []

    # fig1, ax1 = plt.subplots(2, len(noUsersList))
    # fig2, ax2 = plt.subplots(1, len(noUsersList))
    # fig3, ax3 = plt.subplots(2, len(noUsersList))
    # fig4, ax4 = plt.subplots(1, len(noUsersList))
    # fig,ax5 = plt.subplots(1, len(noUsersList))
    ourFairClicks = []
    uniformClicks = []
    uniGropuClick = []

    for it in range(0, len(noUsersList)):
        noUsers = noUsersList[it]
        inputFile = 'data/movieLens_features_88.csv'
        inputGroupFile = 'data/group_all.csv'

        X_train, Y_train, dataDict_train, movieId_train, group_train = readInputLTR(inputFile)["train"]
        X_test, Y_test, dataDict_test, movieId_test, group_test = readInputLTR(inputFile)["test"]
        groups = readGroup(inputGroupFile)
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': 10,
            'learnnig_rage': 0.05,
            'metric': {'l2', 'l1'},
            'verbose': -1
        }
        x_train = np.array(X_train)
        y_train = np.array(Y_train)
        x_test = np.array(X_test)
        y_test = np.array(Y_test)
        x_train = normalize(x_train)
        x_test = normalize(x_test)
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
        # lgb.LGBMRegressor(random_state=1, num_leaves=6000, n_estimators=1, num_boost_round=500, max_depth=300,
        #                   learning_rate=0.002,
        #                   n_jobs=8)  #
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        # accuracy check
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** (0.5)
        mae = mean_absolute_error(y_test, y_pred)
        # print("MSE: %.2f" % mse)
        print("RMSE: %.2f" % rmse)
        print("MAE: %.2f" % mae)

        ############################## input param ################################
        n = len(Y_test)
        # theta = 0.3
        theta = 0.3
        k = 3
        ############################### experiments ###########################################

        dataPair = createDataPairs(X_test, Y_test, movieId_test)
        movieEligible = findEligibleCandidates(dataPair, theta, n, k)
        eligibleTopk = getEligibleTopK(dataDict_test, dataPair, movieEligible, theta, k)

        ############################# exp 1 ##############################################

        x_exp1, y_exp1, item_probs1 = getItemProbExp1(dataDict_test, movieId_test, eligibleTopk, noUsers)

        x_exp1 = normalize(x_exp1)



        y_exp1_pred = model.predict(x_exp1)
        mae_test = mean_absolute_error(y_test, y_exp1_pred)
        r_sq = r2_score(y_test, y_exp1_pred)
        rsList_our_fairness.append(r_sq)


        print(f"MAE our fairness {noUsersList[it]} = : {mae_test * 100:.2f}%")
        maeList_our_fairness.append(mae_test)



        x_exp2, y_exp2, item_probs2 = getItemProbExp2(dataDict_test, movieId_test, eligibleTopk, noUsers)

        x_exp2 = normalize(x_exp2)

        y_exp2_pred = model.predict(x_exp2)
        mae_test = mean_absolute_error(y_test, y_exp2_pred)
        maeList_our_uniform.append(mae_test)
        r_sq = r2_score(y_test, y_exp2_pred)
        rsList_our_uniform.append(r_sq)


        print("############## user = ", noUsers, " ####################")


        x_exp3, y_exp3, item_probs3 = getItemProbExp3(dataDict_test, movieId_test, eligibleTopk, noUsers, groups)

        x_exp3 = normalize(x_exp3)

        y_exp3_pred = model.predict(x_exp3)
        mae_test = mean_absolute_error(y_test, y_exp3_pred)
        maeList_our_groupfair.append(mae_test)
        r_sq = r2_score(y_test, y_exp3_pred)
        rsList_our_groupfair.append(r_sq)


        x_exp1_click = [i[0] for i in x_exp1]
        x_exp2_click = [i[0] for i in x_exp2]
        x_exp3_click = [i[0] for i in x_exp3]
        ourFairClicks.append(x_exp1_click)
        uniformClicks.append(x_exp2_click)
        uniGropuClick.append(x_exp3_click)



    plt.rcParams.update({'font.size': 15})

    movieIds = [i for i in range(0, len(item_probs1.keys()))]
    labels = ["beginning", "after 10k", "after 20k", "after 30k"]
    for i in range(0, len(ourFairClicks)):
        clicks = ourFairClicks[i]
        plt.plot(movieIds, clicks, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel("movie ids")
    plt.ylabel("#normalized clicks")
    plt.savefig("figures/maxminfair_normalized_clicks_vs_movie_ids.pdf", dpi=1000)
    plt.show()

    for i in range(0, len(ourFairClicks)):
        clicks = uniformClicks[i]
        plt.plot(movieIds, clicks, label=labels[i])
    plt.xlabel("movie ids")
    plt.ylabel("#normalized clicks")
    plt.legend(loc='best')
    plt.savefig("figures/uniform_normalized_clicks_vs_movie_ids.pdf", dpi=1000)
    plt.show()

    for i in range(0, len(uniGropuClick)):
        clicks = uniGropuClick[i]
        plt.plot(movieIds, clicks, label=labels[i])
    plt.xlabel("movie ids")
    plt.ylabel("#normalized clicks")
    plt.legend(loc='best')
    plt.savefig("figures/uniform_fair_clicks_vs_movie_ids.pdf", dpi=1000)
    plt.show()



    # rs
    plt.plot(noUsersList, rsList_our_fairness, "-b", label="MaxMinFair")
    plt.plot(noUsersList, rsList_our_uniform, "-r", label="UniformRandom")
    plt.plot(noUsersList, rsList_our_groupfair, "-g", label="UniformRandomGroupFair")
    plt.legend(loc='best')
    plt.ylabel("r square score")
    plt.xlabel("#users")
    plt.savefig("r_square_plot.pdf", dpi=1000)
    plt.xticks([0, 10000, 20000, 30000], ["0", "10k", "20k", "30k"])
    plt.savefig("figures/rs_maxminfair_vs_uniform.pdf", dpi=1000)
    plt.show()

    # mae

    plt.plot(noUsersList, maeList_our_fairness, "-b", label="MaxMinFair")
    plt.plot(noUsersList, maeList_our_uniform, "-r", label="UniformRandom")
    plt.plot(noUsersList, maeList_our_groupfair, "-g", label="UniformRandomGroupFair")

    # plt.plot(noUsersList, maeList_our_groupfair,label = "group fair")
    plt.legend(loc='best')
    plt.ylabel("mean absolute error ")
    plt.xlabel("#users")
    # plt.savefig("mae_plot.pdf",dpi=1000)
    plt.xticks([0, 10000, 20000, 30000], ["0", "10k", "20k", "30k"])
    plt.savefig("figures/mae_maxminfair_vs_uniform.pdf", dpi=1000)
    plt.show()

    # item frequecny

    itemfreq = getItemFrequency(movieId_test, eligibleTopk)
    yvalues = list(itemfreq.values())
    yvalues.sort(reverse=True)
    xvalues = [i for i in range(0, len(movieId_test))]

    plt.bar(xvalues, yvalues, color='maroon', width=0.4)
    plt.xlabel("movie ids")
    plt.ylabel("frequency")
    # plt.legend(loc='best')
    plt.savefig(f"figures/movie_ids_vs_frequency_{theta}.pdf", dpi=1000)
    plt.show()




