from genInputData import genData,readInput,readInputLTR,readGroup
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from utils import plot
from utils import createDataPairs
from utils import findEligibleCandidates
from  utils import getEligibleTopK

from  exp_1 import  getItemProbExp1
from exp_4 import  getItemProbExp4
from exp_2 import  getItemProbExp2,getItemFrequency
from exp_3 import getItemProbExp3
from utils import normalize,saveToCsv
from sklearn import preprocessing
import pandas as pd

import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin
import lightgbm as lgb

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split



def runLTR():
    inputFile = 'data/movie_80.csv'
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
    mae = mean_absolute_error(y_test,y_pred)
    #print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)
    print("MAE: %.2f" %mae)







def runRealV1():
        ############################# input data ########################################

        # dataDict_train,movieId_train,group_train,X_train,Y_train = genData(n,g)
        # dataDict_test,movieId_test,group_test,X_test,Y_test = genData(n,g)
        noUsersList = [0,30000]
        maeList_our_fairness = []
        maeList_our_uniform = []
        maeList_our_groupfair = []
        maeList_exp_fairness=[]

        rsList_our_fairness = []
        rsList_our_uniform = []
        rsList_our_groupfair = []
        rsList_exp_fairness = []


        # fig1, ax1 = plt.subplots(2, len(noUsersList))
        # fig2, ax2 = plt.subplots(1, len(noUsersList))
        # fig3, ax3 = plt.subplots(2, len(noUsersList))
        # fig4, ax4 = plt.subplots(1, len(noUsersList))
        #fig,ax5 = plt.subplots(1, len(noUsersList))
        ourFairClicks = []
        uniformClicks = []
        uniGropuClick = []
        fairExposureClick = []

        for  it in range(0,len(noUsersList)):
                noUsers = noUsersList[it]
                inputFile = 'data/movie_88.csv'
                inputGroupFile = 'data/group_all.csv'

                X_train, Y_train, dataDict_train, movieId_train, group_train = readInputLTR(inputFile)["train"]
                X_test, Y_test, dataDict_test, movieId_test, group_test = readInputLTR(inputFile)["test"]
                groups = readGroup(inputGroupFile)
                for i,v in groups.items():
                        if v != 0:
                                groups[i] = 1
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

                model = lgb.LGBMRegressor(random_state=1, num_leaves =6000, n_estimators=1,num_boost_round=500, max_depth=300, learning_rate = 0.002, n_jobs=8) #lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
                model.fit(x_train,y_train)
                # model = lgb.train(params,
                #                   train_set=lgb_train,
                #                   valid_sets=lgb_eval,
                #                   early_stopping_rounds=30)

                # prediction
                y_pred = model.predict(x_test)

                # accuracy check
                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** (0.5)
                mae = mean_absolute_error(y_test, y_pred)
                # print("MSE: %.2f" % mse)
                print("RMSE: %.2f" % rmse)
                print("MAE: %.2f" % mae)

                ############################## input param ################################
                #n = 25
                n = len(Y_test)
               # theta = 0.3
                theta = 0.3
                k = 3

                g = 2


                ################################## train on train data ####################################

                # model = LinearRegression()
                # x_train = np.array(X_train).reshape((-1, 1))
                # y_train = np.array(Y_train)
                # x_train = normalize(x_train)
                # model.fit(x_train, y_train)
                # r_sq = model.score(x_train, y_train)
                #
                # y_pred_train = model.predict(x_train)
                # mae_train = mean_absolute_error(y_train,y_pred_train)
                #print(f"coefficient of determination train: {r_sq * 100:.2f}%")
                #print(f"MAE train: {mae_train * 100:.2f}%")



                ################################# test data gen ###########################################



                # x_test = np.array(X_test).reshape((-1, 1))
                # y_test = np.array(Y_test)
                # X = [i[0] for i in x_test]
                # corel = np.corrcoef(X, y_test)
                # print("corrcoef = ",corel)
                # x_test = normalize(x_test)
                # r_sq = model.score(x_test, y_test)
                # y_pred_test = model.predict(x_test)
                # mae_test = mean_absolute_error(y_test, y_pred_test)
                # #print(f"coefficient of determination test: {r_sq*100:.2f}%")
                # print(f"MAE test: {mae_test * 100:.2f}%")
                # if it == 0:
                #         maeList_our_fairness.append(mae_test)
                #         maeList_our_uniform.append(mae_test)
                #         maeList_our_groupfair.append(mae_test)

                ############################### experiments ###########################################

                dataPair = createDataPairs(X_test,Y_test,movieId_test)
                movieEligible = findEligibleCandidates(dataPair,theta,n,k)
                eligibleTopk = getEligibleTopK(dataDict_test,dataPair,movieEligible, theta,k)



                ############################# exp 1 ##############################################

                x_exp1,y_exp1,item_probs1 = getItemProbExp1(dataDict_test,movieId_test,eligibleTopk,noUsers,groups)

                x_exp1 = normalize(x_exp1)

                #ourFairClicks.append(x_exp1[0])

                # print("before clicks ", x_test)
                # print("after clicks ", x_exp1)






                y_exp1_pred = model.predict(x_exp1)
                mae_test = mean_absolute_error(y_test, y_exp1_pred)
                r_sq = r2_score(y_test, y_exp1_pred)
                rsList_our_fairness.append(r_sq)

                #print(f"coefficient of determination our fairness: {r_sq*100:.2f}%")
                print(f"MAE our fairness {noUsersList[it]} = : {mae_test * 100:.2f}%")
                maeList_our_fairness.append(mae_test)

                # plt.figure(it)
                # plt.subplot(1, 3, 1)
                # plt.xlabel('movie id')
                # plt.ylabel('#click addition')
                # title = "our fairness " + str(noUsersList[it])
                # plt.title(title)
                # xvalues = [i for i in range(0,len(item_probs1.keys()))]
                # plt.bar(xvalues,item_probs1.values(), color ='maroon',width = 0.4)

                ############################# exp 2 ##############################################


                x_exp2,y_exp2,item_probs2 = getItemProbExp2(dataDict_test,movieId_test,eligibleTopk,noUsers)

                x_exp2 = normalize(x_exp2)
                #uniformClicks.append(x_exp2)

                #r_sq = model.score(x_exp2,y_exp2 )
                #rsList_our_uniform.append(r_sq)
                y_exp2_pred = model.predict(x_exp2)
                mae_test = mean_absolute_error(y_test, y_exp2_pred)
                maeList_our_uniform.append(mae_test)
                r_sq = r2_score(y_test, y_exp2_pred)
                rsList_our_uniform.append(r_sq)



                # print(f"coefficient of determination uniform: {r_sq*100:.2f}%")
                # print(f"MAE uniform: {mae_test * 100:.2f}%")


                # plt.subplot(1, 3, 2)
                # xvalues = [i for i in range(0,len(item_probs2.keys()))]
                # plt.bar(xvalues,item_probs2.values(), color ='maroon',width = 0.4)
                # plt.xlabel('movie id')
                # plt.ylabel('#click addition')
                # title = "uniform " + str(noUsersList[it])
                # plt.title(title)

                print("############## user = ", noUsers, " ####################")
                output = {}
                # before = []
                # after_our = []
                # after_unifor = []
                # for i in range(0, len(x_test)):
                #         before.append(f"{x_test[i][0]:.2f}")
                #         after_our.append(f"{x_exp1[i][0]:.2f}")
                #         after_unifor.append(f"{x_exp2[i][0]:.2f}")
                #
                # df = pd.DataFrame({'before_our': before, 'after_our': after_our,'after_uniform': after_unifor})
                # df.to_csv(f"user_{noUsers}.csv")
                ############################# exp 2 ##############################################

                x_exp3,y_exp3,item_probs3 = getItemProbExp3(dataDict_test,movieId_test,eligibleTopk,noUsers,groups)

                x_exp3 = normalize(x_exp3)
                # r_sq = model.score(x_exp3,y_exp3 )
                # rsList_our_groupfair.append(r_sq)
                y_exp3_pred = model.predict(x_exp3)
                mae_test = mean_absolute_error(y_test, y_exp3_pred)
                maeList_our_groupfair.append(mae_test)
                r_sq = r2_score(y_test, y_exp3_pred)
                rsList_our_groupfair.append(r_sq)


                # print(f"coefficient of determination uniform with group fairness: {r_sq*100:.2f}%")
                # print(f"MAE uniform with group fairness: {mae_test * 100:.2f}%")


                # plt.subplot(1, 3, 3)
                # xvalues = [i for i in range(0,len(item_probs3.keys()))]
                # plt.bar(xvalues,item_probs3.values(), color ='maroon',width = 0.4)
                # plt.xlabel('movie id')
                # plt.ylabel('#click addition')
                # title = "group " + str(noUsersList[it])
                # plt.title(title)
                # figName = "clickAddition_" +str(it)+".pdf"
                # plt.savefig(figName)

                #plt.show()
                ################################## train datae plot ########################################


                # if it == 0:
                #         plt.figure(1000)
                #         plt.subplot(2, len(noUsersList)+1, 1)
                #         xvalues = [i for i in range(0, len(item_probs3.keys()))]
                #         yvalues = [i[0] for i in x_test]
                #         plt.bar(xvalues, yvalues, color='maroon', width=0.4)
                #         plt.xlabel('#click')
                #         plt.ylabel('movieIds')
                #         plt.title("original test")

                ############################# exp 4 ##############################################

                x_exp4, y_exp4, item_probs4 = getItemProbExp4(dataDict_test, movieId_test, eligibleTopk, noUsers,
                                                              groups)

                x_exp4 = normalize(x_exp4)

                # ourFairClicks.append(x_exp1[0])

                # print("before clicks ", x_test)
                # print("after clicks ", x_exp1)

                y_exp4_pred = model.predict(x_exp4)
                mae_test = mean_absolute_error(y_test, y_exp4_pred)
                r_sq = r2_score(y_test, y_exp4_pred)
                rsList_exp_fairness.append(r_sq)

                # print(f"coefficient of determination our fairness: {r_sq*100:.2f}%")
                print(f"MAE our fairness {noUsersList[it]} = : {mae_test * 100:.2f}%")
                maeList_exp_fairness.append(mae_test)

                # plt.figure(it)
                # plt.subplot(1, 3, 1)
                # plt.xlabel('movie id')
                # plt.ylabel('#click addition')
                # title = "our fairness " + str(noUsersList[it])
                # plt.title(title)
                # xvalues = [i for i in range(0,len(item_probs1.keys()))]
                # plt.bar(xvalues,item_probs1.values(), color ='maroon',width = 0.4)


                ######################################################
                x_exp1_click = [i[0] for i in x_exp1]
                x_exp2_click = [i[0] for i in x_exp2]
                x_exp3_click = [i[0] for i in x_exp3]
                x_exp4_click = [ [i[0] for i in x_exp4]]
                ourFairClicks.append(x_exp1_click)
                uniformClicks.append(x_exp2_click)
                uniGropuClick.append(x_exp3_click)
                fairExposureClick.append(x_exp4_click)

                #######################################################
                # if it == 0:
                #         title = "beginning"
                #         ax1[0][it].set_ylabel('#normalized clicks')
                #         ax1[1][it].set_ylabel('#added clicks')
                #         ax3[1][it].set_ylabel('#added clicks')
                #         ax3[0][it].set_ylabel('#normalized clicks')
                # else:
                #         title = "after " + str(int(noUsersList[it]/1000))+"k"
                #
                #
                # xvalues = [i for i in range(0, len(item_probs1.keys()))]
                # yvalues = [i[0] for i in x_exp1]
                # ax1[0][it].bar(xvalues, yvalues, color='maroon', width=0.4)
                # #ax1[0][it].set_xlabel('movie ids')
                #
                # ax1[0][it].set_title(title)
                #
                # xvalues = [i for i in range(0, len(item_probs1.keys()))]
                # yvalues = item_probs1.values()
                # ax1[1][it].bar(xvalues, yvalues, color='maroon', width=0.4)
                # #ax1[1][it].set_xlabel('movie ids')
                #
                # ax1[1][it].set_title(title)
                # ###########################################################
                # xvalues = [i for i in range(0, len(item_probs2.keys()))]
                # yvalues = [i[0] for i in x_exp2]
                # ax3[0][it].bar(xvalues, yvalues, color='maroon', width=0.4)
                # #ax3[0][it].set_xlabel('movie ids')
                #
                # ax3[0][it].set_title(title)
                #
                # xvalues = [i for i in range(0, len(item_probs2.keys()))]
                # yvalues = item_probs2.values()
                # ax3[1][it].bar(xvalues, yvalues, color='maroon', width=0.4)
                # #ax3[1][it].set_xlabel('movie ids')
                #
                # ax3[1][it].set_title(title)
                # ##############################################################
                #
                #
                # ###############################################################
                #
                # ax2[it].plot(x_exp1_click, y_exp1, ".")
                # ax2[it].plot(x_exp1_click, y_exp1_pred)
                # #ax2[it].set_xlabel('#normalized clicks')
                # #ax2[it].set_ylabel('predicted ratings')
                # ax2[it].set_title(title)
                # ##############################################################
                # ax4[it].plot(x_exp2_click, y_exp2, ".")
                # ax4[it].plot(x_exp2_click, y_exp2_pred)
                # #ax4[it].set_xlabel('#normalized clicks')
                # #ax4[it].set_ylabel('predicted ratings')
                # ax4[it].set_title(title)

        # fig1.suptitle("our fairness")
        # fig2.suptitle("our fairness")
        # fig3.suptitle("uniform")
        # fig4.suptitle("uniform")
        #plt.subplots_adjust(wspace=0.8, hspace=0.8, left=0.8, bottom=0.8, right=0.8, top=0.8)





        # fig2.supxlabel('#normalized clicks')
        # fig2.supylabel('ratings')
        #
        # fig4.supxlabel('#normalized clicks')
        # fig4.supylabel('ratings')
        #
        # plt.tight_layout()
        # plt.rcParams.update({'font.size': 10})
        # #fig1.savefig("figures/maxminfair_click_vs_movieids.pdf",dpi=1000)
        # fig2.savefig("figures/uniform_click_vs_movieids.pdf",dpi=1000)
        # #fig3.savefig("figures/maxminfair_ratins_vs_clicks.pdf",dpi=1000)
        # fig4.savefig("figures/uniform_ratins_vs_clicks.pdf",dpi=1000)
        # plt.show()

        plt.rcParams.update({'font.size': 15})

        ###########################
        movieIds = [i for i in range(0, len(item_probs1.keys()))]
        labels = ["beginning", "after 30k"]
        for i in range(0, len(fairExposureClick)):
                clicks = fairExposureClick[i][0]
                plt.plot(movieIds, clicks, label=labels[i])
        plt.legend(loc='best')
        plt.xlabel("movie ids")
        plt.ylabel("#normalized clicks")
        plt.savefig("figures/maxminfair_normalized_clicks_vs_movie_ids.pdf", dpi=1000)
        plt.show()
        #############################
        movieIds  = [i for i in range(0, len(item_probs1.keys()))]
        labels = ["beginning","after 30k"]
        for i in range(0,len(ourFairClicks)):
                clicks = ourFairClicks[i]
                plt.plot(movieIds,clicks,label=labels[i])
        plt.legend(loc='best')
        plt.xlabel("movie ids")
        plt.ylabel("#normalized clicks")
        plt.savefig("figures/maxminfair_normalized_clicks_vs_movie_ids.pdf",dpi=1000)
        plt.show()

        ###save####
        filename = "result/normalized_clicks_ours.csv"
        values = []
        values.append(movieIds)
        values.extend(ourFairClicks)
        values.extend([uniGropuClick[1]])
        columnNames = ['movie ids', 'beginning','after 30k ours','after 30k group']
        saveToCsv(values, columnNames, filename)

        filename = "result/normalized_clicks_fair_exp.csv"
        values = []
        values.append(movieIds)
        values.extend(ourFairClicks)
        values.extend(fairExposureClick[1])
        columnNames = ['movie ids', 'beginning', 'after 30k ours','after 30k fairexp']
        saveToCsv(values, columnNames, filename)



        for i in range(0, len(ourFairClicks)):
                clicks = uniformClicks[i]
                plt.plot(movieIds, clicks, label=labels[i])
        plt.xlabel("movie ids")
        plt.ylabel("#normalized clicks")
        plt.legend(loc='best')
        plt.savefig("figures/uniform_normalized_clicks_vs_movie_ids.pdf",dpi=1000)
        plt.show()


        for i in range(0, len(uniGropuClick)):
                clicks = uniGropuClick[i]
                plt.plot(movieIds, clicks, label=labels[i])
        plt.xlabel("movie ids")
        plt.ylabel("#normalized clicks")
        plt.legend(loc='best')
        plt.savefig("figures/uniform_fair_clicks_vs_movie_ids.pdf",dpi=1000)
        plt.show()

        ###save####
        filename = "result/normalized_clicks_groups.csv"
        values = []
        values.append(movieIds)
        values.extend(uniGropuClick)
        columnNames = ['movie ids', 'beginning', 'after 30k']
        saveToCsv(values, columnNames, filename)




        # plt.figure(1000)
        # plt.savefig("click_vs_movieIds.pdf")
        # plt.show()
        #
        # plt.figure(2000)
        # plt.savefig("click_vs_ratings.pdf")
        # plt.show()


        #rs
        plt.plot(noUsersList, rsList_our_fairness, "-b", label = "MaxMinFair")
        plt.plot(noUsersList, rsList_our_uniform, "-r", label = "UniformRandom")
        plt.plot(noUsersList, rsList_our_groupfair,"-g", label="UniformRandomGroupFair")
        plt.legend(loc='best')
        plt.ylabel("r square score")
        plt.xlabel("#users")
        plt.savefig("r_square_plot.pdf",dpi=1000)
        plt.xticks([0,30000],["0",  "30k"])
        plt.savefig("figures/rs_maxminfair_vs_uniform.pdf", dpi=1000)
        plt.show()


        #mae

        plt.plot(noUsersList,maeList_our_fairness,"-b",label = "MaxMinFair")
        plt.plot(noUsersList, maeList_our_uniform, "-r", label="UniformRandom")
        plt.plot(noUsersList, maeList_our_groupfair, "-g", label="UniformRandomGroupFair")



        #plt.plot(noUsersList, maeList_our_groupfair,label = "group fair")
        plt.legend(loc='best')
        plt.ylabel("mean absolute error ")
        plt.xlabel("#users")
        #plt.savefig("mae_plot.pdf",dpi=1000)
        plt.xticks([0,30000],["0", "30k"])
        plt.savefig("figures/mae_maxminfair_vs_uniform.pdf", dpi=1000)
        plt.show()

        ###save####
        filename  = "result/mae_maxminfair_vs_uniform.csv"
        values = [noUsersList,maeList_our_fairness,maeList_our_uniform,maeList_our_groupfair]
        columnNames = ['#users','MaxMinFair','UniformRandom','UniformRandomGroupFair']
        saveToCsv(values,columnNames,filename)

        #item frequecny

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

        ###save####
        filename = "result/movie_ids_vs_frequency.csv"
        values = [xvalues, yvalues]
        columnNames = ['movie ids', 'frequency']
        saveToCsv(values, columnNames, filename)




        # plot(X_train,Y_train,y_pred_train,"train data")
        # plot(x_exp2,y_exp2,y_exp2_pred,"random")





