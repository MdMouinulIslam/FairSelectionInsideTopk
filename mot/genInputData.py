import random as rnd
import  pandas as pd



def readInput(fileName):
    df = pd.read_csv(fileName)
    X_train = []
    Y_train = []
    dataDict_train = {}
    movieId_train = []
    X_test = []
    Y_test = []
    dataDict_test = {}
    movieId_test = []
    for index, row in df.iterrows():
        if index % 2 == 0:
            movieId_train.append(row['movie_id'])
            X_train.append(row['count'])
            Y_train.append(row['avg_ratings'])
            dataDict_train[int(row['movie_id'])] = (row['count'], row['avg_ratings'])
        else:
            movieId_test.append(row['movie_id'])
            X_test.append(row['count'])
            Y_test.append(row['avg_ratings'])
            dataDict_test[int(row['movie_id'])] = (row['count'], row['avg_ratings'])

    g = 2
    groups_train = getGroupRandom(movieId_train, g)
    groups_test = getGroupRandom(movieId_test, g)
    inputDict = {"train":(X_train,Y_train,dataDict_train,movieId_train,groups_train),"test":(X_test,Y_test,dataDict_test,movieId_test,groups_test)}

    return inputDict

def genData(n,g):
    X = []
    Y = []
    movieId = []
    dataDict = {}
    groupId = {}
    for i in range(0,n):
        key = rnd.randint(0,n)
        val = (key*1 + 200 + rnd.randint(0,1)) / (n*2)
        X.append(key)
        Y.append(val)
        groupId[i] = rnd.randint(1,g)
        movieId.append(i)
        dataDict[i] = (key,val)
    return dataDict,movieId,groupId,X,Y

def getGroupRandom(movieIds,g):
    groups = {}
    for m in movieIds:
        groups[m] = rnd.randint(1,g)
    return groups


def readInputLTR(fileName):
    df = pd.read_csv(fileName)
    X_train = []
    Y_train = []
    dataDict_train = {}
    movieId_train = []
    X_test = []
    Y_test = []
    dataDict_test = {}
    movieId_test = []
    for index, row in df.iterrows():
        # print(index)
        if index % 2 == 1:
            movieId_train.append(int(row['movie_id_left']))
            X_train.append([row['count'], row['Horror'], row['Action']])
            Y_train.append(row['avg_ratings'])
            dataDict_train[int(row['movie_id_left'])] = (X_train[int(index / 2)], Y_train[int(index / 2)])
        else:
            movieId_test.append(int(row['movie_id_left']))
            X_test.append([row['count'], row['Horror'], row['Action']])
            Y_test.append(row['avg_ratings'])
            dataDict_test[int(row['movie_id_left'])] = (X_test[int(index / 2)], Y_test[int(index / 2)])

    g = 2
    groups_train = getGroupRandom(movieId_train, g)
    groups_test = getGroupRandom(movieId_test, g)
    inputDict = {"train": (X_train, Y_train, dataDict_train, movieId_train, groups_train),
                 "test": (X_test, Y_test, dataDict_test, movieId_test, groups_test)}

    return inputDict

def readGroup(inputGroupFile):
    groupall = pd.read_csv(inputGroupFile)
    groupDict = {}
    for index, row in groupall.iterrows():
        groupDict[row['movie_id']] = row['group']
    return groupDict
