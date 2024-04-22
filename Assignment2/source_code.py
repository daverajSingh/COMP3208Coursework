# Description: This file contains the source code for the assignment 2.
# Task Description:
# The source_code must code a large-scale matrix factorisation recommender system 
# algorithm to train and then predict ratings for a large (20M) set of items.

import numpy as np
# File Handling
def loadData(file):
    return np.genfromtxt(file, delimiter=',', dtype=np.float64)

def saveData(file, data):
    np.savetxt(file, data, delimiter=',')

def matrixToCSV(data, timestamps):
    rows, cols = data.shape
    dataCSV = []
    for i in range(rows):
        for j in range(cols):
            if(data[i, j] != 0):
                dataCSV.append([i+1, j+1, data[i, j], timestamps[i, j]])
    return np.array(dataCSV)

# Matrices
def timestampMatrix(data):
    timestamps = np.zeros(data.shape)
    for row in data:
        timestamps[row[0], row[1]] = row[3]
    return timestamps

def userItemMatrix(data, users, items):
    userItemMatrix = np.zeros((users, items))
    for row in data:
        userItemMatrix[int(row[0])-1, int(row[1])-1] = row[2].astype(np.float64)
    return userItemMatrix

def predictionMatrix(data, users, items):
    predictionMatrix = np.zeros((users, items))
    for row in data:
        predictionMatrix[int(row[0])-1, int(row[1])-1] = 1
    return predictionMatrix

# Matrix Factorisation
def matrixFactorisation(userItemMatrix, k, learningRate, iterations):
# Implement code here
    ratings = []
    return ratings

train = loadData('train_20M_withratings.csv')
test = loadData('test_20M_withoutratings.csv')

usersTrain = int(np.max(train[:, 0]))
itemsTrain = int(np.max(train[:, 1]))
usersTest = int(np.max(test[:, 0]))
itemsTest = int(np.max(test[:, 1]))
totalUsers = max(usersTrain, usersTest)
totalItems = max(itemsTrain, itemsTest)

userItemMatrixTrain = userItemMatrix(train, totalUsers, totalItems)
predictionMatrixTest = predictionMatrix(test, totalUsers, totalItems)
timestamps = timestampMatrix(train)

P, Q = matrixFactorisation(userItemMatrixTrain, 20, 0.001, 100)
predictedRatings = np.dot(P, Q.T)
predictedRatings = np.where(test == 1, predictedRatings)

ratings = matrixToCSV(predictedRatings, timestamps)
saveData('output.csv', ratings)

