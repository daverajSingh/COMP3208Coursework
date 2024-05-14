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
def timestampMatrix(data, users, items):
    timestamps = np.zeros((users, items))
    for row in data:
        timestamps[int(row[0])-1, int(row[1])-1] = row[3].astype(np.uint64)
    return timestamps

def userItemMatrix(data, users, items):
    userItemMatrix = np.zeros((users, items))
    for row in data:
        userItemMatrix[int(row[0])-1, int(row[1])-1] = row[2].astype(np.float64)
    return userItemMatrix

# Matrix Factorisation
def matrixFactorisation(userItemMatrix, k, learningRate, iterations, regularisation):
    """
    Performs matrix factorisation using stochastic gradient descent.
    
    args:
    userItemMatrix: matrix with users as rows and items as columns
    k: number of latent features
    learningRate: learning rate
    iterations: number of iterations
    regularisation: regularisation parameter
    
    returns:
    P: user matrix
    Q: item matrix
    """
    N = len(userItemMatrix)
    M = len(userItemMatrix[0])
    
    # Randomly initialize user and item matrices
    P = np.random.rand(N, k)
    Q = np.random.rand(M, k)
    
    # Transpose Q for easier dot product calculation
    Q = Q.T
    
    # SGD for given number of steps
    for step in range(iterations):
        for u in range(N):
            for i in range(M):
                if userItemMatrix[u, i] > 0:  # Only consider non-zero ratings
                    eui = userItemMatrix[u, i] - np.dot(P[u, :], Q[:, i])
                    for t in range(k):
                        P[u, t] += learningRate * (2 * eui * Q[k, i] - regularisation * P[u, t])
                        Q[t, i] += learningRate * (2 * eui * P[u, k] - regularisation * Q[t, i])
        
        # Calculate the error
        e = 0
        for u in range(N):
            for i in range(M):
                if userItemMatrix[u, i] > 0:
                    e += pow(userItemMatrix[u, i] - np.dot(P[u, :], Q[:, i]), 2)
                    for t in range(k):
                        e += (regularisation / 2) * (pow(P[u, t], 2) + pow(Q[t, i], 2))
        
        if e < 0.001:
            break
    
    return P, Q.T

def getPredictions(predictedRatings, testData):
    """
    Get the predicted ratings for the test data.
    
    args:
    predictedRatings: matrix with predicted ratings
    testData: test data
    
    returns:
    ratings: predicted ratings for the test data
    """
    ratings = np.zeros(testData.shape)
    for row in testData:
        ratings[row[0], row[1]] = predictedRatings[row[0]-1, row[1]-1]
    return ratings

print("Loading data...")

train = loadData('Assignment2/train_20M_withratings.csv')
test = loadData('Assignment2/test_20M_withoutratings.csv')

print("Data loaded.")

usersTrain = int(np.max(train[:, 0]))
itemsTrain = int(np.max(train[:, 1]))
usersTest = int(np.max(test[:, 0]))
itemsTest = int(np.max(test[:, 1]))
totalUsers = max(usersTrain, usersTest)
totalItems = max(itemsTrain, itemsTest)

print("Making matrices...")

userItemMatrixTrain = userItemMatrix(train, totalUsers, totalItems)
timestamps = timestampMatrix(train, totalUsers, totalItems)

print("Matrices made.")
print("Starting matrix factorisation...")

P, Q = matrixFactorisation(userItemMatrixTrain,)

print("Matrix factorisation done.")
print("Fetching ratings...")

predictedRatings = np.dot(P, Q.T)
predictedRatings = getPredictions(predictedRatings, test)

print("Ratings fetched.")
print("Saving data...")

ratings = matrixToCSV(predictedRatings, timestamps)
saveData('output.csv', ratings)

print("Data saved.")
