# Description: This file contains the source code for the assignment 2.
# Task Description:
# The source_code must code a large-scale matrix factorisation recommender system 
# algorithm to train and then predict ratings for a large (20M) set of items.

import numpy as np
import sqlite3

#Setup the database

def setupDatabase(dbFile):
    """
    Set up the database with the given schema.
    
    returns:
    conn: connection to the database
    """
    conn = sqlite3.connect(dbFile)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS train (userId INT, itemId INT, rating REAL, timestamp INT)')
    c.execute('CREATE TABLE IF NOT EXISTS test (userId INT, itemId INT, timestamp INT)')
    conn.commit()
    return conn


def loadData(file, conn):
    """
    Load data to the database.
    
    args:
    dbFile: path to the database file
    """
    
    c = conn.cursor()
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            if len(line) == 4:
                c.execute('INSERT INTO train VALUES (?, ?, ?, ?)', (line[0], line[1], line[2], line[3]))
            elif len(line) == 3:
                c.execute('INSERT INTO test VALUES (?, ?, ?)', (line[0], line[1], line[2]))
    conn.commit()
    

def userItemMatrix(conn):
    c = conn.cursor()
    users, items = getMaxUserItemId(conn)
    userItemMatrix = np.zeros((users, items))
    for row in c.execute('SELECT * FROM train'):
        userItemMatrix[int(row[0])-1, int(row[1])-1] = row[2]
    return userItemMatrix
    
# Matrix Factorisation
def matrixFactorisation(userItemMatrix, learningRate, iterations, regularisation, NFactors=40):
    """
    Performs matrix factorisation using stochastic gradient descent.
    
    args:
    userItemMatrix: matrix with users as rows and items as columns
    learningRate: learning rate
    iterations: number of iterations
    regularisation: regularisation parameter
    
    returns:
    P: user matrix
    Q: item matrix
    """
    nUsers, nItems = userItemMatrix.shape
    
    # Randomly initialize user and item matrices
    q = np.random.rand(nItems, NFactors)
    p = np.random.rand(nUsers, NFactors)
    
    # SGD for given number of steps
    for step in range(iterations):
        # Make a randomly shuffled list of user-item pairs
        indices = np.arange(nUsers)
        np.random.shuffle(indices)
        #Loop on each user-item pair
        for i in indices:
            for j in range(nItems):
                if userItemMatrix[i, j] > 0:
                    predictedRating = np.dot(p[i, :], q[j, :])
                    # Calculate the error
                    error = userItemMatrix[i, j] - predictedRating
                    # Update the user and item matrices
                    p[i, :] += learningRate * (error * q[j, :] - regularisation * p[i, :])
                    q[j, :] += learningRate * (error * p[i, :] - regularisation * q[j, :])
        print("Iteration", step, " - Error = ", error) # Print the error for each iteration
    return p, q


def getPredictions(P, Q, conn):
    Q = Q.T
    predictedRatings = np.dot(P, Q)
    c = conn.cursor()
    for row in c.execute("SELECT userId, itemId, timestamp FROM test"):
        userId = int(row[0]) - 1
        itemId = int(row[1]) - 1
        timestamp = row[2]
        predictedRatings.append((userId, itemId, predictedRatings[userId, itemId], timestamp))
    
    np.savetxt('output.csv', predictedRatings, delimiter=',')

def getMaxUserItemId(conn):
    c = conn.cursor()
    maxTrainUser = c.execute('SELECT MAX(userId) FROM train').fetchone()[0]
    maxTestUser = c.execute('SELECT MAX(userId) FROM test').fetchone()[0]
    maxTrainItem = c.execute('SELECT MAX(itemId) FROM train').fetchone()[0]
    maxTestItem = c.execute('SELECT MAX(itemId) FROM test').fetchone()[0]
    print(maxTrainUser, maxTestUser, maxTrainItem, maxTestItem)
    return max(maxTrainUser, maxTestUser), max(maxTrainItem, maxTestItem)   
    

print("Loading data...")
conn = setupDatabase('work.db')
print("Database setup.")
loadData('test_20M_withoutratings.csv', conn)
print("Test data loaded.")
loadData('train_20M_withratings.csv', conn)
print("Train data loaded.")
print("Making User Item Matrix...")
userItemMatrixTrain = userItemMatrix(conn)
print("Matrix made.")
print("Starting matrix factorisation...")
P, Q = matrixFactorisation(userItemMatrixTrain, learningRate=0.1, iterations=10, regularisation=0.0)
print("Matrix factorisation done.")
print("Fetching ratings...")
getPredictions(P, Q, conn)
print("Data saved.")