# Description: This file contains the source code for the assignment 2.
# Task Description:
# The source_code must code a large-scale matrix factorisation recommender system 
# algorithm to train and then predict ratings for a large (20M) set of items.

import numpy as np
import sqlite3

#Setup the database

def setupDatabase(dbFile):
    """
    Set up the database with schema.

    returns:
    conn: connection to the database
    """
    conn = sqlite3.connect(dbFile)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS train')
    c.execute('DROP TABLE IF EXISTS test')
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
    """
    Creates a UserItemMatrix

    args:
    conn: connection to database 
    """
    c = conn.cursor()
    users, items = getMaxUserItemId(conn)
    userItemMatrix = np.zeros((users, items))
    for row in c.execute('SELECT * FROM train'):
        userItemMatrix[int(row[0])-1, int(row[1])-1] = row[2]
    return userItemMatrix
    
# Matrix Factorisation
def matrixFactorisation(userItemMatrix, learningRate, iterations, regularisation, NFactors=100):
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
    # Stochastic Gradient Descent:
    # Aims to minimise the error between the UI matrix and the given predictive UI matrix PQ. 
    # It achieves this by adjusting the user and item vectors by the computed error, learning rate and regularisation parameters as shown in line 110.
    # SGD was chosen instead of ALS as it is faster to run, providing better results more efficiently. This allows for good scalability and predictive accuracy.
    # Additionally, the nature of SGD allows it to avoid local minima due to the noisier updates, therefore allowing it to converge to a global minimum.
    # By changing the number of iterations, we can control how thorough our model learns from the data
    #   - More Iterations = More opportunities for the model to learn and redefine P and Q, potentially leading to better performance, risking overfitting.
    #   - Less Iterations = Less Computation time to learn the model and redefine P and Q, potentially leading to worse performance and underfitting the model.
    # By changing the learning rate, we can control the speed and stability of convergence.
    #   - Higher Learning Rate = Faster convergence at risk of overshooting the minima. This can lead to divergence/oscillation
    #   - Lower Learning Rate = Slower convergence, but more stable. This reduces the risk of overshooting.
    # By changing the regularisation paramater, we can prevent overfitting by penalising the larger weights
    #   - Higher Regularisation - Penalties have stronger magnitude on the weights, which can help to prevent overfitting
    #   - Lower Regularisation - Weaker penalty, allowing the model to fit the training data closely. This can cause overfitting if set too low.
    # NFactors - number of latent factors used in the factorization of the user item matrix.
    #   - More Factors = Higher model capacity, which can capture more complex interactions between usrrs and items. This increases the risk of overfitting and computational cost.
    #   - Less Factors = lower model capacity, which might lead to underfitting if the model is too simple to capture underlying trends and patterns in the data. This decreases computational cost.
    
    nUsers, nItems = userItemMatrix.shape
    
    # Randomly initialize user and item matrices
    q = np.random.rand(nItems, NFactors) * 0.1
    p = np.random.rand(nUsers, NFactors) * 0.1

    # SGD for given number of iterations
    for step in range(iterations):
        # Make a randomly shuffled list of user-item pairs
        for u in range(nUsers):
            for i in range(nItems):
                if userItemMatrix[u, i] > 0:
                    predictedRating = np.dot(p[u, :], q[i, :].T)
                    # Calculate the error
                    error = userItemMatrix[u, i] - predictedRating
                    # Update the user and item matrices
                    p[u, :] += learningRate * (error * q[i, :] - regularisation * p[u, :])
                    q[i, :] += learningRate * (error * p[u, :] - regularisation * q[i, :])
        # Calculates total error for convergence monitoring
        total_error = 0
        for u in range(nUsers):
            for i in range(nItems):
                if userItemMatrix[u, i] > 0:
                    total_error += (userItemMatrix[u, i] - np.dot(p[u, :], q[i, :].T)) ** 2
        print("Iteration", step, " - Total Error = ", total_error)
    return p, q


def getPredictions(P, Q, conn):
    """
    Extracts Predictions from P and Q, and then writes to csv

    args:
    P: user matrix
    Q: item matrix
    conn: connection to database
     
    """
    Q = Q.T
    predictedRatings = np.dot(P, Q)
    output = []
    c = conn.cursor()
    for row in c.execute("SELECT userId, itemId, timestamp FROM test"):
        userId = int(row[0]) 
        itemId = int(row[1])
        timestamp = int(row[2])
        predicted = np.round(predictedRatings[userId-1, itemId-1]*2)/2 # Rounds to nearest .5
        output.append((int(userId), int(itemId), predicted, int(timestamp)))
    
    np.savetxt('output.csv', output, delimiter=',', fmt='%d,%d,%.1f,%d')

def getMaxUserItemId(conn):
    """
    Returns the maximum userId and maximum itemId

    args: 
    conn: connection to the database

    returns:
    maximumUser, maximumItem: highest user and item id
    """

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
P, Q = matrixFactorisation(userItemMatrixTrain, learningRate=0.01, iterations=100, regularisation=0.01)
print("Matrix factorisation done.")
print("Fetching ratings...")
getPredictions(P, Q, conn)
print("Data saved.")