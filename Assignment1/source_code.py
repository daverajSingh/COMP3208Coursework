import numpy as np
from collections import defaultdict

# Load data from file
def loadData(file):
    return np.genfromtxt(file, delimiter=',', dtype=int)

# Compute cosine similarity between items
def cosineSimilarityItem(userItemMatrix):
    dotProduct = np.dot(userItemMatrix.T, userItemMatrix)
    magnitude = np.sqrt(np.sum(userItemMatrix.T ** 2, axis=1))
    denominator = np.outer(magnitude, magnitude)
    denominator[denominator == 0] = 1
    similarityMatrix = dotProduct / denominator
    np.fill_diagonal(similarityMatrix, 1)
    return similarityMatrix

# Compute cosine similarity between users
def cosineSimilarityUser(userItemMatrix):
    userItemMatrixNoNaN = np.nan_to_num(userItemMatrix)
    dotProduct = np.dot(userItemMatrixNoNaN, userItemMatrixNoNaN.T)
    magnitude = np.sqrt(np.sum(userItemMatrixNoNaN ** 2, axis=1))
    denominator = np.outer(magnitude, magnitude)
    denominator[denominator == 0] = 1
    similarityMatrix = dotProduct / denominator
    np.fill_diagonal(similarityMatrix, 1)
    return similarityMatrix


# Predict the rating of an item given user similarities
def userPrediction(userSim, userItemMatrix, i, j, n=74):
    usersWhoRatedItem = np.where(~np.isnan(userItemMatrix[:, j]))[0]
    userSimilarites = userSim[i, :][usersWhoRatedItem]
    
    neighbourhoodIndices = np.where(((userSimilarites > 0.3)) & (userSimilarites != 0))[0]
    sortedUserIndexes = np.argsort(-np.abs(userSimilarites[neighbourhoodIndices]))
    
    sortedUserIds = usersWhoRatedItem[neighbourhoodIndices][sortedUserIndexes]
    sortedIndices = np.array(sortedUserIds[:n])
    topIndices = sortedIndices[sortedIndices != i]
    
    neighbourhood = [userSim[index, i] for index in topIndices]
    
    numerator = denominator = 0
    
    for neighbour in topIndices:
        similarity = userSim[neighbour, i]
        rating = userItemMatrix[neighbour, j]
        numerator += similarity * rating
        denominator += abs(similarity)
        
    if denominator == 0 or numerator == 0:
        return 0, 0.7
    
    weight = 0 
    neighbourhoodSize = len(neighbourhood)
    if neighbourhoodSize == 0:
        weight = 0
    elif neighbourhoodSize > 40:
        weight = 1     
    else:     
        0.5 + 0.5 * np.exp(-0.16 * neighbourhoodSize)
    
    pred = numerator / denominator
    pred = 0 if np.isnan(pred) else pred
    return pred, weight

# Predict the rating of an item given item similarities
def itemPrediction(itemSim, userItemMatrix, i, j, n=115):
    similarities = itemSim[j, :]
    similarities = [-50 if np.isnan(x) else similarities[i] for i, x in enumerate(userItemMatrix[i, :])]
    similarities = [-50 if ((v > 0.3) or v == 0 or v == -50) else v for v in similarities]
    
    sortedIndices = sorted((index for index, value in enumerate(similarities) if value != -50), key=lambda i: abs(similarities[i]), reverse=True)
    sortedIndices = np.array(sortedIndices[:n])
    topIndices = sortedIndices[sortedIndices != j]
    
    neighbourhood = [itemSim[index, j] for index in topIndices]
    
    numerator = denominator = 0
    
    for neighbour in topIndices:
        similarity = itemSim[neighbour, j]
        rating = userItemMatrix[i, neighbour]
        numerator += similarity * rating
        denominator += abs(similarity)
        
    if denominator == 0 or numerator == 0:
        return 0, 0.3
    
    weight = 0 
    neighbourhoodSize = len(neighbourhood)
    if neighbourhoodSize == 0:
        weight = 0
    elif neighbourhoodSize > 40:
        weight = 1     
    else:     
        0.2 + 0.8 * np.exp(-0.16 * neighbourhoodSize)
    
    pred = numerator / denominator
    pred = 0 if np.isnan(pred) else pred
    return pred, weight

# Load data
trainData = loadData("train_100k_withratings.csv")
testData = loadData("test_100k_withoutratings.csv")
testData = [list(map(int, values)) for values in testData if len(values) == 3]

# Create user-item matrix
allUserIdsArray = np.array(list(range(1, 944)))
allItemIdsArray = np.array(list(range(1, 1683)))
nUsers = len(allUserIdsArray)
nItems = len(allItemIdsArray)

userItemRatingsDict = defaultdict(dict)
trainArray = np.array(trainData)
for userId, itemId, rating in trainArray[:, :3].astype(np.float16):
    userItemRatingsDict[userId][itemId] = rating
    
userItemMatrixTrain = np.full((nUsers, nItems), np.nan)
for i, usr in enumerate(allUserIdsArray):
    itemsForUser = userItemRatingsDict[usr]
    indices = np.searchsorted(allItemIdsArray, list(itemsForUser.keys()))
    userItemMatrixTrain[i, indices] = list(itemsForUser.values())

# Normalize user-item matrix
userMeans = np.nanmean(userItemMatrixTrain, axis=1)
normalizedUserItemMatrix = userItemMatrixTrain.copy()
nanMask = np.isnan(userItemMatrixTrain)
for u in range(nUsers):
    normalizedUserItemMatrix[u, ~nanMask[u]] -= userMeans[u]

userItemMatrixTrain = normalizedUserItemMatrix
userSimMatrix = cosineSimilarityUser(userItemMatrixTrain)
itemSimMatrix = cosineSimilarityItem(userItemMatrixTrain)

# Predict ratings 
predictedRatings = {}
for userId, itemId, timestamp in testData:
    userPred, userQuality = userPrediction(userSimMatrix, userItemMatrixTrain, userId-1, itemId-1) + userMeans[userId-1]
    itemPred, itemQuality = itemPrediction(itemSimMatrix, userItemMatrixTrain, userId-1, itemId-1) + userMeans[userId-1]
    
    pred = ((userPred * userQuality) + (itemPred * itemQuality)) / (userQuality + itemQuality)
    predictedRatings[(userId, itemId)] = (round(pred), timestamp)
    
# Write predictions to file
with open("output.csv", "w") as file:
    for (userId, itemId), (pred, timestamp) in predictedRatings.items():
        file.write(f"{userId},{itemId},{pred},{timestamp}\n")

print("Done")    