import numpy as np
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

#Compute Prediction of rating - User or Item based
def prediction(similarityMatrix, userItemMatrix, user, item, isUser):
    if isUser:
        n=74
        usersWhoRatedItem = np.where(~np.isnan(userItemMatrix[:, item]))[0]
        userSimilarites = similarityMatrix[user, :][usersWhoRatedItem]
        
        neighbourhoodIndices = np.where(((userSimilarites > 0.3)) & (userSimilarites != 0))[0]
        sortedUserIndexes = np.argsort(-np.abs(userSimilarites[neighbourhoodIndices]))
        
        sortedUserIds = usersWhoRatedItem[neighbourhoodIndices][sortedUserIndexes]
        sortedIndices = np.array(sortedUserIds[:n])
        topIndices = sortedIndices[sortedIndices != user]
        
        neighbourhood = [similarityMatrix[index, user] for index in topIndices]
        numerator = denominator = 0    

        for neighbour in topIndices:
            numerator += similarityMatrix[neighbour, user] * userItemMatrix[neighbour, item]
            denominator += abs(similarityMatrix[neighbour, user])
    else:
        n=115
        similarities = similarityMatrix[item, :]
        similarities[np.isnan(userItemMatrix[user, :])] = -50
        mask = (similarities > 0.3) | (similarities == 0) | (similarities == -50) 
        similarities[mask] = -50
                
        validMask = similarities != -50
        validIndices = np.arange(len(similarities))[validMask]
        sortedIndices = validIndices[np.argsort(-np.abs(similarities[validMask]))]
        
        sortedIndices = np.array(sortedIndices[:n])
        topIndices = sortedIndices[sortedIndices != item]
        
        neighbourhood = [similarityMatrix[index, item] for index in topIndices]
        numerator = denominator = 0    
        for neighbour in topIndices:
            numerator += similarityMatrix[neighbour, item] * userItemMatrix[user, neighbour]
            denominator += abs(similarityMatrix[neighbour, item])
    
    weight = 0
    
    if isUser:
        if denominator == 0 or numerator == 0:
            return 0, 0.3
        
        neighbourhoodSize = len(neighbourhood)
        if neighbourhoodSize == 0:
            weight = 0
        elif neighbourhoodSize > 40:
            weight = 1     
        else:     
            weight = 0.5 + 0.5 * np.exp(-0.16 * neighbourhoodSize)
    else:
        if denominator == 0 or numerator == 0:
            return 0, 0.7
        
        neighbourhoodSize = len(neighbourhood)
        if neighbourhoodSize == 0:
            weight = 0
        elif neighbourhoodSize > 40:
            weight = 1     
        else:     
            weight = 0.2 + 0.8 * np.exp(-0.16 * neighbourhoodSize)
    
    pred = numerator / denominator
    pred = 0 if np.isnan(pred) else pred
    
    return pred, weight
    
# Load data
trainData = loadData("train_100k_withratings.csv")
testData = loadData("test_100k_withoutratings.csv")
nUsers, nItems = 944, 1683
trainArray = np.zeros((nUsers, nItems))
userItemMatrix = np.full((nUsers, nItems), np.nan)
for userId, itemId, rating, _ in trainData:
    userItemMatrix[userId-1, itemId-1] = rating

# Normalize user-item matrix
userMeans = np.nanmean(userItemMatrix, axis=1)
normalizedUserItemMatrix = userItemMatrix.copy()
nanMask = np.isnan(userItemMatrix)
for u in range(nUsers):
    normalizedUserItemMatrix[u, ~nanMask[u]] -= userMeans[u]

userSimMatrix = cosineSimilarityUser(userItemMatrix)
itemSimMatrix = cosineSimilarityItem(userItemMatrix)

# Predict ratings 
predictedRatings = {}
for userId, itemId, timestamp in testData:
    userPred, userQuality = prediction(userSimMatrix, normalizedUserItemMatrix, userId-1, itemId-1, True) + userMeans[userId-1]
    itemPred, itemQuality = prediction(itemSimMatrix, normalizedUserItemMatrix, userId-1, itemId-1, False) + userMeans[userId-1]
    
    pred = round(((userPred * userQuality) + (itemPred * itemQuality)) / (userQuality + itemQuality))
    pred = 5 if pred > 5 else pred
    pred = 1 if pred < 1 else pred
    predictedRatings[(userId, itemId)] = (pred, timestamp)
    
# Write predictions to file
with open("output.csv", "w") as file:
    for (userId, itemId), (pred, timestamp) in predictedRatings.items():
        file.write(f"{userId},{itemId},{pred},{timestamp}\n")

print("Done")    