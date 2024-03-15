import numpy as np

def loadCSV(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return np.array([line.strip().split(',') for line in data])

def meanCenteredRatingMatrix(data): 
    users = data[:,0].astype(int).max(axis=0)
    items = data[:,1].astype(int).max(axis=0)
    matrix = np.zeros((users, items), dtype=np.float64)
    for user, item, rating, _ in data:
        matrix[int(user)-1, int(item)-1] = float(rating)    
    mean = np.mean(matrix, axis=1).reshape((matrix.shape[0], 1))
    ratings = matrix - mean
    return ratings

def similarityMatrix(ratingMatrix):    
    # Calculate the cosine similarity
    sim_matrix = np.dot(ratingMatrix.T, ratingMatrix)
    norms = np.linalg.norm(ratingMatrix.T, axis=1)
    sim_matrix /= norms[:, np.newaxis]
    sim_matrix /= norms[np.newaxis, :]

    # Handle division by zero
    sim_matrix[np.isnan(sim_matrix)] = 0
    np.fill_diagonal(sim_matrix, 1)  # Set self-similarity to 1
    return sim_matrix
       
def itemBasedPrediction(user, item, ratingMatrix, similarityMatrix, k):
    userId = user - 1
    itemId = item - 1
    
    itemSim, indices = kSimilar(itemId, similarityMatrix, k)
    
    userRatings = ratingMatrix[userId, indices]
    
    ratedItemsMask = userRatings > 0
    if ratedItemsMask.sum() == 0:
        return np.mean(ratingMatrix[ratingMatrix>0])
    
    weightedSum = np.dot(userRatings[ratedItemsMask], itemSim[ratedItemsMask])
    sumWeights = np.sum(itemSim[ratedItemsMask])
    
    prediction = weightedSum / sumWeights if sumWeights > 0 else np.mean(ratingMatrix[ratingMatrix > 0])   
     
    prediction = max(1, min(prediction, 5))
    
    return prediction

def userBasedPrediction(user, item, ratingMatrix, similarityMatrix, k):
    userId = user - 1
    itemId = item - 1
    
    userSim, indices = kSimilar(userId, similarityMatrix, k)
    itemRatings = ratingMatrix[indices, itemId]
    
    ratedUsersMask = itemRatings > 0
    if ratedUsersMask.sum() == 0:
        return np.mean(ratingMatrix[ratingMatrix>0])
    
    weightedSum = np.dot(itemRatings[ratedUsersMask], userSim[ratedUsersMask])
    sumWeights = np.sum(userSim[ratedUsersMask])
    
    prediction = weightedSum / sumWeights if sumWeights > 0 else np.mean(ratingMatrix[ratingMatrix > 0])
    
    prediction = max(1, min(prediction, 5))
    
    return prediction
    

def kSimilar(item, similarityMatrix, k=5):
    similar = []
    indices = []
    
    similar = np.copy(similarityMatrix[item])
    similar[item] = -np.inf
    
    indices = np.argsort(similar)[-k:]
    
    similar = similar[indices]
    
    return similar, indices
 
def calcWeights(userId, itemId, ratingMatrix):
    userRatingCount = np.count_nonzero(ratingMatrix[userId-1,:])
    itemRatingCount = np.count_nonzero(ratingMatrix[:,itemId-1])
        
    userWeight = userRatingCount / (userRatingCount + itemRatingCount)
    itemWeight = itemRatingCount / (userRatingCount + itemRatingCount)
    
    return userWeight, itemWeight

if __name__ == "__main__":
    train = loadCSV('train_100k_withratings.csv')
    test = loadCSV('test_100k_withoutratings.csv')
    print("Files loaded")
    ratingMatrix = meanCenteredRatingMatrix(train)
    print("Rating matrix made")
    itemSimilarityMatrix = similarityMatrix(ratingMatrix)
    userSimilarityMatrix = similarityMatrix(ratingMatrix.T)
    print("Similarity Matrix Made")

    # Test on Training Data
    predictionsTrain = []
    k1 = 5
    k2 = 3
    for user, item, _, _ in train:
        userW, itemW = calcWeights(int(user), int(item), ratingMatrix)
        itemPrediction = itemBasedPrediction(int(user), int(item), ratingMatrix, itemSimilarityMatrix, k1)
        userPrediction = userBasedPrediction(int(user), int(item), ratingMatrix, userSimilarityMatrix, k2)
        predictedRating = round((userW * userPrediction + itemW * itemPrediction))
        predictionsTrain.append(predictedRating)
    
    real = train[:, 2].astype(float)
    pred = np.array(predictionsTrain)
    print(np.abs(pred-real).mean())
    
    # predictions = []
    # for user, item, timestamp in test:
    #     predictedRating = (itemBasedPrediction(int(user), int(item), ratingMatrix, itemSimilarityMatrix) + userBasedPrediction(int(user), int(item), ratingMatrix, userSimilarityMatrix)) / 2
    #     if np.isnan(predictedRating):
    #         predictedRating = round(np.mean(ratingMatrix[ratingMatrix>0]))
    #     else:
    #         predictedRating = round(predictedRating)
    #     predictions.append((user, item, predictedRating, timestamp))
    # print("Saving to output.csv")
    # np.savetxt('output.csv', predictions, delimiter=',', fmt='%s')
    