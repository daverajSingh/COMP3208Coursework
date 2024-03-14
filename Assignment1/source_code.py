import numpy as np

def loadCSV(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return np.array([line.strip().split(',') for line in data])

def meanCenteredRatingMatrix(data): 
    users = data[:,0].astype(int).max(axis=0)
    items = data[:,1].astype(int).max(axis=0)
    matrix = np.zeros((users, items), dtype=np.float16)
    for user, item, rating, _ in data:
        matrix[int(user)-1, int(item)-1] = float(rating)    
    mean = np.mean(matrix, axis=1).reshape((matrix.shape[0], 1))
    indices = np.where(matrix == 0)
    ratings = matrix - mean
    ratings[indices] = 0
    return ratings

def timeStampMatrix(data):
    users = np.unique(data[:,0]).max(axis=0)
    items = np.unique(data[:,1]).max(axis=0)
    matrix = np.zeros((users, items), dtype=int)
    for user, item, _, timestamp in data:
        matrix[int(user)-1, int(item)-1] = int(timestamp)
    return matrix 
    
def adjustedCosineSimilarity(ratingMatrix):
    ratingMatrix = np.nan_to_num(ratingMatrix)
    
    meanRatings = np.where(ratingMatrix != 0, ratingMatrix, np.nan).mean(axis=0, keepdims=True)
    adjustedRatings = ratingMatrix - np.where(ratingMatrix != 0, meanRatings, 0)
    
    simMatrix = adjustedRatings.T @ adjustedRatings
    
    norm = np.sqrt(np.diag(simMatrix))
    
    similarityMatrix = simMatrix / (norm[:, None] * norm)
    
    np.fill_diagonal(similarityMatrix, np.NaN)
        
    return similarityMatrix  
 
def itemBasedPrediction(user, item, ratingMatrix, itemSimilarity):
    itemNeighbours = getNeighbours(itemSimilarity)
    if(len(itemNeighbours) == 0):
        return predictor([],[])
    else:
        neighbourSimilarity = itemSimilarity[itemNeighbours]
        neighbourRatings = ratingMatrix[user, itemNeighbours]
        return predictor(neighbourRatings, neighbourSimilarity)
        
def predictor(neighbourRatings, neighbourSimilarity):
    if len(neighbourRatings) == 0:
        pred = np.NaN
    else:
        neighbourSim = np.ones(neighbourSimilarity.size) * neighbourSimilarity
        pred = (neighbourRatings @ neighbourSim) / neighbourSim.sum()
    
    if np.isnan(pred):
        pred = np.NaN
    else:
        pred = np.round(pred, 1)
    
    return pred

def getNeighbours(itemSimilarity):
    valid = itemSimilarity[~np.isnan(itemSimilarity)]
    sortedIndices = np.argsort(valid)[::-1]
    return sortedIndices
    
def itemPredictionMatrix(data, ratingMatrix, similarityMatrix):
    predictions = []
    for n, (user, item, _) in enumerate(data):
        user = int(user)-1
        item = int(item)-1
        pred = itemBasedPrediction(user, item, ratingMatrix, similarityMatrix[item])
        if np.isnan(pred):
            pred = np.mean(ratingMatrix[user])
            if np.isnan(pred):
                pred = 2.5
        predictions.append(pred)
    predictions = np.asarray(predictions, dtype=np.string_)
    predictions == np.expand_dims(predictions, axis=1)
    return np.append(np.append(data[:,:2], predictions, axis=1),  np.expand_dims(data[:, 2], axis=1), axis=1)

if __name__ == "__main__":
    train = loadCSV('train_100k_withratings.csv')
    test = loadCSV('test_100k_withoutratings.csv')
    ratingMatrix = meanCenteredRatingMatrix(train)
    similarityMatrix = adjustedCosineSimilarity(ratingMatrix)
    predictions = itemPredictionMatrix(test, ratingMatrix, similarityMatrix)
    np.savetxt('predictions.csv', predictions, delimiter=',', fmt='%s')
    