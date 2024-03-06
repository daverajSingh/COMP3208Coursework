# Implement a Cosine Similarity Recommender System Algorithm to train and predict ratings for a small (100k) set of items 
import numpy as np

# Loads data from CSV
def loadData(file = None) :
    data = []
    with open(file) as f:
        for line in f:
            i = line.strip().split(',')
            if len(i) == 4:
                userid = int(i[0])
                itemid = int(i[1])
                rating = float(i[2])
                timestamp = int(i[3])
                data.append([userid, itemid, rating, timestamp])
            else:
                userid = int(i[0])
                itemid = int(i[1])
                timestamp = int(i[2])
                data.append([userid, itemid, timestamp])
    return np.array(data)        
        
def createUserItemMatrix(data, isTraining=True):
    users = data[:, 0]
    items = data[:, 1]
    
    numberOfUsers = len(set(users))
    numberOfItems = len(set(items))
    
    userItemMatrix = np.zeros((numberOfUsers, numberOfItems))
    if isTraining:
        for line in data:
            userItemMatrix[line[0]-1, line[1]-1] = line[2]
    return userItemMatrix

def cosineSimilarity(UImatrix):
    dotProduct = np.dot(UImatrix.T, UImatrix)
    magnitude = np.sqrt(np.diag(dotProduct))
    similarityMatrix = dotProduct / np.outer(magnitude, magnitude)
    similarityMatrix[np.isnan(similarityMatrix)]=0
    return similarityMatrix

def recommenderSystem(itemSim, test, UIMatrix):
    predictedRatings = []
    
    for user, item, timestamp in test:
        userIdx = user -1
        itemIdx = item -1
        
        itemSimilarity = itemSim[itemIdx]
        
        userRating = UIMatrix[userIdx]
        
        nonZero = userRating > 0
        itemSimilarity = itemSimilarity * nonZero
        
        if np.sum(itemSimilarity) > 0:
            predictedRating = np.dot(itemSimilarity, userRating)/np.sum(itemSimilarity)
        else:
            predictedRating = np.mean(userRating[userRating > 0])
        
        predictedRatings.append((user, item, predictedRating, timestamp))
                
    return np.array(predictedRatings)

# Main
if __name__ == '__main__':
    train = loadData("train_100k_withratings.csv")
    test = loadData("test_100k_withoutratings.csv")
    
    trainUIMatrix = createUserItemMatrix(train)
    testUIMatrix = createUserItemMatrix(test, isTraining=False)
    cosineSim = cosineSimilarity(trainUIMatrix)
    predictions = recommenderSystem(test, cosineSim, trainUIMatrix)
    
    print(predictions)