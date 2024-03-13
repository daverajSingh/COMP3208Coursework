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

#Create a User Item Matrix from the data
def createUserItemMatrix(data, users, items, isTraining=True):
    UIMatrix = np.zeros((users, items))
    for i in range(len(data)):
        userIndex = np.where(users == data[i,0])
        itemIndex = np.where(items == data[i,1])
        if isTraining:
            UIMatrix[userIndex, itemIndex] = data[i,2]
        else:
            UIMatrix[userIndex, itemIndex] = 1
    return UIMatrix

# Item Based Adjusted Cosine Similarity
def cosineSimilarity(UImatrix):
    sim = UImatrix.T @ UImatrix
    norm = np.sqrt(np.diag(sim))
    return (sim/norm[:, np.newaxis])/norm


# Predicts the ratings for the test data 
def predictRating(trainUI, testUI, similarityMatrix):
    userMeans = np.true_divide(trainUI.sum(1), (trainUI != 0).sum(1))
    normalisedTrainUI = np.where(trainUI != 0, trainUI - userMeans[:, np.newaxis], 0)
    
    predictNumerator = similarityMatrix @ normalisedTrainUI.T
    predictDenominator = np.abs(similarityMatrix).sum(1).reshape(-1,1)
    
    predictDenominator[predictDenominator == 0] = 1
    
    predictionMatrix = predictNumerator/predictDenominator
    
    predictions = (predictionMatrix.T + userMeans[:, np.newaxis]) * (testUI == 1)
    
    return predictions
    pass

def getTimestamps(data):
    numUsers = len(np.unique(data[:,0]))
    numItems = len(np.unique(data[:,1]))
    timestampArr = np.zeros((numUsers, numItems))
    for i in range(len(data)):
        userIndex = np.where(np.unique(data[:,0]) == data[i,0])
        itemIndex = np.where(np.unique(data[:,1]) == data[i,1])
        timestampArr[userIndex, itemIndex] = data[i,2]
    return 

# Main
if __name__ == '__main__':
    train = loadData("train_100k_withratings.csv")
    test = loadData("test_100k_withoutratings.csv")
    timestampArr = getTimestamps(test)

    numberUsers = np.maximum(len(np.unique(train[:,0])), len(np.unique(test[:,0])))
    numberItems = np.maximum(len(np.unique(train[:,1])), len(np.unique(test[:,1])))
    
    trainUIMatrix = createUserItemMatrix(train, numberUsers, numberItems)
    testUIMatrix = createUserItemMatrix(test, numberUsers, numberItems, False)
    similarityMatrix = cosineSimilarity(trainUIMatrix)
    predictedRatings = predictRating(trainUIMatrix, testUIMatrix, similarityMatrix)
    
    rows_to_write = []
    user_ids = test[:, 0]
    item_ids = test[:, 1]
    
    for (user_index, item_index), rating in np.ndenumerate(predictedRatings):
        if rating > 0:
            userid = user_ids[user_index]
            itemid = item_ids[item_index]
            timestamp = timestampArr[user_index][item_index] 
            rows_to_write.append([userid, itemid, rating, timestamp])
    
    
    with open("output.csv", "w") as f:
        for row in rows_to_write:
            f.write(','.join(map(str, row)) + '\n')
        f.close()
