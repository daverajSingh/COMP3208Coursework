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
def createUserItemMatrix(data, isTraining=True):
    users = np.unique(data[:,0])
    items = np.unique(data[:,1])
    UIMatrix = np.zeros((len(users), len(items)))
    for i in range(len(data)):
        userIndex = np.where(users == data[i,0])
        itemIndex = np.where(items == data[i,1])
        if isTraining:
            UIMatrix[userIndex, itemIndex] = data[i,2]
        else:
            UIMatrix[userIndex, itemIndex] = 0
    return UIMatrix

# Item Based adjusted Cosine Similarity
def cosineSimilarity(UImatrix):
    similarityMatrix = np.zeros((UImatrix.shape[1], UImatrix.shape[1]))
    for i in range(UImatrix.shape[1]):
        for j in range(UImatrix.shape[1]):
            if i == j:
                similarityMatrix[i,j] = 1
            else:
                similarityMatrix[i,j] = np.dot(UImatrix[:,i], UImatrix[:,j])/(np.linalg.norm(UImatrix[:,i])*np.linalg.norm(UImatrix[:,j]))
    return similarityMatrix
    
# Predicts the ratings for the test data
def predictRating(trainUI, testUI, similarityMatrix):

# Main
if __name__ == '__main__':
    train = loadData("train_100k_withratings.csv")
    test = loadData("test_100k_withoutratings.csv")
    
    trainUIMatrix = createUserItemMatrix(train)
    testUIMatrix = createUserItemMatrix(test, False)
    similarityMatrix = cosineSimilarity(trainUIMatrix)

    
    
    print()
    
