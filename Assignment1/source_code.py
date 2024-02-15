# Implement a Cosine Similarity Recommender System Algorithm to train and predict ratings for a small (100k) set of items 

import numpy as np

# Training Function for Cosine Similarity Recommender System Algorithm
def trainModel(trainingData = None) :
    model = ''
    return model
    
# Prediction generated from Model trained 
def predictRatings(model = None, data = None) :
    predictions = ''
    return predictions
    
# Loads data from CSV
def loadData(file = None) :
    with open(file) as data:
        lines = data.readlines
    return (np.asarray([point.strip('\n').split(',')] for point in data))

# Main
if __name__ == '__main__':
    train = loadData("train_100k_withratings.csv")
    test = loadData("test_100k_withoutratings.csv")
