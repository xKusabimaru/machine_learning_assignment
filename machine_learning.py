import numpy as np
import pandas as pd

def preprocess_classification_dataset():
    #train
    train_df = pd.read_csv("train.csv")

    train_feat_df = train_df.iloc[:,:-1]
    train_output = train_df[['output']]

    X_train = train_feat_df.values
    y_train = train_output.values

    #val
    val_df = pd.read_csv("val.csv")

    val_feat_df = val_df.iloc[:,:-1]
    val_output = val_df[['output']]

    X_val = val_feat_df.values
    y_val = val_output.values

    #test
    test_df = pd.read_csv("test.csv")

    test_feat_df = test_df.iloc[:,:-1]
    test_output = test_df[['output']]

    X_test = test_feat_df.values
    y_test = test_output.values

    return X_train, y_train, X_val, y_val, X_test, y_test

def knn_classification(X_train, y_train, x_new, k=5):
    output = []
    
    for X_train_row in X_train:
        sums = []

        for i in range(len(X_train_row)):
            sums.append(np.abs(X_train_row[i]-x_new[i])**2)
        
        output.append((sum(sums))**0.5)
    
    indexes = np.argsort(output)[:k]
    y_list = []

    for i in range(len(indexes)):
        y_list.append(y_train[indexes[i]])

    values, counts = np.unique(y_list, return_counts=True)

    if len(counts) > 1:
        if counts[0] > counts[1]:
            return values[0]
        return values[1]
    return values[0]

def sigmoid(z):
    return 1 / (1 + (np.e ** (-z)))

def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    X_train_ones = np.hstack((np.ones((len(X_train), 1)), X_train))

    np.random.seed(random_seed)
    weights = np.random.normal(loc=0.0, scale=1.0, size=(len(X_train_ones[0]), 1))
    
    for i in range(max_iters):
        weights = weights - ((alpha * X_train_ones.T).dot(sigmoid(X_train_ones @ weights) - y_train))
    
    return weights

def logistic_regression_prediction(X, weights, threshold=0.5):
    X_ones = np.hstack((np.ones((len(X), 1)), X))
    y_preds = np.zeros(shape=(len(X), 1))
    
    for i in range(len(y_preds)):
        if sigmoid(X_ones @ weights)[i] >= threshold:
            y_preds[i][0] = 1

    return y_preds

def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    return None

