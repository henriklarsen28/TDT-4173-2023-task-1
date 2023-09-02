import random

import numpy as np
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, iterations=200):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit

        self.iterations = iterations
        self.learning_rate = 0.01
        self.regularization = 0.1
        self.b = 0

        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """

        number_of_samples, number_of_features = X.shape

        # Add extra column of ones for bias
        best_accuracy = 0

        self.w = np.random.randn(number_of_features+1) * 0.01

        # Add bias column
        X = np.c_[np.ones(number_of_samples), np.array(X)]


        for _ in range(1):
            # Initialize weights and bias

            #X = np.c_[np.ones(number_of_features), np.array(X)]
            #print("w: ",self.w.shape)
            # Gradient descent
            for i in range(self.iterations):
                linear_model = np.dot(X, self.w)
                y_pred = sigmoid(linear_model)

                # Calculate gradients for logic regression
                dw = 2 * np.sum(np.matmul(X.T, (y_pred - y)))
                db = 2 * np.sum(y_pred - y)

                W = np.diag(y_pred * (np.repeat(1, len(y_pred)) - y_pred))
                inf_mat = np.matmul(np.transpose(X), np.matmul(W, X))
                inv_mat = np.linalg.inv(np.linalg.cholesky(inf_mat))

                print("dw: ",dw)
                print("db: ",db)


                #db = np.dot(X,(y_pred - y))

                # Update weights and bias
                self.w = self.w - self.learning_rate * np.dot(X, dw.T)
                self.b = self.b - self.learning_rate * db



    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        X = np.c_[np.ones(len(X)), np.array(X)]
        linear_model = np.matmul(X, self.w)
        y_pred = sigmoid(linear_model)
        #y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        #print("Test ",y_pred_class)
        return y_pred
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        