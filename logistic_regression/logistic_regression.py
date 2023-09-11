import random

import numpy as np
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, iterations=200, input_dimension=2, tolerance=0.0001):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit

        self.iterations = iterations
        self.learning_rate = 0.01
        self.regularization = 0.1
        self.input_dimension = input_dimension
        self.tolerance = tolerance

    def preprocess(self, X):
        X = pd.DataFrame(X) # Convert to pandas dataframe since it came as numpy array
        number_of_samples, number_of_features = X.shape

        for i in range(2, self.input_dimension + 1):
            for j in range(number_of_features):
                column = X.columns[j]
                X[str(column) + "^" + str(i)] = X[column] ** i
        X = np.c_[np.ones(number_of_samples), X]
        return X

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """


        # Add bias column
        X = self.preprocess(X)
        number_of_samples, number_of_features = X.shape

        # Add extra column of ones for bias
        self.w = np.random.randn(number_of_features) * 0.01
        # Gradient descent
        for i in range(self.iterations):
            linear_model = np.dot(X, self.w)
            y_pred = sigmoid(linear_model)

            # Gradient
            dw = 2 * np.dot(X.T, y_pred - y)
            
            # Old weights
            oldW = self.w
            # Update weights using gradient descent with gradient descent
            self.w = self.w - self.learning_rate * dw

            # Check if weights are not changing anymore
            if np.allclose(oldW, self.w, rtol=self.tolerance):
                break



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
        X = self.preprocess(X)
        linear_model = np.dot(X, self.w)
        y_pred = sigmoid(linear_model)

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

        