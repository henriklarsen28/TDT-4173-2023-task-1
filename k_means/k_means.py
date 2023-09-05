import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, number_of_clusters = 0, number_of_iterations=20, tolerance= 0.002):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.number_of_clusters = number_of_clusters
        self.number_of_iterations = number_of_iterations
        self.centroids = None
        self.tolerance = tolerance
        pass

    def kmeans_plusplus_inizialization(self, X, k):
        centroids = []

        # Choose random centroid at first
        first_centroid = np.random.choice(X.shape[0])

        centroids.append(X[first_centroid])

        for _ in range(1, k):
            distances = np.array([np.min(np.linalg.norm(centroid - centroids, axis=1)) ** 2 for centroid in X])
            total_distance = np.sum(distances)
            probabilities = distances / total_distance

            next_centroid = np.random.choice(X.shape[0], p=probabilities)
            centroids.append(X[next_centroid])
        return centroids

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """

        # https://neptune.ai/blog/k-means-clustering

        # Initialize best centroids and silhouette variables
        best_silhouette = 0
        best_distorion = 99999
        best_centroids = None

        best_number_of_clusters = 2

        start = self.number_of_clusters
        end = self.number_of_clusters+1
        if self.number_of_clusters == 0:
            start = self.number_of_clusters
            end = 12
        for k in range(start, end):
            # Iterate through the code to improve position of centroids
            for _ in range(self.number_of_iterations):

                # Initialize centroids using kmeans++, comment out if you want to use random centroids
                #self.centroids = X_np[np.random.choice(X_np.shape[0], self.number_of_clusters, replace=False)]
                self.centroids = KMeans.kmeans_plusplus_inizialization(self, X,k)
                last_centroid = np.random.randn(k, 2)

                # Iterate through the code to improve position of centroids
                for j in range(5):
                    cluster_nodes = [np.empty((0, X.shape[1])) for _ in range(k)]

                    # Loop through coordinates to find the and assign them to different nodes
                    for i in range(len(X)):
                        # Find the closest centroid and insert node into that list
                        closest_centroid = self.closestCentroid(X[i])
                        cluster_nodes[closest_centroid] = np.vstack((cluster_nodes[closest_centroid],X[i]))

                    # Calculate mean of each cluster
                    for j in range(k):
                        #Find the new centroid
                        new_centroid = np.mean(cluster_nodes[j], axis=0)
                        self.centroids[j] = new_centroid

                    # Finds the best silhouette score and stores the centroids
                    prediction = KMeans.predict(self, X)

                    distortion = euclidean_distortion(X, prediction)
                    if distortion < best_distorion:
                        silhouette = euclidean_silhouette(X, prediction)
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_distorion = distortion
                            best_centroids = self.centroids
                            best_number_of_clusters = k

                    # Check if the centroids have hit the tolerance
                    tol = self.calculateTolerance(last_centroid, self.centroids)
                    last_centroid = self.centroids
                    if tol <= self.tolerance:
                        self.centroids = best_centroids
                        break
        # Updates the number of clusters
        if self.number_of_clusters == 0:
                self.number_of_clusters = best_number_of_clusters

        self.centroids = best_centroids
    def calculateTolerance(self, last_centroids, centroids):
        last_centroids = np.asarray(last_centroids)
        centroids = np.asarray(centroids)
        return np.linalg.norm(centroids - last_centroids)


    def closestCentroid(self, node):
        distances = [euclidean_distance(node, centroid) for centroid in self.centroids]
        return np.argmin(distances)

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        prediction = np.empty((0,1))
        # Classify which node belongs to which centroid
        for i in range(len(X)):   # loop through each point

            # Find the closest centroid and insert centroid index into prediction array
            prediction = np.append(prediction, self.closestCentroid(X[i]))

        #print(prediction)
        return prediction.astype(int)

    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return np.asarray(self.centroids)

    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """

    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  