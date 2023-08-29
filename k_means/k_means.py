import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.c0 = [0, 0]
        self.c1 = [1, 1]

        pass
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        def eucledeanDistance(centroid, x_coordinate, y_coordinate):
            x = x_coordinate - centroid[0]
            y = y_coordinate - centroid[1]

            distance = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
            return distance
        #raise NotImplemented()
        # https://neptune.ai/blog/k-means-clustering
        # Find middle of two clusters centroid
        # Use eucledean distance to find the which category they should be in
        # Select two random spots and define them as centroids

        plt.plot(self.c0[0],self.c0[1],marker="*", markersize=12)

        #Iterate through the code to improve position of centroids
        for i in range(4):
            c0_x_nodes = []
            c0_y_nodes = []
            c1_x_nodes = []
            c1_y_nodes = []

            print(len(X))
            # Loop through coordinates to find the and assign them to different nodes
            for i in range(len(X)-1):
                # Find min distance for point

                min = sys.maxsize
                c0_distance = eucledeanDistance(self.c0,X["x0"][i],X["x1"][i])
                c1_distance = eucledeanDistance(self.c1, X["x0"][i], X["x1"][i])
                if c0_distance < c1_distance:
                    c0_x_nodes.append(X["x0"][i])
                    c0_y_nodes.append(X["x1"][i])
                else:
                    c1_x_nodes.append(X["x0"][i])
                    c1_y_nodes.append(X["x1"][i])

                print("C0 %.4f" %c0_distance)
                print("C1 %.4f" %c1_distance)

            # Find mean coordinates of every node in c0 and c1 list
            c0_x = np.mean(c0_x_nodes)
            c0_y = np.mean(c0_y_nodes)

            c1_x = np.mean(c1_x_nodes)
            c1_y = np.mean(c1_y_nodes)


            # Update centroid coordinates
            self.c0 = [c0_x,c0_y]
            self.c1 = [c1_x,c1_y]
            plt.plot(self.c0[0], self.c0[1], marker="*", markersize=12)






    
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
        # TODO: Implement 
        raise NotImplemented()
    
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
        pass
    
    
    
    
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
        distortion += ((Xc - mu) ** 2).sum(axis=1)
        
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
  