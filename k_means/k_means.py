import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, number_of_clusters):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.number_of_clusters = number_of_clusters
        self.max_iters= 100
        pass

    def kmeans_plusplus_inizialization(self, X, num_clusters):
        centroids = []

        # Convert X to np
        X_np = np.asarray(X)
        # Choose random centroid at first
        first_centroid = np.random.choice(X_np.shape[0])

        centroids.append(X_np[first_centroid])

        for _ in range(1, num_clusters):
            distances = np.array([np.min(np.linalg.norm(centroid - centroids, axis=1)) ** 2 for centroid in X_np])
            total_distance = np.sum(distances)
            probabilities = distances / total_distance

            next_centroid = np.random.choice(X_np.shape[0], p=probabilities)
            centroids.append(X_np[next_centroid])
        #print("Centroids ", centroids)
        return centroids

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # https://neptune.ai/blog/k-means-clustering
        # Find middle of two clusters centroid
        # Use eucledean distance to find the which category they should be in
        # Select two random spots and define them as centroids
        # self.c0 = np.array([0, 0])
        # self.c1 = np.array([1, 1])
        # self.centroids = np.array([self.c0, self.c1])
        #print(X.shape)
        # self.centroids = X[np.random.choice(X.shape[0], self.number_of_clusters, replace=False)]
        X_np = np.asarray(X)
        candidate_centroids = np.empty((0, 2))
        best_silhouette = 0
        best_centroids = None
        for i in range(5):


            # self.centroids = X_np[np.random.choice(X_np.shape[0], self.number_of_clusters, replace=False)]
            self.centroids = KMeans.kmeans_plusplus_inizialization(self, X_np, self.number_of_clusters)
            #self.centroids = np.array([[40,1],[45,7]])

            #print(self.centroids)

            #print(self.centroids)

            # Iterate through the code to improve position of centroids
            for i in range(4):
                c0_nodes = np.empty((0, 3))
                c1_nodes = np.empty((0, 3))
                c0_mean = 0
                c1_mean = 0
                # Loop through coordinates to find the and assign them to different nodes
                for i in range(len(X)):
                    # Make x and y to vectors


                    #print(X_np2)
                    x0 = np.array([X[i][0], X[i][1]])
                    y0 = self.centroids[0]

                    x1 = np.array([X[i][0], X[i][1]])
                    y1 = self.centroids[1]

                    # Find min distance for point
                    c0_distance = euclidean_distance(y0, x0)
                    c1_distance = euclidean_distance(y1, x1)

                    try:
                        c0_mean = np.median(c0_nodes[:, 2])
                        c1_mean = np.median(c1_nodes[:, 2])
                    except:
                        c0_mean = 0
                        c1_mean = 0

                    # Calculate the sum of squares and append to list
                    c0_sum = np.sum(c0_nodes[:, 2]-c0_mean)
                    c1_sum = np.sum(c1_nodes[:, 2]-c1_mean)
                    node = np.min([c0_sum, c1_sum])

                    if c0_distance < c1_distance:
                        row = [X[i][0], X[i][1], c0_distance]
                        c0_nodes = np.vstack((c0_nodes, row))
                    elif c0_distance > c1_distance:
                        row = [X[i][0], X[i][1], c1_distance]
                        c1_nodes = np.vstack((c1_nodes, row))


                # Find mean coordinates of every node in c0 and c1 list
                # TODO: Sort after distance and remove the top

                # TODO: Print after every iteration

                c0_x = np.mean(c0_nodes[:, 0])
                c0_y = np.mean(c0_nodes[:, 1])

                c1_x = np.mean(c1_nodes[:, 0])
                c1_y = np.mean(c1_nodes[:, 1])
                #print(minDistance0)
                #print(np.median(c0_nodes[:, 2]))
                #Remove the nodes over the mean distance + 10%
                #c0_nodes = c0_nodes[c0_nodes[:, 2] < np.mean(c0_nodes[:, 2])*1.4]
                #c1_nodes = c1_nodes[c1_nodes[:, 2] < np.mean(c1_nodes[:, 2])*1.4]
                #print(c0_nodes)



                # Update centroid coordinates
                c0 = [c0_x, c0_y]
                c1 = [c1_x, c1_y]

                #_, ax = plt.subplots(figsize=(5, 5), dpi=100)

                self.centroids = np.array([c0, c1])

                silhouette = euclidean_silhouette(X, KMeans.predict(self, X))
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_centroids = self.centroids



                # Plot the centroids
                #ax.scatter(c0[0], c0[1], c='r', marker='x', s=100)
                #ax.scatter(c1[0], c1[1], c='b', marker='x', s=100)
                #plt.show()
        self.centroids = best_centroids


    
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
            # Make x and y to vectors
            x0 = np.array([X[i][0], X[i][1]])
            y0 = self.centroids[0]

            x1 = np.array([X[i][0], X[i][1]])
            y1 = self.centroids[1]

            # Find min distance for point
            c0_distance = euclidean_distance(y0, x0)
            c1_distance = euclidean_distance(y1, x1)

            if c0_distance < c1_distance:
                prediction = np.append(prediction,0)
            else:
                prediction = np.append(prediction,1)

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
        return self.centroids

    
    
    
    
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
  