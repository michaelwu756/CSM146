#!/usr/bin/python
"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])

    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """

    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.savefig(title.replace(' ','')+'.pdf')


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.

    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    initial_points = []
    newlist = list(points)
    for i in range(0, k):
        choice = np.random.choice(len(newlist))
        initial_points.append(newlist.pop(choice))
    return initial_points


def cheat_init(points) :
    """
    Initialize clusters by cheating!

    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.

    Parameters
    --------------------
        points         -- list of Points, dataset

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    initial_points = []
    clusterMap = {}
    for p in points:
        if p.label in clusterMap.keys():
            clusterMap[p.label].append(p)
        else:
            clusterMap[p.label]=[p]
    for cluster in clusterMap.values():
        initial_points.append(Cluster(cluster).medoid())
    return initial_points


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.

    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable:
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm

    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    return kAverages(points, k, ClusterSet.centroids, init, plot)

def kAverages(points, k, average, init, plot):
    k_clusters = ClusterSet()
    if init=='random':
        initSet = random_init(points, k)
    elif init=='cheat':
        initSet = cheat_init(points)
    clusterMap = {}
    for c in initSet:
        clusterMap[c]=[]
    for p in points:
        closest = None
        distance = float('inf')
        for c in initSet:
            if p.distance(c)<distance:
                closest=c
                distance=p.distance(c)
        clusterMap[closest].append(p)
    for cluster in clusterMap.values():
        k_clusters.add(Cluster(cluster))
    iteration = 0
    while True:
        if plot:
            if average==ClusterSet.centroids:
                plot_clusters(k_clusters, "kMeans "+init+" iteration "+str(iteration), average)
            elif average==ClusterSet.medoids:
                plot_clusters(k_clusters, "kMedoids "+init+" iteration "+str(iteration), average)
        iteration+=1
        newClusters = ClusterSet()
        clusterMap = {}
        updatedCenters = average(k_clusters)
        for c in updatedCenters:
            clusterMap[c]=[]
        for p in points:
            closest = None
            distance = float('inf')
            for c in updatedCenters:
                if p.distance(c)<distance:
                    closest=c
                    distance=p.distance(c)
            clusterMap[closest].append(p)
        for cluster in clusterMap.values():
            newClusters.add(Cluster(cluster))
        if k_clusters.equivalent(newClusters):
            break
        else:
            k_clusters=newClusters
    return k_clusters

def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    return kAverages(points, k, ClusterSet.medoids, init, plot)


######################################################################
# main
######################################################################

def main() :

    np.random.seed(1234)
    points = generate_points_2d(20)
    kMeans(points, 3, init='random', plot=True)
    kMedoids(points, 3, init='random', plot=True)
    kMeans(points, 3, init='cheat', plot=True)
    kMedoids(points, 3, init='cheat', plot=True)

if __name__ == "__main__" :
    main()
