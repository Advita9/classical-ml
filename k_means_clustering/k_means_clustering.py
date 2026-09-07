# k means clustering --> unsupervised learning model that helps us find clusters in unlabelled data
# unlike k nearest neighbors, no labels are present
# there are strategies to find the most optimal k
# we find clusters --> initialise cluster centroids (randomly at first within data range)
# we classify new points to the nearest centroids to them --> recalculate new centroids after new assignments 
# we do this uptil max iteration number / upto a point of no change

import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k = 3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis = 1))
    
    def fit(self, X, max_iterations = 200):
        # for every dimension of the coordinates generated we keep it within the min and max of the respective dimension of X
        # initialise random centroids
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size = (self.k, X.shape[1]))

        for iteration in range(max_iterations):
            # calculate 
            cluster_labels = []
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                # returns the index of smallest value(we get centroid with min distance to that point)
                cluster_num = np.argmin(distances)
                # assign each data point to its closest centroid
                cluster_labels.append(cluster_num)
            cluster_labels = np.array(cluster_labels)

            # reposition centroids based on generated labels
            cluster_centers = []
            for cluster_num in range(self.k):
                # list of all X points in each cluster
                cluster_points = X[cluster_labels == cluster_num]

                if len(cluster_points) == 0:
                    cluster_centers.append(self.centroids[cluster_num])
                else:
                    cluster_centers.append(np.mean(cluster_points, axis = 0))
            cluster_centers = np.array(cluster_centers)

            # checking centroid movement
            centroid_movement = np.max(np.abs(self.centroids - cluster_centers))
            if centroid_movement < 0.0001:
                self.centroids = cluster_centers
                print(
                    f"K={self.k} converged after "
                    f"{iteration + 1} iterations"
                )

                break
            self.centroids = cluster_centers
        return cluster_labels
        

# randomised un-clustered data 
random_points = np.random.randint(0, 100, (100, 2))
kmeans = KMeansClustering(k = 5)
labels = kmeans.fit(random_points)
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
            marker="*", s = 200)
plt.savefig("clusters_random_unclustered_datapoints"
".png", dpi=300)
plt.show() 

# trying with data that contains actual clusters
np.random.seed(42)

cluster_1 = np.random.normal(
    loc=[20, 20],
    scale=5,
    size=(50, 2)
)

cluster_2 = np.random.normal(
    loc=[70, 25],
    scale=5,
    size=(50, 2)
)

cluster_3 = np.random.normal(
    loc=[50, 75],
    scale=5,
    size=(50, 2)
)

X = np.vstack([
    cluster_1,
    cluster_2,
    cluster_3
])

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    marker="*",
    s=300
)

plt.savefig("clusters_previously_clustered_data.png", dpi=300)
plt.show()

# code to compare different k value clusters
# for k in [2, 3, 4, 5, 6]:
#     kmeans = KMeansClustering(k=k)
#     labels = kmeans.fit(X)

#     print(
#         f"k={k}, centroids={kmeans.centroids}"
#     )
#     plt.scatter(
#         X[:, 0],
#         X[:, 1],
#         c=labels
#     )

#     plt.scatter(
#         kmeans.centroids[:, 0],
#         kmeans.centroids[:, 1],
#         marker="*",
#         s=300
#     )

#     plt.show()

# Results:
# k=2, centroids=[[44.42217873 22.67011162]
#  [50.64473376 75.00422877]]
# k=3, centroids=[[69.52273561 25.70031026]
#  [19.32162185 19.63991298]
#  [50.64473376 75.00422877]]
# k=4, centroids=[[69.52273561 25.70031026]
#  [56.0279329  72.75962963]
#  [19.32162185 19.63991298]
#  [47.34535364 76.37995082]]
# k=5, centroids=[[48.37855355 75.5725677 ]
#  [19.32162185 19.63991298]
#  [69.52273561 25.70031026]
#  [52.57523843 94.26365745]
#  [58.29785589 71.29001895]]
# k=6, centroids=[[69.52273561 25.70031026]
#  [19.32162185 19.63991298]
#  [46.46240088 74.84712529]
#  [55.55442888 75.18865459]
#  [13.51642167 51.69308772]
#  [11.07599149 56.33331273]]


# K=5 converged after 6 iterations (un-clustered data)
# K=3 converged after 2 iterations (clustered data with k = 3)