# K nearest neighbors : used to determine the class of a new point
# uses Euclidean distance
# lazy learner / instance-based learner --> unline other classifiers, it does not learn model parameters
# during training, it just stores training data, only performs distance calculation when a prediction is requested
# the value of k determines ias-variance tradeoff [low k --> low bias, high variance and vice versa]
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 2D
points = {'blue': [[2,4], [1,3], [2,3], [3,2], [2,1]],
          'orange': [[5,6], [4,5], [4,6], [6,6], [5,4]]}

new_point = [3,3]

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

class KNearestNeighbors:
    def __init__(self, k = 9):
        self.k = k
        self.points = None
    def fit(self, points):
        self.points = points
    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category, point])
        # if there is a distance tie, class labels should not determine the winning neighbor
        nearest = sorted(distances, key = lambda x: x[0])[:self.k]
        categories = [category for distance, category, point in nearest]
        result = Counter(categories).most_common(1)[0][0]
        return result, nearest

clf = KNearestNeighbors(k = 3)
clf.fit(points)
new_class, nearest = clf.predict(new_point)

# printing the selected k nearest neighbors
print("Prediction:", new_class)
print("Nearest neighbors:")

for distance, category, point in nearest:
    print(
        f"point={point}, "
        f"class={category}, "
        f"distance={distance:.3f}"
    )

# visualise KNN distances

fig, ax = plt.subplots(figsize=(8, 8))

ax.grid(False)
ax.set_facecolor('black')
fig.set_facecolor('#121212')

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')


# Plot all points

for point in points['blue']:
    ax.scatter(
        point[0],
        point[1],
        color='#104DCA',
        s=60
    )

for point in points['orange']:
    ax.scatter(
        point[0],
        point[1],
        color='#EF6C35',
        s=60
    )


# Plot new point

color = (
    '#EF6C35'
    if new_class == 'orange'
    else '#104DCA'
)

ax.scatter(
    new_point[0],
    new_point[1],
    color=color,
    marker='*',
    s=250,
    zorder=100
)

# Draw distances to every point
for category in points:

    line_color = (
        '#104DCA'
        if category == 'blue'
        else '#EF6C35'
    )

    for point in points[category]:

        ax.plot(
            [new_point[0], point[0]],
            [new_point[1], point[1]],
            color=line_color,
            linestyle='--',
            linewidth=0.7,
            alpha=0.25
        )


# Highlight the K nearest neighbors
for distance, category, point in nearest:

    ax.scatter(
        point[0],
        point[1],
        facecolors='none',
        edgecolors='yellow',
        s=250,
        linewidths=2.5,
        zorder=50
    )

    ax.plot(
        [new_point[0], point[0]],
        [new_point[1], point[1]],
        color='yellow',
        linestyle='-',
        linewidth=2.5,
        zorder=40
    )

new_class = clf.predict(new_point)
color = '#EF6C35' if new_class == 'orange' else '#104DCA'
ax.scatter(new_point[0], new_point[1], color=color, marker='*', s=200, zorder=100)

for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#104DCA', linestyle='--', linewidth=1)

for point in points['orange']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#EF6C35', linestyle='--', linewidth=1)

plt.savefig(
    "knn_2d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




# 3 D
points = {'blue': [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
          'orange': [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6], [10, 10, 4]]}

new_point = [3, 3, 4]

clf = KNearestNeighbors(k=3)
clf.fit(points)

new_class, nearest = clf.predict(new_point)

print("\n3D KNN")
print("Prediction:", new_class)

# printing the selected k nearest neighbors
print("Nearest neighbors:")

for distance, category, point in nearest:
    print(
        f"point={point}, "
        f"class={category}, "
        f"distance={distance:.3f}"
    )

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.grid(True, color='#323232')

ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Draw all distances faintly

for category in points:

    line_color = (
        '#104DCA'
        if category == 'blue'
        else '#EF6C35'
    )

    for point in points[category]:

        ax.plot(
            [new_point[0], point[0]],
            [new_point[1], point[1]],
            [new_point[2], point[2]],
            color=line_color,
            linestyle='--',
            linewidth=0.7,
            alpha=0.25
        )


# Highlight K nearest neighbors

for distance, category, point in nearest:

    # Highlight neighbor
    ax.scatter(
        point[0],
        point[1],
        point[2],
        facecolors='none',
        edgecolors='yellow',
        s=250,
        linewidths=2.5
    )

    # Highlight distance
    ax.plot(
        [new_point[0], point[0]],
        [new_point[1], point[1]],
        [new_point[2], point[2]],
        color='yellow',
        linestyle='-',
        linewidth=2.5
    )

new_class = clf.predict(new_point)
color = '#EF6C35' if new_class == 'orange' else '#104DCA'
ax.scatter(new_point[0], new_point[1], new_point[2], color=color, marker='*', s=200, zorder=100)

for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color='#104DCA', linestyle='--', linewidth=1)

for point in points['orange']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color='#EF6C35', linestyle='--', linewidth=1)

plt.savefig(
    "knn_3d.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# final results:
# Prediction: blue
# Nearest neighbors:
# point=[2, 3], class=blue, distance=1.000
# point=[3, 2], class=blue, distance=1.000
# point=[2, 4], class=blue, distance=1.414


# 3D KNN
# Prediction: blue
# Nearest neighbors:
# point=[3, 2, 3], class=blue, distance=1.414
# point=[2, 4, 3], class=blue, distance=1.732
# point=[1, 3, 5], class=blue, distance=2.236