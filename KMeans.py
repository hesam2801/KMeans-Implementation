import matplotlib.pyplot as plt
import numpy as np


df = np.loadtxt("datasets/iris.csv", delimiter=",", dtype=str)
display(df)


sepal_length=np.array(df[1:,0],dtype=np.float64)
sepal_width=np.array(df[1:,1],dtype=np.float64)
petal_length=np.array(df[1:,2],dtype=np.float64)
petal_width=np.array(df[1:,3],dtype=np.float64)

x1=sepal_length
x2=sepal_width
x3=petal_length
x4=petal_width


class KMeans:
    def __init__(self, data, n_clusters=3):
        self.data = data
        self.n_clusters = n_clusters

    def fit(self):
        X = self.data
        k = self.n_clusters
        clusters = X[np.random.choice(X.shape[0], k, replace=False)]
        while True:
            labels = np.array([np.argmin([np.sqrt(np.sum((point - c) ** 2)) for c in clusters]) for point in X])
            if X.shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50)
                ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], c='red', s=500, marker='x')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()
            else:
                plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
                plt.scatter(clusters[:, 0], clusters[:, 1], c='red', s=200, marker='x')
                plt.show()
            new_clusters = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            if np.all(clusters == new_clusters):
                break
            clusters = new_clusters
        return clusters, labels
    

A=np.array([x1,x2,x3]).T
print(A.shape)
k=KMeans(A,3)
clusters, labels=k.fit()
print(clusters)


A=np.array([x1,x2,x3,x4]).T
print(A.shape)
k=KMeans(A,3)
clusters, labels=k.fit()
print(clusters)