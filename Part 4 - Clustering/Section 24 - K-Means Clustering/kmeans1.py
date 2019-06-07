# K-Means Clustering

# Importing the libraries
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,3:5].values

# Applying the elbow method to select the number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.scatter(range(1,11) , wcss)
plt.plot(range(1,11) , wcss)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS value')
plt.show()

# Applying K-Means to the dataset
kmeans=KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans=kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='Red', label='Careful')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='Blue', label='Standard')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='Pink', label='Target')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='Green', label='Careless')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='Magenta', label='Sensible')
plt.title('Clusters of clients')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()