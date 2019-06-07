# My first project

# Importing the libraries and the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_excel('Iris.xls')
X=dataset.iloc[:,[0,1,2,3]].values
y=dataset.iloc[:,4].values

# Plotting sepal length vs sepal width to see the trend
plt.scatter(X[y=='Iris-setosa',0], X[y=='Iris-setosa',1],color='red', label='Iris-setosa')
plt.scatter(X[y=='Iris-versicolor',0], X[y=='Iris-versicolor',1], color='blue', label='Iris-versicolor')
plt.scatter(X[y=='Iris-virginica',0], X[y=='Iris-virginica',1], color='yellow', label='Iris-virginica')
plt.legend()
plt.show()

# Plotting petal length vs petal width to see the trend
plt.scatter(X[y=='Iris-setosa',2], X[y=='Iris-setosa',3],color='red', label='Iris-setosa')
plt.scatter(X[y=='Iris-versicolor',2], X[y=='Iris-versicolor',3], color='blue', label='Iris-versicolor')
plt.scatter(X[y=='Iris-virginica',2], X[y=='Iris-virginica',3], color='yellow', label='Iris-virginica')
plt.legend()
plt.show()

""" After looking at the two graphs it seems that petal features will give better predictions
than sepal features because in petal features 3 noticeable clusters are formed. So now we
will remove the sepal features from our matrix X"""
X=X[:,[2,3]]

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the test set and the training sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling for better predictions
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Fitting the classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5) 
classifier.fit(X_train, y_train)

# Predicting the results
y_pred=classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

# Using k fold cross validation to evaluate model performance
from sklearn.cross_validation import cross_val_score #In the video sklearn.model_selection is used which is not present in this scikit version
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()