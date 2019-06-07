# Diamonds

# Importing the libraries and the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('diamonds.csv')
dataset=dataset.drop(['Unnamed: 0'],axis=1) #Removing the useless 1st column.
X=dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values
y=dataset.iloc[:,6]

# Encoding the categorical variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(1,4):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Kernel PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca=PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
exp_variance=pca.explained_variance_ratio_

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

regressor.score(X_test, y_test)

n=[]
for i in range(1,400):
    n.append(i)
    
y_pred1=y_pred[1:400]
y_test1=y_test[1:400]
plt.scatter(n,y_test1,color='red')
plt.scatter(n,y_pred1,color='green')

from sklearn import metrics
accuracy = regressor.score(X_test,y_test)
print(accuracy*100,'%')

"""Study some ways to evaluate regression model performance using metrics or
any other method."""
