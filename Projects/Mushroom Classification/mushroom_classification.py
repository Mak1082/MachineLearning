# Mushroom Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('mushrooms.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Encoding Categorical variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(0,22):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

# Splitting the dataset into the training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""# Using PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
exp_variance=pca.explained_variance_ratio_"""

# Fitting the classifier to the training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', gamma=0.045, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the results
y_pred=classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

# Aplying K-Fold cross validation for evaluatin model performance
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()

# Applying Grid Search for finding the optimal model and the optimal parameters
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000], 'kernel':['linear']},
             {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.045, 0.04, 0.03, 0.02, 0.001, 0.0001] }
             ]
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_search=grid_search.fit(X_train, y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_