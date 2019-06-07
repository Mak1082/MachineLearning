# XG Boost

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding the categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder1=LabelEncoder()
label_encoder2=LabelEncoder()
X[:,1]=label_encoder1.fit_transform(X[:,1])
X[:,2]=label_encoder2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred=classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

# Aplying k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()