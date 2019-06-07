# Artificial Neural network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the training set and the test stet
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Part 2 - Making of ANN

# Importing the keras libraries and packages
import keras
from keras.layers import Dense
from keras.models import Sequential

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, input_dim=11, init='uniform', activation='relu'))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
""" If your output is a categorical variable then give the value 3 to the output_dim
parameter. In this way you will get 3 columns as output in case you have 3 dummy
variables."""
""" In the video he also explained some more queries, like what changes to make if 
you have more than 1 output variable."""

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)