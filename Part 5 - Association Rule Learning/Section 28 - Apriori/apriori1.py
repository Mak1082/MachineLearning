# Apriori

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv', header=None) #Header = None is included to tell python that the first observation is not the heading.
transactions=[]
temp=[]
for i in range(0,7501):
    for j in range(0,20):
        temp.append(str(dataset.values[i,j]))
    transactions.append(temp)
    temp=[]

# Training Apriori on the dataset
from apyori import apriori
rules=apriori(transactions, min_support = 0.003, min_confidence=0.2, min_lift=3, min_length=3)

# Visualising the results
results=list(rules)
