# Big mart sales prediction

# Importing the libraries and the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
datset=pd.concat([train,test],sort=True)