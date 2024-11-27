#Importing the Python Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
# Load the Dataset
data = pd.read_csv('raw_crowd_dataset.csv')
data.info()
# Finding Null Values
data.isnull().sum()
# Finding duplicates
data.duplicated().sum()
# Removing duplicates (removing from the dataset)
data = data.drop_duplicates()
data.duplicated().sum()
