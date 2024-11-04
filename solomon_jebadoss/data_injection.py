import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = './data/final_adjusted_crowd_dataset.csv'
df = pd.read_csv(file_path)
df.head()