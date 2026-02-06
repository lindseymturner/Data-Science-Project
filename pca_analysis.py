"""
Description: PCA analysis of bankruptcy dataset to visualize how companies
cluster in reduced dimensional space and identify key patterns.
Name: Max Lovinger, Lindsey Turner
Date: 12/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

# data
df = pd.read_csv('data.csv')

# Separate fts and labels
X = df.drop(columns=['Bankrupt?'])
y = df['Bankrupt?']

# Remove constant fts
X = X.loc[:, X.std() != 0]

# Normalize fts
X_normalized = (X - X.mean()) / X.std()

# Apply PCA : 2 components -- From sklearn 
pca = decomposition.PCA(n_components=2)
pca.fit(X_normalized)
X_pca = pca.transform(X_normalized)

# colors
color_dict = {0: 'g', 1: 'r'}
names = ['Not Bankrupt', 'Bankrupt']

# Assign colors
colors = [color_dict[label] for label in y]

# Plot - Plot majority class first, then minority class on top
plt.figure(figsize=(10, 8))

# Separate bankrupt and non-bankrupt
n_b = y == 0
b = y == 1

# plot non-bankrupt
plt.scatter(X_pca[n_b, 0], X_pca[n_b, 1], 
            c='g', edgecolor='k', alpha=0.3, s=30, label='Not Bankrupt (n=6525)')

# plot bankrupt
plt.scatter(X_pca[b, 0], X_pca[b, 1], 
            c='r', edgecolor='k', alpha=0.8, s=100, label='Bankrupt (n=294)', marker='X')

plt.title("PCA of Bankruptcy Dataset")
plt.xlabel(f"Component 1")
plt.ylabel(f"Component 2")
plt.legend()
plt.show()
