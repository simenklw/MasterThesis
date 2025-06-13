import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#%matplotlib widget

#Read SNP data
df = pd.read_feather("data/processed/massBV.feather")

island_count = df.hatchisland.value_counts()    

#Remove islands with less than 10 (or some other number) observations
def filter_islands(df, island_count, threshold):
    islands = island_count[island_count > threshold].index
    return df[df.hatchisland.isin(islands)]

df = filter_islands(df, island_count, 10)

#Set X to be SNP data
X = np.nan_to_num(df.iloc[:, 9:].values)

# Center and scale each feature
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#Run t-SNE for two dimensions. Set perplexity
tsne = TSNE(n_components=3, random_state=42, perplexity=50, metric='cosine')
X_tsne = tsne.fit_transform(X)