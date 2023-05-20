from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

# Load dataset
dataset = pd.read_csv('framingham_modified (1).csv')

# Scale the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(dataset.drop('target', axis=1))

# Choose the number of clusters (K)
k = 100

# Create KMeans model and fit it to the data
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Predict the clusters for the data
y_pred = kmeans.predict(X)

# Calculate silhouette score
silhouette = silhouette_score(X, y_pred)
print(f'K-Means Clustering Silhouette Score: {silhouette}')
