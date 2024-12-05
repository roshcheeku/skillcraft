import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('sct_ml_02\Mall_Customers.csv')  

print(df.head())

print(df.isnull().sum())

df_cleaned = df.drop(['CustomerID'], axis=1)

df_cleaned['Gender'] = df_cleaned['Gender'].map({'Male': 0, 'Female': 1})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

df_cleaned = df_cleaned.fillna(df_cleaned.mean())  # Filling missing values with the column mean

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

optimal_k = 5  

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(df_scaled)

df['Cluster'] = y_kmeans

print(df.head())

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = y_kmeans

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments using KMeans Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


numeric_df = df.drop(columns=['CustomerID'])
cluster_summary = numeric_df.groupby('Cluster').mean()
print(cluster_summary)
