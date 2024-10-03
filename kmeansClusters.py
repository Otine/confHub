import pandas as pd
from sklearn.cluster import KMeans
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Charger les données
data = pd.read_csv('input.csv')

# Sélectionner les colonnes numériques pour le clustering
numerical_cols = ['age', 'investment.total', 'financial_score']
X = data[numerical_cols]

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# Appliquer HDBSCAN
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
data['hdbscan_cluster'] = hdbscan_clusterer.fit_predict(X_scaled)

# Visualisation des clusters KMeans
plt.figure(figsize=(14, 7))

# Visualisation des clusters KMeans
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=data['kmeans_cluster'], palette='viridis')
plt.title('KMeans Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Ajout des annotations pour KMeans
for i in range(len(X_scaled)):
    plt.text(X_scaled[i, 0], X_scaled[i, 1], f"{round(data.iloc[i]['financial_score'], 2)}", fontsize=8)

# Visualisation des clusters HDBSCAN
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=data['hdbscan_cluster'], palette='viridis')
plt.title('HDBSCAN Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Ajout des annotations pour HDBSCAN
for i in range(len(X_scaled)):
    plt.text(X_scaled[i, 0], X_scaled[i, 1], f"{data.iloc[i]['situationPro']}", fontsize=8)

plt.tight_layout()
plt.show()
