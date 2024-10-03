import pandas as pd
from sklearn.cluster import KMeans
import hdbscan
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

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

# Créer une figure 3D interactive pour KMeans
fig_kmeans = px.scatter_3d(
    data, 
    x='age', 
    y='investment.total', 
    z='financial_score', 
    color='kmeans_cluster',
    title='KMeans Clusters',
    labels={'age': 'Age', 'investment.total': 'Investment Total', 'financial_score': 'Financial Score'}
)

# Ajouter des annotations pour chaque point (facultatif)
fig_kmeans.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

# Afficher la visualisation interactive KMeans
fig_kmeans.show()

# Créer une figure 3D interactive pour HDBSCAN
fig_hdbscan = px.scatter_3d(
    data, 
    x='age', 
    y='investment.total', 
    z='financial_score', 
    color='hdbscan_cluster',
    title='HDBSCAN Clusters',
    labels={'age': 'Age', 'investment.total': 'Investment Total', 'financial_score': 'Financial Score'}
)

# Ajouter des annotations pour chaque point (facultatif)
fig_hdbscan.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

# Afficher la visualisation interactive HDBSCAN
fig_hdbscan.show()
