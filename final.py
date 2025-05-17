import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import os

try:
    # Load CSV
    df = pd.read_csv("database_24_25.csv")

    # Convert MP (Minutes Played) from MM.SS to float
    df['MinutesPlayed'] = df['MP'].astype(str).str.split('.').apply(
        lambda x: int(x[0]) + int(x[1])/60 if len(x) == 2 else float(x[0]))

    # Average per player stats
    df_avg = df.groupby('Player').agg({
        'MinutesPlayed': 'mean',
        'PTS': 'mean',
        'AST': 'mean',
        'TRB': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'TOV': 'mean',
        'FG%': 'mean',
        '3P%': 'mean',
        'FT%': 'mean',
        'GmSc': 'mean'
    }).dropna()

    # Filter low-minute players
    df_avg = df_avg[df_avg['MinutesPlayed'] > 15]

    # Standardize data
    features = df_avg.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_avg)

    # Compute distance matrix
    distance_matrix = euclidean_distances(X_scaled)

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_avg['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_avg['PC1'], df_avg['PC2'] = pca_result[:, 0], pca_result[:, 1]

    # === Plot 1: PCA Clusters ===
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_avg, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, edgecolor='w')
    plt.title("PCA Projection of NBA Player Clusters", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_clusters.png")
    plt.close()

    # === Plot 2: Similarity Heatmap ===
    target_player = 'Jrue Holiday'
    if target_player in df_avg.index:
        idx = df_avg.index.tolist().index(target_player)
        sim_indices = np.argsort(distance_matrix[idx])[1:6]
        similar_players = df_avg.index[sim_indices]
        heatmap_data = pd.DataFrame(distance_matrix[idx][sim_indices].reshape(1, -1),
                                    columns=similar_players)
        plt.figure(figsize=(8, 2))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar=False)
        plt.title(f"Top 5 Similar Players to {target_player}")
        plt.yticks([])
        plt.tight_layout()
        plt.savefig("similarity_heatmap.png")
        plt.close()

    # === Plot 3: Radar Chart for Cluster Profiles ===
    cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    radar_data = (cluster_centroids - cluster_centroids.min()) / (cluster_centroids.max() - cluster_centroids.min())

    # Radar setup
    labels = features.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # Complete the circle
    labels += [labels[0]]
    angles += [angles[0]]

    # Plot
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, polar=True)

    for i in range(radar_data.shape[0]):
        values = radar_data.iloc[i].tolist()
        values.append(values[0])  # wrap around to match angles
        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_title('Cluster Centroid Feature Profiles (Radar Chart)', size=14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig("cluster_radar.png")
    plt.close()

    print("✅ Script finished successfully. All plots saved in Downloads folder.")

except Exception as e:
    print("❌ Error occurred:")
    traceback.print_exc()

    print("Saving to directory:", os.getcwd())

