from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

class DimensionalityReduction:
    def __init__(self):
        self.tsne_results = None

    def tsne_reduction(self, X, n_components=2, perplexity=50, learning_rate=100, n_iter=1000,  early_exaggeration=1, init = 'random', random_state=505, **kwargs):
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate = learning_rate, n_iter=n_iter, early_exaggeration=early_exaggeration, init = init, random_state=random_state, **kwargs)
        self.tsne_results = tsne.fit_transform(X)
        return self.tsne_results
    
    def visualize(self, tsne_input = None, figsize=(8, 6), colorcode = None, cmap = 'viridis_r', size = 0.1, label = None, **kwargs):
        if tsne_input is not None:
            self.tsne_results = tsne_input
        if self.tsne_results is None:
            raise ValueError("The t-SNE projection must be computed before visualization, or pass is as an input.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        if colorcode is None:
            ax.scatter(self.tsne_results[:, 0], self.tsne_results[:, 1], s = size, cmap=cmap, **kwargs)
        else:
            se = ax.scatter(self.tsne_results[:, 0], self.tsne_results[:, 1], s = size, c=colorcode, cmap=cmap, **kwargs)
            cbar = fig.colorbar(se, ax = ax)
            cbar.set_label(label, rotation=270)
        return ax

class EmbeddingClustering:
    def __init__(self):
        self.dbscan_results = None

    def dbscan_clustering(self, X, eps=2.5, min_samples=25, **kwargs):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.dbscan_results = dbscan.fit_predict(X) + 1
        return self.dbscan_results
    
    def visualize(self, X, dbscan_clusters, figsize=(6.5, 6), size = 0.1, **kwargs):
        if dbscan_clusters is not None:
            self.dbscan_results = dbscan_clusters
        if self.dbscan_results is None:
            raise ValueError("The DBSCAN clustering must be computed before visualization.")
        
        # Create dataframe for sorting the clusters
        df = pd.DataFrame({'tsne1': X[:, 0], 'tsne2': X[:, 1], 'cluster': self.dbscan_results})
        df = df.sort_values(by='cluster')
        num_clusters = len(np.unique(df['cluster']))

        # Define cluster colors
        cluster_colors = ['silver', 'black', 'peru', 'cornflowerblue', 'crimson', 'forestgreen', 'orange', 'purple', 'gold', 'turquoise', 'pink', 'darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'darkkhaki', 'darkturquoise', 'hotpink', 'brown', 'lightblue','lightcoral', 'lightgreen', 'lightsalmon', 'lavender', 'cornsilk', 'lightcyan', 'lightpink', 'tomato', 'royalblue', 'springgreen', 'orangered', 'tan', 'mediumvioletred', 'mediumslateblue', 'mediumseagreen']
        if num_clusters > len(cluster_colors):
            additional_colors = np.random.rand(num_clusters - len(cluster_colors), 3)
            cluster_colors.extend([tuple(color) for color in additional_colors])
        cluster_cmap = ListedColormap(cluster_colors[:num_clusters])

        # Plot the t-SNE embedding
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.scatter(df['tsne1'], df['tsne2'], c=df['cluster'], cmap=cluster_cmap, s = size, **kwargs)

        unique_labels = sorted(np.unique(df['cluster']))
        cluster_legend_handles = []
        for i, label in enumerate(unique_labels):
            color = cluster_colors[i % len(cluster_colors)]
            if label == 0:
                label_name = 'Outlier (0)'
            else:
                label_name = f'Cluster {label}'
            cluster_legend_handles.append(mpatches.Patch(color=color, label=label_name))
            
            ax.legend(handles=cluster_legend_handles, bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=2, frameon=False)

        return ax, cluster_legend_handles
    