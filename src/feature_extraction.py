import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class PcaExtractor:
    def __init__(self, n_components=25, random_state = 11, **kwargs):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state, **kwargs)
        self.pca_results = None

    def extract_features(self, X, normalize=False):
        pca_features = self.pca.fit_transform(X)
        if normalize:
            scaler = StandardScaler()
            self.pca_results = scaler.fit_transform(pca_features)
        else:
            self.pca_results = pca_features
        return self.pca_results

    def visualize(self, figsize=(8,5), color_eigenvalues='blue', color_cumulative='red'):

        if self.pca_results is None:
            raise ValueError("The PCA model must be fitted before visualization. Call fit_transform() first.")
        
        # Get the eigenvalues
        eigenvalues = self.pca.explained_variance_
        cumulatives = np.cumsum(self.pca.explained_variance_ratio_)

        # Create subplots 
        fig, ax1 = plt.subplots(figsize=figsize) 

        # Plot the scree plot
        bar = ax1.bar(range(len(eigenvalues)), eigenvalues, alpha=1, color = color_eigenvalues,label = 'Abs. Individual Variance')
        ax1.set_xlabel(r'Principal Component')
        ax1.set_ylabel(r'Eigenvalue')
        ax1.yaxis.label.set_color(color_eigenvalues)
        ax1.tick_params(axis='y', colors=color_eigenvalues)
        ax1.set_xticks(range(-1,len(eigenvalues),5))
        ax1.set_xticklabels([i+1 for i in range(-1,len(eigenvalues),5)])

        # Plot the cumulative variance plot
        ax2 = ax1.twinx()
        cumu = ax2.plot(cumulatives, c=color_cumulative,label = 'Rel. Cumulative Variance', marker='.')
        ax2.set_ylabel('Cumulative Variance')
        ax2.yaxis.label.set_color(color_cumulative)
        ax2.tick_params(axis='y', colors=color_cumulative)
        ax2.tick_params(which='both', direction='in', top=True, right=True)

        # Combine the legends
        handles, labels = ax1.get_legend_handles_labels()
        handles += cumu
        labels += ['Rel. Cumulative Variance']
        plt.legend(handles=handles, labels=labels, loc='lower right', frameon=False)
        plt.show()
        return
    
class AutoencoderExtractor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.ae_results = None

    def extract_features(self, X, normalize=False):
        latent_features = self.encoder.predict(X)
        if normalize:
            scaler = StandardScaler()
            self.ae_results = scaler.fit_transform(latent_features)
        else:
            self.ae_results = latent_features
        return self.ae_results