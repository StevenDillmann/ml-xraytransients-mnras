
import os
import numpy as np
import pandas as pd
import pickle
from feature_projection import EmbeddingClustering
import sys

def cluster_embeddings(embedding_path, eps=2.5, min_samples=25, **kwargs):
    
        # Load the embeddings
        df_embedding = pd.read_csv(embedding_path)
        embeddings = df_embedding.drop(['obsreg_id'], axis=1).values
    
        # Initialize the clustering object
        ec = EmbeddingClustering()
        save_path = embedding_path.replace('_embedding.csv', f'_clusters_eps{eps}ms{min_samples}.csv')
    
        # Check if clustering already exists
        print(f'Performing Embedding Clustering...')
        if not os.path.exists('../output/embeddings/'):
                os.makedirs('../output/embeddings/')
        if os.path.exists(save_path):
            print(f'Embedding Clustering already exists at {save_path}.')
            return None
        else:
    
            # Perform DBSCAN clustering
            clustering = ec.dbscan_clustering(embeddings, eps=eps, min_samples=min_samples, **kwargs)
            print('Embedding Clustering Completed.')
            
            # Save the clustered embedding
            df_embedding['cluster'] = clustering
            df_embedding.to_csv(save_path, index=False)
            return None

def parse_args(args):
    # Defaults
    options = {
        'eps': 2.5,
        'min_samples': 25,
    }

    # Map the arguments to the options
    arg_map = {
        '-eps': 'eps',
        '-ms': 'min_samples',
    }

    # Parse the arguments
    for i in range(1, len(args), 2):
        if args[i] in arg_map:
            options[arg_map[args[i]]] = float(args[i + 1])
        else:
            print(f'Invalid argument: {args[i]}')
            sys.exit(1)

    return options

def main():
    if len(sys.argv) < 1:
        raise ValueError("Usage: python run_embedding_clustering.py <embedding_path> [-eps eps] [-ms min_samples]")
    
    # Parse required arguments
    embedding_path = sys.argv[1]

    # Parse optional arguments
    options = parse_args(sys.argv)

    # Call the function with the parsed arguments
    cluster_embeddings(embedding_path, options['eps'], options['min_samples'])

if __name__ == "__main__":
    main()

    