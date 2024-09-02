
import os
import numpy as np
import pandas as pd
import pickle
from feature_projection import DimensionalityReduction
import sys


def reduce_dimensions(feature_path, n_components=2, perplexity=50, learning_rate=100, n_iter=1000, early_exaggeration=1, init='random', random_state=505, **kwargs):

    # Load the features
    with open(feature_path, 'rb') as f:
        feature_dict = pickle.load(f)
    id_list = list(feature_dict.keys())
    features = np.array(list(feature_dict.values()))

    # Initialize the dimensionality reduction object
    dr = DimensionalityReduction()
    save_path = feature_path.replace('features.pkl', f'{n_components}Dperp{perplexity}lr{learning_rate}iter{n_iter}exag{early_exaggeration}init{init}rs{random_state}_embedding.csv').replace('output/features', 'output/embeddings')

    # Check if embedding already exists
    print(f'Performing Dimensionality Reduction...')
    if not os.path.exists('../output/embeddings/'):
            os.makedirs('../output/embeddings/')
    if os.path.exists(save_path):
        print(f'Embedding already exists at {save_path}.')
        return None
    else:

        # Perform t-SNE dimensionality reduction
        embedding = dr.tsne_reduction(features, n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, early_exaggeration=early_exaggeration, init=init, random_state=random_state, **kwargs)
        print('Dimensionality Reduction Completed.')

        # Save the embedding
        df = pd.DataFrame(embedding, columns=[f'tsne{i+1}' for i in range(embedding.shape[1])])
        df['obsreg_id'] = id_list
        df.to_csv(save_path, index=False)
        print(f'Embedding saved to {save_path}.')
        return None

def parse_args(args):
    # Defaults
    options = {
        'n_components': 2,
        'perplexity': 50,
        'learning_rate': 100,
        'n_iter': 1000,
        'early_exaggeration': 1,
        'init': 'random',
        'random_state': 505,
    }

    # Map the arguments to the options
    arg_map = {
        '-n': 'n_components',
        '-p': 'perplexity',
        '-lr': 'learning_rate',
        '-iter': 'n_iter',
        '-exag': 'early_exaggeration',
        '-init': 'init',
        '-rs': 'random_state',
    }

    # Parse args into options
    for i in range(1, len(args), 2):
        key = args[i]
        if key in arg_map:
            value = args[i + 1]
            if key == '-init':
                options[arg_map[key]] = value
            else:
                options[arg_map[key]] = float(value) if '.' in value else int(value)

    return options

def main():
    if len(sys.argv) < 1:
        raise ValueError("Usage: python run_dimensionality_reduction.py <feature_path> [-n n_components] [-p perplexity] [-lr learning_rate] [-iter n_iter] [-exag early_exaggeration] [-init init] [-rs random_state]")
    
    # Parse required arguments
    feature_path = sys.argv[1]

    # Parse optional arguments
    options = parse_args(sys.argv)

    # Call the function with the parsed arguments
    reduce_dimensions(feature_path, options['n_components'], options['perplexity'], options['learning_rate'], options['n_iter'], options['early_exaggeration'], options['init'], options['random_state'])

if __name__ == "__main__":
    main()
