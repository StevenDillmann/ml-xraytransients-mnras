
import os
import numpy as np
import pickle
from feature_extraction import PcaExtractor, AutoencoderExtractor
import sys
from tensorflow.keras.models import load_model

def extract_features(representation_path, feature_extractor= 'PCA', n_components=25, normalize=False):

    # Open the representation
    with open(representation_path, 'rb') as f:
        representations = pickle.load(f)
        id_list = representations.keys()
        rep_vals = representations.values()

    # Initialize extraction object
    if feature_extractor == 'PCA':
        extractor = PcaExtractor(n_components=n_components)
        save_path = representation_path.replace('representations.pkl', f'PCA{n_components}_features.pkl').replace('output/representations', 'output/features')
        histograms = np.array([np.array(h).flatten() for h in rep_vals])
        print('Extracting features using ' + feature_extractor + ' (' + str(n_components) + ' components)...')
    else:
        encoder = load_model(feature_extractor)
        extractor = AutoencoderExtractor(encoder)
        n_latents = int(encoder.layers[-1].output_shape[1])
        save_path = representation_path.replace('representations.pkl', f'AE{n_latents}_features.pkl').replace('output/representations', 'output/features')
        histograms = np.array([np.array(h) for h in rep_vals])
        print('Extracting features using AE ('+ feature_extractor + ')...')
    
    # Check if extracted features already exist
    if not os.path.exists('../output/features/'):
            os.makedirs('../output/features/')
    if os.path.exists(save_path):
        print(f'Extracted Features already exist at {save_path}.')
        return None
    else:

        # Extract the features
        features = extractor.extract_features(histograms, normalize=normalize)
        print('Feature Extraction Completed.')
        
        # Create the dictionary
        feature_dict = dict(zip(id_list, features))

        # Save the dictionary
        with open(save_path, 'wb') as f:
            pickle.dump(feature_dict, f)
        print(f'Extracted Features saved to {save_path}.')
        return None

def parse_args(args):
    # Defaults
    options = {
        'feature_extractor': 'PCA',
        'n_components': 25,
        'normalize': False,
    }

    # Map the arguments to the options
    arg_map = {
        '-f': 'feature_extractor',
        '-n': 'n_components',
        '-norm': 'normalize',
    }

    # Parse args into options
    for i in range(1, len(args), 2):
        key = args[i]
        if key in arg_map:
            value = args[i + 1]
            if key == '-norm':
                options[arg_map[key]] = value.lower() == 'true'
            else:
                options[arg_map[key]] = float(value) if '.' in value else int(value)
    
    return options

def main():
    if len(sys.argv) < 1:
        raise ValueError("Usage: python run_feature_extraction.py <representation_path> [-f feature_extractor] [-n n_components] [-norm normalize]")

    # Parse required arguments
    representation_path = sys.argv[1]

    # Parse optional arguments
    options = parse_args(sys.argv)

    # Call the function with the parsed arguments
    extract_features(representation_path, options['feature_extractor'], options['n_components'], options['normalize'])

if __name__ == "__main__":
    main()