import os
import pandas as pd
import pickle
from eventfile_representation import EtMap, EtdtCube
import sys

def create_representation(eventfiles_path, representation_type = 'et', t_bins=24, E_bins=16, dt_bins = 16, E_min=500, E_max=7000, normalize=True):

    # Load and group the event files
    df_eventfiles = pd.read_csv(eventfiles_path).groupby('obsreg_id')

    # Initialize representation object
    if representation_type == 'et':
        representation = EtMap(t_bins=t_bins, E_bins=E_bins, E_min=E_min, E_max=E_max, normalize=normalize)
        save_path = f'../output/representations/{representation_type}_{E_bins}-{t_bins}_norm{normalize}_representations.pkl'
    elif representation_type == 'etdt':
        representation = EtdtCube(t_bins=t_bins, E_bins=E_bins, dt_bins=dt_bins, E_min=E_min, E_max=E_max, normalize=normalize)
        save_path = f'../output/representations/{representation_type}_{E_bins}-{t_bins}-{dt_bins}_norm{normalize}_representations.pkl'

    # Check if representations already exist
    if os.path.exists(save_path):
        print(f'Representations already exist at {save_path}.')
        return None
    else:
    
        # Initialize dictionary lists
        representation_list = []
        id_list = []

        # Initialize loop counter
        count = 0
        count_limit = df_eventfiles.ngroups
        clear_command = 'cls' if os.name == 'nt' else 'clear'

        # Loop and create the representations
        for id_name, df in df_eventfiles:
            id_list.append(id_name)
            t_array = df['time'].values
            E_array = df['energy'].values
            representation_list.append(representation.create_representation(t_array, E_array))
            count = count + 1
            print(f'Counter: {count} of {count_limit}', end='\r')
        print(f'Event File Representation Completed.')

        # Create the dictionary
        representation_dict = dict(zip(id_list, representation_list))

        # Save the dictionary
        if not os.path.exists('../output/representations/'):
            os.makedirs('../output/representations/')
        with open(save_path, 'wb') as f:
            pickle.dump(representation_dict, f)
        print(f'Event File Representation saved to {save_path}.')
        return None

def parse_args(args):
    # Defaults
    options = {
        't_bins': 24,
        'E_bins': 16,
        'dt_bins': 24,
        'E_min': 500.0,
        'E_max': 7000.0,
        'normalize': True
    }

    # Map the arguments to the options
    arg_map = {
        '-tb': 't_bins',
        '-Eb': 'E_bins',
        '-dtb': 'dt_bins',
        '-emin': 'E_min',
        '-emax': 'E_max',
        '-norm': 'normalize'
    }

    # Parse args into options
    for i in range(3, len(args), 2):
        key = args[i]
        if key in arg_map:
            value = args[i + 1]
            if key == '-norm':
                options[arg_map[key]] = value.lower() == 'true'
            else:
                options[arg_map[key]] = float(value) if '.' in value else int(value)
    
    return options


def main():
    if len(sys.argv) < 3:
        raise ValueError("Usage: python run_eventfile_representation.py <eventfiles_path> <representation_type> [-tb t_bins] [-Eb E_bins] [-dtb dt_bins] [-emin E_min] [-emax E_max] [-norm normalize]")
    
    # Parse required arguments
    eventfiles_path = sys.argv[1]
    representation_type = sys.argv[2]

    # Parse optional arguments
    options = parse_args(sys.argv)

    # Call the function with the parsed arguments
    create_representation(eventfiles_path, representation_type, options['t_bins'], options['E_bins'], options['dt_bins'], options['E_min'], options['E_max'], options['normalize'])

if __name__ == "__main__":
    main()