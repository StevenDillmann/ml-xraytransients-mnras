import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LightCurve:
    def __init__(self):
        self.lightcurves = None
        self.binned_lightcurves = None

    def bin(self, df_eventfiles_input, bin_size_sec=500):
        # Prepare df
        df = df_eventfiles_input.copy()
        df['time'] = df_eventfiles_input['time'] - min(df_eventfiles_input['time'])
        df = df.sort_values(by='time') 
        df = df.reset_index(drop=True)
        # Create binned lightcurve
        # Define the bins: the range of times divided into intervals of bin_size_sec
        max_time = df['time'].max()
        bins = np.arange(0, max_time + bin_size_sec, bin_size_sec)
        df['bin'] = np.digitize(df['time'], bins) - 1
        # Group by bin and count the events in each bin
        binned_counts = df.groupby('bin').size()
        # Error for each bin assuming Poisson statistics
        binned_errors = np.sqrt(binned_counts)
        # Time for each bin (use the bin edges)
        bin_centers = bins[:-1] + np.diff(bins) / 2
        bin_centers_ks = bin_centers / 1000
        # Only plot bins where there are counts
        valid_bins = binned_counts.index.values
        binned_counts = binned_counts.reindex(np.arange(len(bin_centers)), fill_value=0)
        binned_errors = binned_errors.reindex(np.arange(len(bin_centers)), fill_value=0)
        # running average of the lightcurve
        binned_counts_running = binned_counts.rolling(window=3, center=True).mean()
        # Save the lightcurves 
        self.binned_lightcurves = {'bin_size_sec': bin_size_sec, 'bin_centers_ks': bin_centers_ks, 'binned_counts': binned_counts, 'binned_errors': binned_errors, 'binned_counts_running': binned_counts_running}
        return self.binned_lightcurves
    
    def visualise(self, color='k', ecolor='k', ccolor=None, rcolor='red', lw=2, lwe = 0.5, capsize=1, markersize=5):
        if self.binned_lightcurves is None:
            print('Please bin the lightcurve first.')
        bin_size_sec = self.binned_lightcurves['bin_size_sec']
        bin_centers_ks = self.binned_lightcurves['bin_centers_ks']
        binned_counts = self.binned_lightcurves['binned_counts']
        binned_errors = self.binned_lightcurves['binned_errors']
        binned_counts_running = self.binned_lightcurves['binned_counts_running']
        plt.figure(figsize=(8.5, 5.5))
        plt.errorbar(bin_centers_ks, binned_counts, yerr=binned_errors, fmt='.', color=color, ecolor=ecolor, capsize=capsize, markersize=markersize, lw=lwe, label = f'{int(bin_size_sec)}s Bins')
        plt.plot(bin_centers_ks, binned_counts_running, color=ecolor, lw=0)
        if ccolor is not None:
            plt.plot(bin_centers_ks, binned_counts, color=ccolor, lw=lw, label = 'Lightcurve')
        if rcolor is not None:
            plt.plot(bin_centers_ks, binned_counts_running, color=rcolor, lw=lw, label = 'Running Average Lightcurve')
        plt.xlabel('Time [ks]')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()
