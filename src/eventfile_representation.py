import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class EtMap:
    def __init__(self, t_bins=24, E_bins=16, E_min=500, E_max=7000, normalize=True):
        """
        Initialize the EtMap class with binning and normalization parameters.
        
        Parameters:
        t_bins (int): Number of bins for the time axis.
        E_bins (int): Number of bins for the energy axis.
        E_min (float): Minimum energy value for the energy axis range.
        E_max (float): Maximum energy value for the energy axis range.
        normalise (bool): Whether to normalize the histogram.
        """
        self.t_bins = t_bins
        self.E_bins = E_bins
        self.normalize = normalize
        self.E_min = E_min
        self.E_max = E_max
        self.et_map = None

    def create_representation(self, t_array, E_array):
        """
        Create the energy-time map representation from input time and energy arrays.
        
        Parameters:
        t_array (array-like): Array of time values.
        E_array (array-like): Array of energy values.
        
        Returns:
        et_map (array-like): The created energy-time map.
        """
        # Normalize E and t
        eps_array = np.log10(E_array)
        tau_array = (t_array - min(t_array)) / (max(t_array) - min(t_array))

        # Create E-t map
        histogram, _, _ = np.histogram2d(
            tau_array, 
            eps_array, 
            range=[[0, 1], [np.log10(self.E_min), np.log10(self.E_max)]], 
            bins=(self.t_bins, self.E_bins)
        )

        # Normalize the histogram
        if self.normalize:
            self.et_map = (histogram - np.min(histogram)) / (np.max(histogram) - np.min(histogram))
        else:
            self.et_map = histogram

        return self.et_map

    def visualize(self, figsize=(4,4), cmap='viridis', lognorm=False):
        """
        Visualize the energy-time map.
        
        Parameters:
        figsize (tuple): Size of the figure to create.
        cmap (str): Colormap to use for visualization.
        lognorm (bool): Whether to apply logarithmic normalization to the colormap.
        """
        if self.et_map is None:
            raise ValueError("No representation found. Please call `create_representation` first.")

        plt.figure(figsize=figsize)
        plt.imshow(
            self.et_map.T,
            origin='lower',
            aspect='auto',
            extent=[0, 1, np.log10(self.E_min), np.log10(self.E_max)],
            cmap=cmap,
            norm=LogNorm() if lognorm else None
        )
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\epsilon$')
        plt.title(r'$E-t$ Map')
        plt.show()


class EtdtCube:
    def __init__(self, t_bins=24, E_bins=16, dt_bins=16, E_min=500, E_max=7000, normalize=True):
        """
        Initialize the EtdtCube class with binning and normalization parameters.

        Parameters:
        t_bins (int): Number of bins for the time axis.
        E_bins (int): Number of bins for the energy axis.
        dt_bins (int): Number of bins for the delta time axis.
        E_min (float): Minimum energy value for the energy axis range.
        E_max (float): Maximum energy value for the energy axis range.
        normalise (bool): Whether to normalize the histogram.
        """
        self.t_bins = t_bins
        self.E_bins = E_bins
        self.dt_bins = dt_bins
        self.normalize = normalize
        self.E_min = E_min
        self.E_max = E_max
        self.etdt_cube = None
        self.edges = None

    def create_representation(self, t_array, E_array):
        """
        Create the energy-time-delta time cube representation from input time and energy arrays.

        Parameters:
        t_array (array-like): Array of time values.
        E_array (array-like): Array of energy values.

        Returns:
        etdt_cube (array-like): The created energy-time-delta time cube.
        """
        # Create dt array
        dt_array = np.diff(t_array)

        # Normalize E, t, and dt
        eps_array = np.log10(E_array[:-1])
        tau_array = (t_array[:-1] - min(t_array[:-1])) / (max(t_array[:-1]) - min(t_array[:-1]))
        dtau_array = (dt_array - min(dt_array)) / (max(dt_array) - min(dt_array))

        # Create E-t-dt cube
        histogram, edges = np.histogramdd(
            (tau_array, eps_array, dtau_array), 
            range=[[0, 1], [np.log10(self.E_min), np.log10(self.E_max)], [0, 1]], 
            bins=(self.t_bins, self.E_bins, self.dt_bins)
        )

        # Store the edges for plotting purposes
        self.edges = edges

        # Normalize the histogram
        if self.normalize:
            self.etdt_cube = (histogram - np.min(histogram)) / (np.max(histogram) - np.min(histogram))
        else:
            self.etdt_cube = histogram

        return self.etdt_cube

    def visualize(self, figsize=(10, 10), cmap='viridis', lognorm=False):
        """
        Visualize the energy-time-delta time cube.

        Parameters:
        figsize (tuple): Size of the figure to create.
        cmap (str): Colormap to use for visualization.
        lognorm (bool): Whether to apply logarithmic normalization to the colormap.
        """
        if self.etdt_cube is None:
            raise ValueError("No representation found. Please call `create_representation` first.")

        # Create the figure
        fig = plt.figure(figsize=figsize, constrained_layout=True)
  
        # Plot the E-t projection
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(self.etdt_cube.sum(axis=2).T, origin='lower',
                   extent=[0, 1, np.log10(self.E_min), np.log10(self.E_max)], cmap=cmap, norm=LogNorm() if lognorm else None)
        ax1.set_xlabel(r'$\tau$')
        ax1.set_ylabel(r'$\epsilon$')
        ax1.set_title(r'$E-t$ Projection')

        # Plot the dt-t projection
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(self.etdt_cube.sum(axis=1).T, origin='lower', extent=[0, 1, 0, 1], cmap=cmap, norm=LogNorm() if lognorm else None)
        ax2.set_xlabel(r'$\tau$')
        ax2.set_ylabel(r'$\delta\tau$')
        ax2.set_title(r'$dt-t$ Projection')

        # Plot the E-dt projection
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(self.etdt_cube.sum(axis=0), origin='lower', extent=[0, 1, np.log10(self.E_min), np.log10(self.E_max)], cmap=cmap, norm=LogNorm() if lognorm else None)
        ax3.set_xlabel(r'$\delta\tau$')
        ax3.set_ylabel(r'$\epsilon$')
        ax3.set_title(r'$E-dt$ Projection')

        # Plot the E-t-dt cube in 3D
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        t_mesh, E_mesh, dt_mesh = np.meshgrid(self.edges[0][:-1], self.edges[1][:-1], self.edges[2][:-1], indexing='ij')
        t_mesh = np.ravel(t_mesh)
        E_mesh = np.ravel(E_mesh)
        dt_mesh = np.ravel(dt_mesh)
        Etdt_mesh = np.ravel(self.etdt_cube)
        ax4.scatter(dt_mesh, t_mesh, E_mesh, s=2*np.log(Etdt_mesh*10000), alpha=0.9, edgecolors='k',
                    c=Etdt_mesh, cmap=cmap, norm=LogNorm() if lognorm else None, linewidths=0.4)
        ax4.set_xlabel(r'$\delta\tau$')
        ax4.set_ylabel(r'$\tau$')
        ax4.set_zlabel(r'$\epsilon$')
        ax4.set_title(r'$E-t-dt$ Cube')
        ax4.view_init(elev=30, azim=45)
        ax4.xaxis.set_ticks_position('bottom')
        ax4.yaxis.set_ticks_position('top')
        ax4.zaxis.set_ticks_position('bottom')
        ax4.invert_xaxis()

        plt.show()
