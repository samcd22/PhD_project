import bisect
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd

from matplotlib import pyplot as plt
from typing import List, Tuple


class BoxGridder:
    """
    Class to create a 3D grid enclosing 3D spatial data.
    The class averages the data points in each grid box.
    The grid box centroids are set as the location of each averaged value.
    The class can visualise either averages, or data points count, in each box.
    The class can generate histograms for a random sample of grid boxes.

    Arguments:
        - data: DataFrame containing the 3D spatial data.
        - grid_size: List containing the grid size in each dimension.
        - output_path: String, path to the data directory.
        - independent_variables: String List of independent variable names.
        - dependent_variable: String, the dependent variable name.
        - input_data_logarithmised: Boolean, indicating logarithmised data.
        - output_data_logarithmised: Boolean, flag output logarithmisation.
        - processed_data: DataFrame containing the already averaged data.

    Attributes:
        - data: DataFrame containing the 3D spatial data.
        - grid_size: List containing the grid size in each dimension.
        - output_path: String, path to the data directory.
        - independent_variables: String List of independent variable names.
        - dependent_variable: String, the dependent variable name.
        - input_data_logarithmised: Boolean, indicating logarithmised data.
        - output_data_logarithmised: Boolean, flag output logarithmisation.
        - processed_data: DataFrame containing the averaged data.
        - samples: List of tuples, containing the samples within each box.
    """

    def __init__(self,
                 data: pd.DataFrame = None,
                 grid_size: List[float] = None,
                 output_path: str = None,
                 independent_variables: List[str] = ['x', 'y', 'z'],
                 dependent_variable: str = 'C',
                 input_data_logarithmised: bool = False,
                 output_data_logarithmised: bool = False,
                 processed_data=None) -> None:

        """
        Initialise a BoxGridder object.
        Generates averaged data if it wasn't already input.

        Arguments:
            - data: DataFrame containing the 3D spatial data.
            - grid_size: Tuple containing the grid size in each dimension.
            - output_path: String, path to the data directory.
            - independent_variables: String List of independent variable names.
            - dependent_variable: String, the dependent variable name.
            - input_data_logarithmised: Boolean, indicating logarithmised data.
            - output_data_logarithmised: Boolean, flag output logarithmisation.
            - processed_data: DataFrame containing the averaged data.
        """

        self.data = data
        self.grid_size = grid_size
        self.output_path = output_path
        self.independent_variables = independent_variables
        self.dependent_variable = dependent_variable
        self.input_data_logarithmised = input_data_logarithmised
        self.output_data_logarithmised = output_data_logarithmised

        output_path_csv = output_path + '/data_plus.csv'
        if os.path.exists(output_path_csv):
            self.processed_data = pd.read_csv(output_path_csv)
        else:
            os.makedirs(output_path, exist_ok=True)
            if processed_data is not None:
                self.processed_data = processed_data
            else:
                self.processed_data = self._get_averages()
                self.processed_data.to_csv(output_path_csv, index=False)

    def _flexible_floor(self, x: float, base: float) -> float:
        """
        Round down to a given base number, i.e. a flexible floor operation.
            Examples:
                _flexible_floor(3.1, 1.0) gives 3.0.
                _flexible_floor(3.1, 2.0) gives 2.0.

        Arguments:
            - x: Float, number to be rounded.
            - base: Float, base number.

        Returns:
            - Float, rounded number.
        """
        return base * np.floor(x/base)

    def _get_bounds(
            self, haystack: np.ndarray, needle: float) -> Tuple[float, float]:
        """
        Check which grid boundary points a point of interest lies between.

        Arguments:
            - haystack: Numpy array, grid points.
            - needle: Float, point of interest.

        Returns:
            - Tuple of floats, lower and upper bounds.
        """
        idx = bisect.bisect(haystack, needle)
        if 0 < idx < len(haystack):
            return haystack[idx-1], haystack[idx]
        else:
            raise ValueError(
                f"BoxGridder - {needle} is out of bounds of {haystack}")

    def _get_averages(self) -> pd.DataFrame:
        """
        Calculate the averages of the data points within each box.

        Each point is assigned to the box [lower, lower + size) along every axis, boxes are laid
        from the flexible floor of the minimum to the flexible floor of the maximum plus one box,
        and the cell takes the mean of the (linear) values at its centroid. Empty cells are dropped.
        Vectorised: linear in the number of points (the previous implementation rebuilt the whole
        output table once per point).

        Returns:
            - DataFrame: averaged data with columns [x, y, z, dependent, counts, samples].
        """
        if self.data is None:
            raise Exception('RawDataProcessor - Data not provided')
        if self.grid_size is None:
            raise Exception('RawDataProcessor - Grid size not provided')

        print('Generating gridded averages...')
        cols = self.independent_variables
        pts = self.data[cols].to_numpy(dtype=float)
        values = self.data[self.dependent_variable].to_numpy(dtype=float)
        if self.input_data_logarithmised:
            values = 10.0 ** values
        sizes = np.asarray(self.grid_size, dtype=float)
        mins = np.array([self._flexible_floor(pts[:, d].min(), sizes[d]) for d in range(3)])
        idx = np.floor((pts - mins) / sizes).astype(int)
        centroids = mins + (idx + 0.5) * sizes

        frame = pd.DataFrame({cols[0]: centroids[:, 0], cols[1]: centroids[:, 1], cols[2]: centroids[:, 2],
                              '_value': values})
        grouped = frame.groupby(cols, sort=True)['_value']
        processed_data = grouped.agg(['mean', 'count', list]).reset_index()
        processed_data.columns = [*cols, self.dependent_variable, 'counts', 'samples']
        if self.output_data_logarithmised:
            with np.errstate(divide='ignore', invalid='ignore'):
                processed_data[self.dependent_variable] = np.log10(processed_data[self.dependent_variable])
        processed_data = processed_data.dropna().reset_index(drop=True)
        return processed_data

    def visualise_counts(self) -> None:
        """
        Visualise a 3D grid of averaged data.
        Visualises either averages, or data points count, in each box.
        The plot is saved in the output directory.

        Arguments:
            - type: String, either 'value' or 'counts' depending on the desired
                       visualisation.
        """
        file_name = self.output_path + '/counts_grid_plot.png'
        if not os.path.exists(file_name):

            colour_output = self.processed_data['counts']
            title = 'Number of data points within each grid square'
            fig = plt.figure(figsize=(10, 10))

            # Create grid for plot and colour bar.
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 0.2])

            ax = fig.add_subplot(gs[0], projection='3d')
            p = ax.scatter(self.processed_data[self.independent_variables[0]],
                           self.processed_data[self.independent_variables[1]],
                           self.processed_data[self.independent_variables[2]],
                           c=colour_output, s=15, cmap='viridis',
                           vmin=np.percentile(colour_output, 5),
                           vmax=np.percentile(colour_output, 95), alpha=1)
            ax.set_xlabel(self.independent_variables[0], fontsize=15)
            ax.set_ylabel(self.independent_variables[1], fontsize=15)
            ax.set_zlabel(self.independent_variables[2], fontsize=15)
            ax.set_title(title, fontsize=20)
            cbr_ax = fig.add_subplot(gs[1])
            fig.colorbar(p, cax=cbr_ax, orientation='horizontal')

            fig.tight_layout()
            fig.savefig(file_name)
            plt.close()

    def get_sample_histograms(self, n_hists: int = 5) -> None:
        """
        Generate histograms for a random sample of grid squares.
        Histograms are saved in the output directory.

        Arguments:
            - n_hists: Integer, number of histograms to generate.
        """

        samples = self.processed_data.samples.sample(n_hists, random_state=1)

        samples_dir = self.output_path + '/sample_grid_squares'
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        for i in samples.index:
            sample = samples[i]
            if isinstance(sample, str):
                string_elements = sample[1:-1].split(", ")
                # Convert the string elements to floats and create a new tuple.
                sample = [float(element) for element in string_elements]
            plot_path = samples_dir + '/sample_grid_square_' + str(i) + '.png'
            x = self.processed_data[self.independent_variables[0]][i]
            y = self.processed_data[self.independent_variables[1]][i]
            z = self.processed_data[self.independent_variables[2]][i]
            title = str([x, y, z])

            if self.output_data_logarithmised:
                sample_results = np.log10(sample)
            else:
                sample_results = sample

            plt.hist(sample_results)
            plt.xlabel(self.dependent_variable.capitalize())
            plt.ylabel('Count')
            plt.title(title)
            plt.savefig(plot_path)
            plt.close()
