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
        - averages_df: DataFrame containing the already averaged data.

    Attributes:
        - data: DataFrame containing the 3D spatial data.
        - grid_size: List containing the grid size in each dimension.
        - output_path: String, path to the data directory.
        - independent_variables: String List of independent variable names.
        - dependent_variable: String, the dependent variable name.
        - input_data_logarithmised: Boolean, indicating logarithmised data.
        - output_data_logarithmised: Boolean, flag output logarithmisation.
        - averages_df: DataFrame containing the averaged data.
        - samples: List of tuples, containing the samples within each box.
    """

    def __init__(self,
                 data: pd.DataFrame = None,
                 grid_size: List[float] = None,
                 output_path: str = None,
                 independent_variables: List[str] = ['x', 'y', 'z'],
                 dependent_variable: str = 'Concentration',
                 input_data_logarithmised: bool = False,
                 output_data_logarithmised: bool = False,
                 averages_df=None) -> None:

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
            - averages_df: DataFrame containing the averaged data.
        """

        self.data = data
        self.grid_size = grid_size
        self.output_path = output_path
        self.independent_variables = independent_variables
        self.dependent_variable = dependent_variable
        self.input_data_logarithmised = input_data_logarithmised
        self.output_data_logarithmised = output_data_logarithmised

        output_path_csv = output_path + '/data.csv'
        if os.path.exists(output_path_csv):
            self.averages_df = pd.read_csv(output_path_csv)
        else:
            os.makedirs(output_path, exist_ok=True)
            if averages_df is not None:
                self.averages_df = averages_df
            else:
                self.averages_df = self._get_averages()
                self.averages_df.to_csv(output_path_csv, index=False)

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

        Returns:
            - DataFrame: averaged data.
        """
        if self.data is None:
            raise Exception('RawDataProcessor - Data not provided')
        if self.grid_size is None:
            raise Exception('RawDataProcessor - Grid size not provided')

        print('Generating gridded averages...')
        min_x = self._flexible_floor(
            np.min(self.data[self.independent_variables[0]]),
            self.grid_size[0])
        min_y = self._flexible_floor(
            np.min(self.data[self.independent_variables[1]]),
            self.grid_size[1])
        min_z = self._flexible_floor(
            np.min(self.data[self.independent_variables[2]]),
            self.grid_size[2])

        max_x = self._flexible_floor(
            np.max(self.data[self.independent_variables[0]]),
            self.grid_size[0]) + self.grid_size[0]
        max_y = self._flexible_floor(
            np.max(self.data[self.independent_variables[1]]),
            self.grid_size[1]) + self.grid_size[1]
        max_z = self._flexible_floor(
            np.max(self.data[self.independent_variables[2]]),
            self.grid_size[2]) + self.grid_size[2]

        # Add x, y, z component to grid.
        grid = (
            np.arange(min_x, max_x + self.grid_size[0], self.grid_size[0]),
            np.arange(min_y, max_y + self.grid_size[1], self.grid_size[1]),
            np.arange(min_z, max_z + self.grid_size[2], self.grid_size[2]))

        grid_len_x = len(grid[0])
        grid_len_y = len(grid[1])
        grid_len_z = len(grid[2])

        indices = []
        centroids = []
        for i in range(grid_len_x - 1):
            for j in range(grid_len_y - 1):
                for k in range(grid_len_z - 1):
                    indices.append([i, j, k])
                    centroids.append([np.mean([grid[0][i], grid[0][i + 1]]),
                                      np.mean([grid[1][j], grid[1][j + 1]]),
                                      np.mean([grid[2][k], grid[2][k + 1]])])

        centroids = np.array(centroids)
        sums = np.zeros((grid_len_x - 1, grid_len_y - 1, grid_len_z - 1))
        counts = np.zeros((grid_len_x - 1, grid_len_y - 1, grid_len_z - 1))
        self.samples = np.empty(
            (grid_len_x - 1, grid_len_y - 1, grid_len_z - 1), object)
        self.samples.fill(())

        for i in range(self.data.shape[0]):
            point = [self.data.loc[i, self.independent_variables[0]],
                     self.data.loc[i, self.independent_variables[1]],
                     self.data.loc[i, self.independent_variables[2]]]
            if self.input_data_logarithmised:
                concentration = 10**self.data.loc[i, self.dependent_variable]
            else:
                concentration = self.data.loc[i, self.dependent_variable]
            y_lower, y_upper = self._get_bounds(grid[1], point[1])
            z_lower, z_upper = self._get_bounds(grid[2], point[2])
            x_lower, x_upper = self._get_bounds(grid[0], point[0])
            box_bounds = np.array([[x_lower, x_upper],
                                   [y_lower, y_upper],
                                   [z_lower, z_upper]])

            # Calculate the data point's position in 3D centroid space.
            # E.g., [1.5, 1.5, 2.5].
            centroid = np.mean(box_bounds, axis=1)

            idx = np.where(np.all(centroid == centroids, axis=1))[0][0]
            sums[indices[idx][0],
                 indices[idx][1],
                 indices[idx][2]] += concentration
            counts[indices[idx][0], indices[idx][1], indices[idx][2]] += 1
            self.samples[indices[idx][0],
                         indices[idx][1],
                         indices[idx][2]] += (concentration,)

            averages = sums/counts
            averages_data = []
            counts_data = []
            samples_data = []
            centroid_x = []
            centroid_y = []
            centroid_z = []
            for i in range(len(indices)):
                averages_data.append(
                    averages[indices[i][0], indices[i][1], indices[i][2]])
                counts_data.append(
                    counts[indices[i][0], indices[i][1], indices[i][2]])
                samples_data.append(
                    list(self.samples[indices[i][0],
                                      indices[i][1],
                                      indices[i][2]]))
                centroid_x.append(centroids[i][0])
                centroid_y.append(centroids[i][1])
                centroid_z.append(centroids[i][2])

            if self.output_data_logarithmised:
                averages_data = np.log10(averages_data)

            averages_df = pd.DataFrame({
                self.independent_variables[0]: centroid_x,
                self.independent_variables[1]: centroid_y,
                self.independent_variables[2]: centroid_z,
                self.dependent_variable: averages_data,
                'counts': counts_data, 'samples': samples_data})
            averages_df = averages_df.dropna()
        return averages_df

    def visualise_average_data(self, type: str = 'values') -> None:
        """
        Visualise a 3D grid of averaged data.
        Visualises either averages, or data points count, in each box.
        The plot is saved in the output directory.

        Arguments:
            - type: String, either 'value' or 'counts' depending on the desired
                       visualisation.
        """
        file_name = self.output_path + '/' + type.lower() + '_grid_plot.png'
        if not os.path.exists(file_name):
            if type == 'values':
                title = self.dependent_variable + ' across all experiments'
                if self.output_data_logarithmised:
                    title = 'Average log ' + title
                else:
                    title = 'Average ' + title

                colour_output = self.averages_df[self.dependent_variable]

            elif type == 'counts':
                colour_output = self.averages_df['counts']
                title = 'Number of data points within each grid square'

            else:
                raise Exception('Type not found')

            fig = plt.figure(figsize=(10, 10))

            # Create grid for plot and colour bar.
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 0.2])

            ax = fig.add_subplot(gs[0], projection='3d')
            p = ax.scatter(self.averages_df[self.independent_variables[0]],
                           self.averages_df[self.independent_variables[1]],
                           self.averages_df[self.independent_variables[2]],
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

        samples = self.averages_df.samples.sample(n_hists, random_state=1)

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
            x = self.averages_df[self.independent_variables[0]][i]
            y = self.averages_df[self.independent_variables[1]][i]
            z = self.averages_df[self.independent_variables[2]][i]
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
