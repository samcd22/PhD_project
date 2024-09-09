from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bisect
import os
import matplotlib.gridspec as gridspec

class BoxGridder:
    """
    Class to create a 3D grid of boxes and average the data points within each box

    Parameters:
    - data: DataFrame containing the 3D data points
    - grid_size: List containing the size of the grid in each dimension
    - output_path: String, path to the data directory
    - independent_variables: List of strings, names of the independent variables
    - dependent_variable: String, name of the dependent variable
    - input_data_logged: Boolean, indicates if the input data is logged
    - output_data_logged: Boolean, indicates if the results should be logged
    - averages_df: DataFrame containing the already averaged data

    Methods:
    - visualise_average_data: Visualize the 3D grid of the averaged data
    - get_sample_histograms: Generate histograms for a random sample of grid squares
    """
    
    def __init__(self, data: pd.DataFrame=None, grid_size: List[float]=None, 
                 output_path: str = None, independent_variables: List[str] = ['x','y','z'], 
                 dependent_variable: str = 'Concentration', input_data_logged: bool = False, 
                 output_data_logged: bool = False, averages_df = None) -> None:
        
        
        """
        Initialize the BoxGridder object and  generates the averaged data if not already inputted.

        Args:
        - data: DataFrame containing the 3D data points
        - grid_size: Tuple containing the size of the grid in each dimension
        - output_path: String, path to the data directory
        - independent_variables: List of strings, names of the independent variables
        - dependent_variable: String, name of the dependent variable
        - input_data_logged: Boolean, indicates if the input data is logged
        - output_data_logged: Boolean, indicates if the results should be logged
        - averages_df: DataFrame containing the averaged data
        """

        self.data = data
        self.output_path = output_path
        self.grid_size = grid_size
        self.independent_variables = independent_variables
        self.dependent_variable = dependent_variable
        self.input_data_logged = input_data_logged
        self.output_data_logged = output_data_logged

        if os.path.exists(self.output_path + '/data.csv'):
            self.averages_df = pd.read_csv(self.output_path + '/data.csv')
        else:
            os.makedirs(self.output_path, exist_ok=True)
            if averages_df is not None:
                self.averages_df = averages_df
            else:
                self.averages_df = self._get_averages()
                self.averages_df.to_csv(self.output_path + '/data.csv', index=False)
        
        
    def _myround(self, x: float, base: float) -> float:
        """
        Round down to the next base number.

        Args:
        - x: Float, number to be rounded
        - base: Float, base number

        Returns:
        - Float, rounded number
        """
        return base * np.floor(x/base)
    
    def _get_bounds(self, haystack: np.ndarray, needle: float) -> Tuple[float, float]:
        """
        Check which boundary points in the grid the point of interest lies between.

        Args:
        - haystack: Numpy array, grid points
        - needle: Float, point of interest

        Returns:
        - Tuple of floats, lower and upper bounds
        """
        idx = bisect.bisect(haystack, needle)
        if 0 < idx < len(haystack):
            return haystack[idx-1], haystack[idx]
        else:
            raise ValueError(f"BoxGridder - {needle} is out of bounds of {haystack}") 


    def _get_averages(self) -> pd.DataFrame:
        """
        Calculate the averages of the data points within each box.

        Returns:
        - DataFrame, averaged data
        """
        if self.data is None:
            raise Exception('RawDataProcessor - Data not provided')
        if self.grid_size is None:
            raise Exception('RawDataProcessor - Grid size not provided')

        print('Generating gridded averages...')
        min_x = self._myround(np.min(self.data[self.independent_variables[0]]),self.grid_size[0])
        min_y = self._myround(np.min(self.data[self.independent_variables[1]]),self.grid_size[1])
        min_z = self._myround(np.min(self.data[self.independent_variables[2]]),self.grid_size[2])
        max_x = self._myround(np.max(self.data[self.independent_variables[0]]),self.grid_size[0])+self.grid_size[0]
        max_y = self._myround(np.max(self.data[self.independent_variables[1]]),self.grid_size[1])+self.grid_size[1]
        max_z = self._myround(np.max(self.data[self.independent_variables[2]]),self.grid_size[2])+self.grid_size[2]
        grid_x = np.arange(min_x,max_x+self.grid_size[0],self.grid_size[0])
        grid_y = np.arange(min_y,max_y+self.grid_size[1],self.grid_size[1])
        grid_z = np.arange(min_z,max_z+self.grid_size[2],self.grid_size[2])
        grid = (grid_x,grid_y,grid_z)
        indices = []
        centroids = []
        for i in range(len(grid[0])-1):
            for j in range(len(grid[1])-1):
                for k in range(len(grid[2])-1):
                    indices.append([i,j,k])
                    centroids.append([np.mean([grid[0][i],grid[0][i+1]]),np.mean([grid[1][j],grid[1][j+1]]),np.mean([grid[2][k],grid[2][k+1]])])
        
        centroids = np.array(centroids)     

        sums = np.zeros((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1))
        counts = np.zeros((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1))
        self.samples = np.empty((len(grid[0])-1,len(grid[1])-1,len(grid[2])-1), object)
        self.samples.fill(())
        for i in range(self.data.shape[0]):
            point = [self.data.loc[i,self.independent_variables[0]],self.data.loc[i,self.independent_variables[1]],self.data.loc[i,self.independent_variables[2]]]
            if self.input_data_logged:
                conc = 10**self.data.loc[i,self.dependent_variable]
            else:
                conc = self.data.loc[i,self.dependent_variable]
            y_lower, y_upper = self._get_bounds(grid[1], point[1])
            z_lower, z_upper = self._get_bounds(grid[2], point[2])
            x_lower, x_upper = self._get_bounds(grid[0], point[0])
            box_bounds = np.array([[x_lower,x_upper],[y_lower,y_upper],[z_lower,z_upper]])
            centroid = np.mean(box_bounds,axis=1)
            idx = np.where(np.all(centroid == centroids, axis=1))[0][0]
            sums[indices[idx][0],indices[idx][1],indices[idx][2]] += conc
            counts[indices[idx][0],indices[idx][1],indices[idx][2]] += 1
            self.samples[indices[idx][0],indices[idx][1],indices[idx][2]]+=(conc,)
            averages = sums/counts
            averages_data = []
            counts_data = []
            samples_data = []
            centroid_x = []
            centroid_y = []
            centroid_z = []
            for i in range(len(indices)):
                averages_data.append(averages[indices[i][0],indices[i][1],indices[i][2]])
                counts_data.append(counts[indices[i][0],indices[i][1],indices[i][2]])
                samples_data.append(list(self.samples[indices[i][0],indices[i][1],indices[i][2]]))
                centroid_x.append(centroids[i][0])
                centroid_y.append(centroids[i][1])
                centroid_z.append(centroids[i][2])

            if self.output_data_logged:
                averages_data = np.log10(averages_data)

            averages_df = pd.DataFrame({self.independent_variables[0]:centroid_x,
                                        self.independent_variables[1]:centroid_y,
                                        self.independent_variables[2]:centroid_z, 
                                        self.dependent_variable: averages_data, 
                                        'counts': counts_data, 'samples':samples_data})
            averages_df = averages_df.dropna()
        return averages_df
    
    def visualise_average_data(self, type: str = 'values') -> None:
        """
        Visualize the 3D grid of the averaged data.

        Args:
        - averages_df: DataFrame containing the averaged data
        - type: String, either 'value' or 'counts' depending on what you want to visualize
        """
        file_name = self.output_path + '/' + type.lower() + '_grid_plot.png'
        if not os.path.exists(file_name):
            if type == 'values':
                if self.output_data_logged:
                    title = 'Average log ' + self.dependent_variable + ' across all experiments'
                else:
                    title = 'Average ' + self.dependent_variable + ' across all experiments'
                colour_output = self.averages_df[self.dependent_variable]
            elif type == 'counts':
                colour_output = self.averages_df['counts']
                title = 'Number of data points within each grid square'
            else:
                raise Exception('Type not found')
            fig = plt.figure(figsize = (10,10))
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 0.2])  # Create grid for plot and colorbar

            ax = fig.add_subplot(gs[0], projection = '3d')
            p = ax.scatter(self.averages_df[self.independent_variables[0]],
                           self.averages_df[self.independent_variables[1]],
                           self.averages_df[self.independent_variables[2]], 
                           c=colour_output, s = 15, cmap='viridis', vmin = np.percentile(colour_output,5), vmax = np.percentile(colour_output,95),alpha=1)
            ax.set_xlabel(self.independent_variables[0], fontsize=15)
            ax.set_ylabel(self.independent_variables[1], fontsize=15)
            ax.set_zlabel(self.independent_variables[2], fontsize=15)
            ax.set_title(title, fontsize=20)
            cbr_ax = fig.add_subplot(gs[1])
            fig.colorbar(p,cax=cbr_ax,orientation='horizontal')

            fig.tight_layout()
            fig.savefig(file_name)
            plt.close()
    
    def get_sample_histograms(self, n_hists: int = 5) -> None:
        """
        Generate histograms for a random sample of grid squares.

        Args:
        - averages_df: DataFrame containing the averaged data
        - n_hists: Integer, number of histograms to generate
        """

        samples = self.averages_df.samples.sample(n_hists, random_state=1)

        samples_dir = self.output_path + '/sample_grid_squares'
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        for i in samples.index:
            sample = samples[i]
            if type(sample) == str:
                string_elements = sample[1:-1].split(", ")
                # Convert the string elements to floats and create a new tuple
                sample = [float(element) for element in string_elements]
            plot_path = samples_dir + '/sample_grid_square_' + str(i) + '.png'
            x = self.averages_df[self.independent_variables[0]][i]
            y = self.averages_df[self.independent_variables[1]][i]
            z = self.averages_df[self.independent_variables[2]][i]
            title = str([x,y,z])
            if self.output_data_logged:
                sample_results = np.log10(sample)
            else:
                sample_results = sample
            plt.hist(sample_results)
            plt.xlabel(self.dependent_variable.capitalize())
            plt.ylabel('Count')
            plt.title(title)
            plt.savefig(plot_path)
            plt.close()