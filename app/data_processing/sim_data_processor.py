import json
import os
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
# import scipy.stats as stats

from data_processing.data_processor import DataProcessor
from inference_toolbox.model import Model
from matplotlib import pyplot as plt
from numpyencoder import NumpyEncoder
from sklearn.model_selection import train_test_split
from visualisation_toolbox.domain import Domain


class SimDataProcessor(DataProcessor):
    """
    A class to simulate data for the Bayesian Inference tool.
    It's a subclass of DataProcessor.
    This class generates data using an analytical model within a specified domain using the processor parameters.
    The processed data are saved as a csv file.
    The construction (configuration) parameters are saved as a json file.

    Args:
        - processed_data_name (str): The name of the simulated data. This will indicate the name of the folder where the data is saved within the data directory, and the name of the folder where the inference results will be saved in the results directory.
        - model (Model): The model used for data simulation.
        - domain (Domain): The domain used for data simulation.
        - train_test_split (float, optional): Ratio of training to test data. Defaults to 0.8.
        - noise_dist (str, optional): The distribution of noise. Defaults to 'no_noise'. Options are:
            - 'gaussian': Gaussian noise. Takes the noise_level as the standard deviation of the Gaussian distribution.
            - 'no_noise': No noise.
            - Add more noise distributions as needed.
        - noise_level (float, optional): The level of noise of the simulated data. Defaults to None. Can be used interchangeably with noise_percentage.
        - noise_percentage (float, optional): The percentage of noise of the simulated data reletive to the predicted values of the model. Defaults to None. Can be used interchangeably with noise_level.
        - data_path (str, optional): The path to save the simulated data. Defaults to '/data'.
        - plot_data (bool, optional): Whether to plot the simulated data. Defaults to True.

    Attributes:
        - processed_data_name (str): The name of the simulated data. This indicates the name of the folder where the data is saved within the data directory, and the name of the folder where the inference results are saved in the results directory.
        - model (Model): The model used for data simulation.
        - domain (Domain): The domain used for data simulation.
        - train_test_split (float): Ratio of training to test data.
        - noise_dist (str): The distribution of noise. Defaults to 'no_noise'.
        - noise_level (float): The level of noise. Can be used interchangeably with noise_percentage.
        - noise_percentage (float): The percentage of noise reletive to the predicted values of the model. Can be used interchangeably with noise_level.
        - simulated_data (pd.DataFrame): The simulated data.
        - data_path (str): The root data path.
        - plot_data (bool): Whether to plot the simulated

    """

    def __init__(self,
                 processed_data_name: str,
                 model: Model,
                 domain: Domain,
                 train_test_split: float = 0.8,
                 noise_dist='no_noise',
                 noise_level=None,
                 noise_percentage=None,
                 data_path='/data',
                 plot_data=True):

        super().__init__(processed_data_name, None,
                         train_test_split, data_path)
        """
        Initialises the SimDataProcess class.

        Args:
            - processed_data_name (str): The name of the simulated data. This will indicate the name of the folder where the data is saved within the data directory, and the name of the folder where the inference results will be saved in the results directory.
            - model (Model): The model used for data simulation.
            - domain (Domain): The domain used for data simulation.
            - train_test_split (float, optional): Ratio of training to test data. Defaults to 0.8.
            - noise_dist (str, optional): The distribution of noise. Defaults to 'no_noise'. Options are:
                - 'gaussian': Gaussian noise. Takes the noise_level as the standard deviation of the Gaussian distribution.
                - 'no_noise': No noise.
            - noise_level (float, optional): The level of noise. Defaults to None. Can be used interchangeably with noise_percentage.
            - noise_percentage (float, optional): The percentage of noise reletive to the predicted values of the model. Defaults to None. Can be used interchangeably with noise_level.
            - data_path (str, optional): The path to save the simulated data. Defaults to '/PhD_project/data/'.
            - plot_data (bool, optional): Whether to plot the simulated data. Defaults to True.

        """

        self.model = model
        self.domain = domain
        self.noise_dist = noise_dist
        self.noise_level = noise_level
        self.noise_percentage = noise_percentage
        self.plot_data = plot_data

        if self.noise_dist == 'no_noise':
            if self.noise_level is not None or self.noise_percentage is not None:
                raise ValueError('SimDataProcess - noise_level and noise_percentage must be None when noise_dist is "no_noise"')
        else:
            if (self.noise_level is None and self.noise_percentage is None) or (self.noise_level is not None and self.noise_percentage is not None):
                raise ValueError('SimDataProcess - either noise_level or noise_percentage must be specified, but not both, when noise_dist is not "no_noise"')

    def get_construction(self) -> dict:
        """
        Gets the construction parameters.
        The construction parameters includes all of the config information used to simulate the data.
        This checks if the simulated data already exists. It includes:
            - processed_data_name: The name of the processed data.
            - noise_dist: The distribution of noise.
            - noise_level: The level of noise.
            - noise_percentage: The percentage of noise.
            - train_test_split: Ratio of training to test data.
            - model_params: The parameters for the model.
            - domain_params: The parameters for the domain.

        Returns:
            - dict: The construction parameters.
        """
        construction = {
            'processed_data_name': self.processed_data_name,
            'noise_dist': self.noise_dist,
            'noise_level': self.noise_level,
            'noise_percentage': self.noise_percentage,
            'train_test_split': self.train_test_split,
        }
        construction['model_params'] = {}
        construction['domain_params'] = {}
        construction['model_params'] = self.model.get_construction()
        construction['domain_params'] = self.domain.get_construction()

        return construction

    def _check_data_exists(self):
        """
        Check if the simulated data already exists.

        Returns:
            - bool: True if the data exists, False otherwise.

        Raises:
            - FileNotFoundError: If the construction.json file is not found.
            - Exception: Mismatched construction.json vs simulator parameters.
        """

        data_filepath = self.data_path + '/processed_sim_data/' + \
            self.processed_data_name
        if not os.path.exists(data_filepath):
            return False
        else:
            try:
                with open(data_filepath + '/construction.json', 'r') as f:
                    construction_data = json.load(f)

                if construction_data == self.get_construction():
                    return True
                else:
                    raise Exception(
                        'SimDataProcessor - construction.json file under the data_name ' + self.processed_data_name + 'does not match simulator parameters')
            except:
                raise FileNotFoundError(
                    'SimDataProcessor - construction.json file not found')

    def _save_data(self, simulated_data):
        """
        Save simulated data as a csv file.
        Save the construction parameters as a json file.

        Args:
            - simulated_data (pd.DataFrame): The simulated data.
        """

        data_file_path = self.data_path + '/processed_sim_data/' + \
            self.processed_data_name
        if not os.path.exists(data_file_path):
            os.makedirs(data_file_path)
        simulated_data.to_csv(data_file_path + '/data.csv')
        with open(data_file_path + '/construction.json', 'w') as f:
            json.dump(self.get_construction(), f, cls=NumpyEncoder,
                      separators=(', ', ': '), indent=4)
        print('Data generated and saved to ' + data_file_path)

    def process_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate the data based on the given model and domain.
        If data has already been simulated, loads the data.
        Saves the simulated data as a csv file.

        Returns:
            - tuple: A tuple containing the training data and test data.
        """
        points = self.domain.create_domain()
        model_func = self.model.get_model()

        all_points = pd.DataFrame({})
        for indep_var in self.model.independent_variables:
            if points.ndim < 2:
                all_points[indep_var] = points
            else:
                all_points[indep_var] = points[:, self.model.independent_variables.index(indep_var)]
        if not self._check_data_exists():
            mu = model_func(pd.Series({}), all_points)
            if self.noise_percentage is not None:
                vec_noise = self.noise_percentage*mu
            else:
                if self.noise_level is not None:
                    vec_noise = self.noise_level*np.ones(mu.size)
            if self.noise_dist == 'gaussian':
                C = np.array([mu[i] + vec_noise[i]**2*np.random.normal() for i in range(mu.size)])
            elif self.noise_dist == 'no_noise':
                C = mu
            else:
                raise Exception('SimDataProcess - Noise distribution invalid!')

            independent_vars = self.model.independent_variables
            data = pd.DataFrame({})
            for i in range(len(independent_vars)):
                if points.ndim < 2:
                    data[independent_vars[i]] = points
                else:
                    data[independent_vars[i]] = points[:, i]
            data[self.model.dependent_variables[0]] = C
            data[self.model.dependent_variables[0] + '_true'] = mu
            data.dropna(inplace=True)
            self.processed_data = data
            self._save_data(self.processed_data)
        else:
            data_path = self.data_path + '/processed_sim_data/' + \
                self.processed_data_name
            self.processed_data = pd.read_csv(data_path + '/data.csv')
            print('Data loaded from ' + data_path)

        train_data, test_data = train_test_split(
            self.processed_data,
            test_size=1 - self.train_test_split,
            random_state=42)

        if self.plot_data:
            self.plot_sim_data()

        return train_data, test_data

    def plot_sim_data(self):
        """
        This function plots the simulated data.
        The plot is saved as a png file in the processed_sim_data folder.
        The plot is only available for 1D and 3D data.
        """
        file_path = self.data_path + '/processed_sim_data/' + \
            self.processed_data_name
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not os.path.exists(file_path + '/data_plot.png'):
            title = 'Simulated ' + self.model.dependent_variables[0] + ' data'
            if self.domain.n_dims == 1:
                fig, ax = plt.subplots()
                ax.scatter(
                    self.processed_data[self.model.independent_variables[0]],
                    self.processed_data[self.model.dependent_variables[0]],
                    label='Simulated Data', s=10)
                ax.plot(
                    self.processed_data[self.model.independent_variables[0]],
                    self.processed_data[
                        self.model.dependent_variables[0] + '_true'],
                    label='Simulated Data No Noise', color='red')
                ax.set_xlabel(self.model.independent_variables[0])
                ax.set_ylabel(self.model.dependent_variables[0])
                ax.set_title(title)
                ax.legend()
            elif self.domain.n_dims == 2:
                raise ValueError(
                    'SimDataProcessor - 2D data plotting not implemented')
            elif self.domain.n_dims == 3:
                fig = plt.figure(figsize=(7, 8))
                # Create grid for plot and colour bar.
                gs = gridspec.GridSpec(2, 1, height_ratios=[5, 0.2])

                ax = fig.add_subplot(gs[0], projection='3d')
                sc = ax.scatter(
                    self.processed_data[self.model.independent_variables[0]],
                    self.processed_data[self.model.independent_variables[1]],
                    self.processed_data[self.model.independent_variables[2]],
                    c=self.processed_data[self.model.dependent_variables[0]],
                    cmap='viridis', s=10,
                    vmin=np.percentile(
                        self.processed_data[self.model.dependent_variables[0]],
                        5),
                    vmax=np.percentile(
                        self.processed_data[self.model.dependent_variables[0]],
                        95))
                ax.set_xlabel(self.model.independent_variables[0])
                ax.set_ylabel(self.model.independent_variables[1])
                ax.set_zlabel(self.model.independent_variables[2])
                ax.set_title(title)

                # Add colorbar in the second grid row.
                cbar_ax = fig.add_subplot(gs[1])
                fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
                fig.tight_layout()

            else:
                raise ValueError('SimDataProcessor - Data dimensionality not supported')
            fig.savefig(file_path + '/data_plot.png')
            plt.close()
