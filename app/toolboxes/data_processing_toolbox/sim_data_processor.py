import os
import numpy as np
import json
from numpyencoder import NumpyEncoder
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec

from toolboxes.plotting_toolbox.domain import Domain
from toolboxes.inference_toolbox.model import Model
from toolboxes.data_processing_toolbox.data_processor import DataProcessor

class SimDataProcessor(DataProcessor):
    """
    A class for simulating data based on a given model and domain. Inherits from the DataProcessor class.

    Attributes:
    - processed_data_name (str): The name of the simulated data.
    - model (Model): The model used for data simulation.
    - domain (Domain): The domain used for data simulation.
    - processor_params (dict): Additional parameters for the data simulator.
    - train_test_split (float): The ratio of training data to test data
    - noise_dist (str): The distribution of noise. Defaults to 'no_noise'.
    - noise_level (float): The level of noise. Can be used interchangeably with noise_percentage.
    - noise_percentage (float): The percentage of noise reletive to the predicted values of the model. Can be used interchangeably with noise_level.
    - simulated_data (pd.DataFrame): The simulated data.
    - data_path (str): The root data path.

    Methods:
    - process_data: Simulate the data based on the given model and domain. If data has already been simulated, loads the data.
    - get_construction: Get the construction parameters for the data simulator.
    - plot_data: Plot the simulated data.
    """

    def __init__(self, processed_data_name: str, model: Model, domain: Domain, processor_params: dict, train_test_split: float = 0.8, noise_dist = 'no_noise', noise_level = None, noise_percentage = None, data_path = '/PhD_project/data/'):
        super().__init__(processed_data_name, processor_params, train_test_split, data_path)
        """
        Initializes the SimDataProcess class.

        Args:
        - processed_data_name (str): The name of the simulated data.
        - model (Model): The model used for data simulation.
        - domain (Domain): The domain used for data simulation.
        - processor_params (dict): Additional parameters for the data simulator.
        - train_test_split (float, optional): The ratio of training data to test data. Defaults to 0.8.
        - noise_dist (str, optional): The distribution of noise. Defaults to 'no_noise'. Options are:
            - 'gaussian': Gaussian noise. Takes the noise_level as the standard deviation of the Gaussian distribution.
            - 'no_noise': No noise.
        - noise_level (float, optional): The level of noise. Defaults to None. Can be used interchangeably with noise_percentage.
        - noise_percentage (float, optional): The percentage of noise reletive to the predicted values of the model. Defaults to None. Can be used interchangeably with noise_level.
        - data_path (str, optional): The path to save the simulated data. Defaults to '/PhD_project/data/'.

        """
        
        self.model = model
        self.domain = domain
        self.noise_dist = noise_dist
        self.noise_level = noise_level
        self.noise_percentage = noise_percentage

        if self.noise_dist == 'no_noise':
            if self.noise_level is not None or self.noise_percentage is not None:
                raise ValueError('SimDataProcess - noise_level and noise_percentage must be None when noise_dist is "no_noise"')
        else:
            if (self.noise_level is None and self.noise_percentage is None) or (self.noise_level is not None and self.noise_percentage is not None):
                raise ValueError('SimDataProcess - either noise_level or noise_percentage must be specified, but not both, when noise_dist is not "no_noise"')
        
    def get_construction(self):
        """
        Get the construction parameters for the data simulator.

        Returns:
            dict: The construction parameters.
        """
        construction = {
            'processed_data_name': self.processed_data_name,
            'processor_params': self.processor_params,
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
            bool: True if the data exists, False otherwise.
        
        Raises:
            FileNotFoundError: If the construction.json file is not found.
            Exception: If the construction.json file does not match the simulator parameters.
        """
        if not os.path.exists(self.data_path + '/processed_sim_data/' + self.processed_data_name):
            return False
        else:
            try:
                with open(self.data_path + 'processed_sim_data/' + self.processed_data_name + '/construction.json', 'r') as f:
                    construction_data = json.load(f)

                if construction_data == self.get_construction():
                    return True
                else:
                    raise Exception('SimDataProcessor - construction.json file under the data_name ' + self.processed_data_name + 'does not match simulator parameters')
            except:
                raise FileNotFoundError('SimDataProcessor - construction.json file not found')

    def _save_data(self, simulated_data):
        """
        Save the simulated data. Saves the data as a csv file and the construction parameters as a json file.

        Args:
            simulated_data (pd.DataFrame): The simulated data.
        """
        if not os.path.exists(self.data_path + '/processed_sim_data/' + self.processed_data_name):
            os.makedirs(self.data_path + '/processed_sim_data/' + self.processed_data_name)
        simulated_data.to_csv(self.data_path + '/processed_sim_data/' + self.processed_data_name + '/data.csv')
        with open(self.data_path + '/processed_sim_data/' + self.processed_data_name + '/construction.json', 'w') as f:
            json.dump(self.get_construction(), f, cls=NumpyEncoder, separators=(', ',': '), indent=4)

    def process_data(self, plot = True):
        """
        Simulate the data based on the given model and domain. If data has already been simulated, loads the data. Saves the simulated data as a csv file.

        Args:
        - plot (bool, optional): Whether to plot the simulated data. Defaults to True.

        Returns:
        - tuple: A tuple containing the training data and test data.
        """
        points = self.domain.create_domain()
        model_func = self.model.get_model()
        
        all_points = pd.DataFrame({})
        for indep_var in self.model.independent_variables:
            if points.ndim <2:
                all_points[indep_var] = points
            else:
                all_points[indep_var] = points[:,self.model.independent_variables.index(indep_var)]
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
            
            if self.processor_params['log_output_data']:
                C = np.log10(C)
                mu = np.log10(mu)

            independent_vars = self.model.independent_variables
            data = pd.DataFrame({})
            for i in range(len(independent_vars)):
                if points.ndim < 2:
                    data[independent_vars[i]] = points
                else:
                    data[independent_vars[i]] = points[:,i]
            data[self.processor_params['output_header']] = C
            data[self.processor_params['output_header'] + '_true'] = mu
            data.dropna(inplace=True)
            self.processed_data = data
            self._save_data(self.processed_data)
        else:
            print('Data already simulated, loading data...')
            self.processed_data = pd.read_csv(self.data_path + '/processed_sim_data/' + self.processed_data_name + '/data.csv')

        train_data, test_data = train_test_split(self.processed_data, test_size=1-self.train_test_split, random_state=42)
        if plot:
            self.plot_data()

        return train_data, test_data
    
    def plot_data(self):
        """
        Plot the simulated data.
        """
        if not os.path.exists(self.data_path + '/processed_sim_data/' + self.processed_data_name):
            os.makedirs(self.data_path + '/processed_sim_data/' + self.processed_data_name)
        if not os.path.exists(self.data_path + '/processed_sim_data/' + self.processed_data_name + '/data_plot.png'):
            if self.processor_params['log_output_data']:
                title = 'Simulated log10(' + self.processor_params['output_header'] + ') data'
            else:
                title = 'Simulated '  + self.processor_params['output_header'] + ' data'
            if self.domain.n_dims == 1:
                fig, ax = plt.subplots()
                ax.scatter(self.processed_data[self.model.independent_variables[0]], self.processed_data[self.processor_params['output_header']], label='Simulated Data', s=10)
                ax.plot(self.processed_data[self.model.independent_variables[0]], self.processed_data[self.processor_params['output_header'] + '_true'], label='Simulated Data No Noise', color='red')
                ax.set_xlabel(self.model.independent_variables[0])
                ax.set_ylabel(self.processor_params['output_header'])
                ax.set_title(title)
                ax.legend()
            elif self.domain.n_dims == 2:
                raise ValueError('SimDataProcessor - 2D data plotting not implemented')
            elif self.domain.n_dims == 3:
                fig = plt.figure(figsize=(7, 8))
                gs = gridspec.GridSpec(2, 1, height_ratios=[5, 0.2])  # Create grid for plot and colorbar

                ax = fig.add_subplot(gs[0], projection='3d')
                sc = ax.scatter(self.processed_data[self.model.independent_variables[0]], 
                                self.processed_data[self.model.independent_variables[1]], 
                                self.processed_data[self.model.independent_variables[2]], 
                                c=self.processed_data[self.processor_params['output_header']], 
                                cmap='viridis', s=10,
                                vmin = np.percentile(self.processed_data[self.processor_params['output_header']],5),
                                vmax = np.percentile(self.processed_data[self.processor_params['output_header']],95))
                ax.set_xlabel(self.model.independent_variables[0])
                ax.set_ylabel(self.model.independent_variables[1])
                ax.set_zlabel(self.model.independent_variables[2])
                ax.set_title(title)

                # Add colorbar in the second grid row
                cbar_ax = fig.add_subplot(gs[1])
                fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
                fig.tight_layout()

                # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
                # sc = ax.scatter(self.processed_data[self.model.independent_variables[0]], self.processed_data[self.model.independent_variables[1]], self.processed_data[self.model.independent_variables[2]], c=self.processed_data[self.processor_params['output_header']], cmap='viridis', s=10)
                # ax.set_xlabel(self.model.independent_variables[0])
                # ax.set_ylabel(self.model.independent_variables[1])
                # ax.set_zlabel(self.model.independent_variables[2])
                # ax.set_title(title)
                # fig.colorbar(sc, ax=ax, label=self.processor_params['output_header'], orientation='horizontal', shrink=0.6, pad = 0.1)
            else:
                raise ValueError('SimDataProcessor - Data dimensionality not supported')
            fig.savefig(self.data_path + '/processed_sim_data/' + self.processed_data_name + '/data_plot.png')
            plt.close()