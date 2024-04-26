import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from numpyencoder import NumpyEncoder

from controllers.controller import Controller
from toolboxes.inference_toolbox.parameter import Parameter
from toolboxes.inference_toolbox.model import Model
from toolboxes.inference_toolbox.likelihood import Likelihood
from toolboxes.inference_toolbox.sampler import Sampler
from toolboxes.inference_toolbox.visualiser import Visualiser
from toolboxes.data_processing_toolbox.get_data import get_data


class Sandbox(Controller):
    """
    Sandbox class - used for generating one instance of the sampler and visualising its results

    Args:
        results_name (str): Name of the results.
        default_params (dict): Default parameters for the sampler.
        data_params (dict): Parameters for data generation.
        results_path (str): Path to save the results.

    Attributes:
        actual_values (list): List of actual parameter values.
        construction_results_path (str): Path to the construction results.
        full_results_path (str): Path to the full results.
        sampler (Sampler): Instance of the sampler.

    """

    def __init__(self, results_name='name_placeholder', default_params=None, data_params=None,
                 results_path='results/inference_results'):
        super().__init__(results_name, data_params, default_params, results_path)
        self.actual_values = []
        if self.data_params['data_type'] == 'simulated_data':
            for inference_param in self.data_params['model']['inference_params'].keys():
                self.actual_values.append(self.data_params['model']['inference_params'][inference_param])
        construction = self.get_data_construction()
        self.init_data_construction(construction)

    def init_data_construction(self, construction):
        """
        Initializes the construction using the construction object, checking and creating all relevant files and folders.

        Args:
            construction (dict): Construction object.

        Raises:
            Exception: If default generator parameters do not match for this folder name.

        """
        self.construction_results_path = self.results_path + '/' + self.results_name
        self.full_results_path = self.construction_results_path + '/general_instances'

        if not os.path.exists(self.construction_results_path):
            os.makedirs(self.construction_results_path)

        if not os.path.exists(self.full_results_path):
            os.makedirs(self.full_results_path)

        if os.path.exists(self.construction_results_path + '/data_construction.json'):
            f = open(self.construction_results_path + '/data_construction.json')
            saved_construction = json.load(f)
            f.close()

            if saved_construction != construction:
                raise Exception('Default generator parameters do not match for this folder name!')
        else:
            with open(self.construction_results_path + '/data_construction.json', "w") as fp:
                json.dump(construction, fp, cls=NumpyEncoder, separators=(', ', ': '), indent=4)

    def run(self):
        """
        Creates an instance of the sampler and visualiser, outputting the visualiser.

        Returns:
            Visualiser: Instance of the visualiser.

        """
        data = get_data(self.data_params)
        training_data, testing_data = train_test_split(data, test_size=0.2, random_state=1)

        params = pd.concat([self.default_params['infered_params']['model_params'],
                            self.default_params['infered_params']['likelihood_params']])
        likelihood = self.default_params['likelihood']
        model = self.default_params['model']
        n_samples = self.default_params['sampler']['n_samples']
        n_chains = self.default_params['sampler']['n_chains']
        thinning_rate = self.default_params['sampler']['thinning_rate']

        self.sampler = Sampler(params,
                               model,
                               likelihood,
                               training_data,
                               testing_data,
                               n_samples,
                               n_chains=n_chains,
                               thinning_rate=thinning_rate,
                               data_path=self.full_results_path)

        self.sampler.sample_all()

        visualiser = Visualiser(testing_data,
                                self.sampler,
                                model,
                                previous_instance=self.sampler.instance,
                                data_path=self.full_results_path,
                                actual_values=self.actual_values)
        return visualiser

