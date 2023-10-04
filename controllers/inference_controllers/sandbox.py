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

# Sandbox class - used for generating one instance of the sampler and visualising its results
class Sandbox(Controller):
    # Initialises the Sandbox class saving all relevant variables and performing some initialising tasks
    def __init__(self, 
                 results_name = 'name_placeholder',
                 default_params = None,
                 data_params = None,
                results_path = 'results/inference_results'):
        
        # Inherits methods and attributes from parent Driver class
        super().__init__(results_name, data_params, default_params,  results_path)

        # Actual parameter values are saved if they are available
        self.actual_values = []
        if self.data_params['data_type'] == 'simulated_data':
            for inference_param in self.data_params['model']['inference_params'].keys():
                self.actual_values.append(self.data_params['model']['inference_params'][inference_param])

        # Generates the construction object
        construction = self.get_data_construction()

        # Initialises the construction
        self.init_data_construction(construction)
        
    # Initialises the construction using the construction object, checking and creating all relevant files and folders
    def init_data_construction(self, construction):
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
                json.dump(construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

    # Creates an instance of the sampler and visualiser, outputting the visualiser
    def run(self):
        # Generates the data
        data = get_data(self.data_params)
        training_data, testing_data = train_test_split(data, test_size=0.2, random_state=1)

        # Assigns the parameter, likelihood, model and sampler specification variables
        params = pd.concat([self.default_params['infered_params']['model_params'], self.default_params['infered_params']['likelihood_params']])
        likelihood = self.default_params['likelihood']
        model = self.default_params['model']
        n_samples = self.default_params['sampler']['n_samples']
        n_chains = self.default_params['sampler']['n_chains']
        thinning_rate = self.default_params['sampler']['thinning_rate']
        
        # Initialises the a sampler object
        sampler = Sampler(params, 
                          model, 
                          likelihood, 
                          training_data, 
                          testing_data,
                          n_samples, 
                          n_chains=n_chains, 
                          thinning_rate=thinning_rate,  
                          data_path = self.full_results_path)
        
        # Runs the sampler for the allotted number of samples
        sampler.sample_all()

        # Initialises the sampler object
        visualiser = Visualiser(testing_data,
                                sampler,
                                model,
                                previous_instance=sampler.instance,
                                data_path = self.full_results_path,
                                actual_values = self.actual_values)
        return visualiser

