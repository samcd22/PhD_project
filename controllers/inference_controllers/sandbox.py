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
                 data_params = {
                    'data_type': 'dummy',
                    'data_path': 'data',
                    'sigma': 'NaN',
                    'model_select': 'log_gpm_alt_norm',
                    'noise_dist': 'gaussian',
                    'model': {
                        'model_params':{
                            'H': 10
                        },
                        'inference_params':{
                            'I_y': 0.1,
                            'I_z': 0.1,
                            'Q': 3e13
                        },
                    },
                    'domain': {
                        'domain_select': 'cone_from_source_z_limited', 
                        'resolution': 20,
                        'domain_params':{
                            'r': 100,
                            'theta': np.pi/8,
                            'source': [0,0,10]}
                    },
                    'output_header': 'Concentration'
                },
                default_params = {
                    'infered_params':pd.Series({
                        'model_params':pd.Series({
                            'I_y': Parameter('I_y', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'I_z': Parameter('I_z', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'Q': Parameter('Q', prior_select = 'gamma', default_value=3e13).add_prior_param('mu', 3e13).add_prior_param('sigma',1e13),
                        }),
                        'likelihood_params':pd.Series({},dtype='float64')
                    }),
                    'model':Model('log_gpm_alt_norm').add_model_param('H',10),
                    'likelihood': Likelihood('gaussian_fixed_sigma').add_likelihood_param('sigma',1),
                    'sampler': {
                        'n_samples': 10000,
                        'n_chains': 3,
                        'thinning_rate': 1
                    }
                },
                results_path = 'results/inference_results'):
        
        # Inherits methods and attributes from parent Controller class
        super().__init__(results_name, data_params, default_params, results_path)

        # Generates results folder
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Actual parameter values are saved if they are available
        self.actual_values = []
        if self.data_params['data_type'] == 'dummy':
            for inference_param in self.data_params['model']['inference_params'].keys():
                self.actual_values.append(self.data_params['model']['inference_params'][inference_param])

        # Generates the construction object
        construction = self.get_constriction()

        # Initialises the construction
        self.init_construction(construction)
        
    # Initialises the construction using the construction object, checking and creating all relevant files and folders
    def init_construction(self, construction):
        self.construction_results_path = self.results_path + '/' + self.results_name
        self.full_results_path = self.construction_results_path + '/general_instances'

        if not os.path.exists(self.construction_results_path):
            os.makedirs(self.construction_results_path)

        if not os.path.exists(self.full_results_path):
            os.makedirs(self.full_results_path)

        if os.path.exists(self.construction_results_path + '/construction.json'):
            f = open(self.construction_results_path + '/construction.json')
            saved_construction = json.load(f)
            f.close()

            if saved_construction != construction:
                raise Exception('Default generator parameters do not match for this folder name!')
        else:
            with open(self.construction_results_path + '/construction.json', "w") as fp:
                json.dump(construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

    # Creates an instance of the sampler and visualiser, outputting the visualiser
    def run(self):
        # Generates the data
        data = get_data(self.data_params['data_type'], self.data_params)
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

