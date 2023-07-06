import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from numpyencoder import NumpyEncoder

from drivers.driver import Driver
from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from inference_toolbox.sampler import Sampler
from inference_toolbox.visualiser import Visualiser
from data_processing.get_data import get_data

import warnings

# Filter all warnings
warnings.simplefilter("ignore")

class Sandbox(Driver):
    def __init__(self, 
                 results_name = 'name_placeholder',
                 default_params = {
                    'infered_params':pd.Series({
                        'model_params':pd.Series({
                            'I_y': Parameter('I_y', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'I_z': Parameter('I_z', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'Q': Parameter('Q', prior_select = 'gamma', default_value=3e13).add_prior_param('mu', 3e13).add_prior_param('sigma',1e13),
                        }),
                        'likelihood_params':pd.Series({})
                    }),
                    'model':Model('log_gpm_alt_norm').add_model_param('H',10),
                    'likelihood': Likelihood('gaussian_fixed_sigma').add_likelihood_param('sigma',1),
                    'sampler': {
                        'n_samples': 10000,
                        'n_chains': 3,
                        'thinning_rate': 1
                    }
                },
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
                results_path = 'results',
                show_log = False):
        super().__init__(results_name, default_params, data_params, results_path)
        self.show_log = show_log
        construction = self.get_constriction()
        self.init_construction(construction)
        

    def init_construction(self, construction):

        self.construction_results_path = self.results_path + '/' + self.results_name
        self.full_results_path = self.construction_results_path + '/inference/general_instances'

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

    def run(self):
        data = get_data(self.data_params['data_type'], self.data_params)
        training_data, testing_data = train_test_split(data, test_size=0.2, random_state=1)
        params = pd.concat([self.default_params['infered_params']['model_params'], *self.default_params['infered_params']['likelihood_params']])
        likelihood = self.default_params['likelihood']
        model = self.default_params['model']
        n_samples = self.default_params['sampler']['n_samples']
        n_chains = self.default_params['sampler']['n_chains']
        thinning_rate = self.default_params['sampler']['thinning_rate']
        
        sampler = Sampler(params, 
                          model, 
                          likelihood, 
                          training_data, 
                          testing_data,
                          n_samples, 
                          show_sample_info = self.show_log, 
                          n_chains=n_chains, 
                          thinning_rate=thinning_rate,  
                          data_path = self.full_results_path)

        sampler.sample_all()

        visualiser = Visualiser(testing_data,
                                sampler,
                                model,
                                previous_instance=sampler.instance,
                                data_path = self.full_results_path)
        
        return visualiser

