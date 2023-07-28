import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from inference_toolbox.sampler import Sampler
from inference_toolbox.visualiser import Visualiser
from inference_toolbox.domain import Domain
from data_processing.get_data import get_data

# Driver class - parent class of Sandbox, Generator and Optimiser
class Driver():
    # Initialises the Driver class saving all relevant variables
    def __init__(self, results_name, default_params, data_params, results_path):
        self.results_name = results_name
        self.default_params = default_params
        self.data_params = data_params
        self.results_path = results_path
        
    # Generates a conctruction object which includes all info on how the system has been constructed, including data generation and all hyperparameters
    def get_constriction(self):
        return {
            'infered_params':{
                'model_params':{
                    param_name: {
                        'prior_func': self.default_params['infered_params']['model_params'][param_name].prior_select,
                        'prior_params': {
                            prior_param_name: self.default_params['infered_params']['model_params'][param_name].prior_params[prior_param_name] for prior_param_name in self.default_params['infered_params']['model_params'][param_name].prior_params.index
                        },
                    } for param_name in self.default_params['infered_params']['model_params'].keys()
                },
                'likelihood_params':{
                    param_name: {
                        'prior_func': self.default_params['infered_params']['likelihood_params'][param_name].prior_select,
                        'prior_params': {
                            prior_param_name: self.default_params['infered_params']['likelihood_params'][param_name].prior_params[prior_param_name] for prior_param_name in self.default_params['infered_params']['likelihood_params'][param_name].prior_params.index
                        }
                    } for param_name in self.default_params['infered_params']['likelihood_params'].keys()
                }
            },
            'model':{
                'model_type': self.default_params['model'].model_select,
                'model_params': {
                    model_param_name: self.default_params['model'].model_params[model_param_name] for model_param_name in self.default_params['model'].model_params.index
                }            
            },
            'likelihood':{
                'likelihood_type': self.default_params['likelihood'].likelihood_select,
                'likelihood_params': {
                    likelihood_param_name: self.default_params['likelihood'].likelihood_params[likelihood_param_name] for likelihood_param_name in self.default_params['likelihood'].likelihood_params.index
                }
            },
            'sampler': self.default_params['sampler'],
            'data': self.data_params
        }
    
    # Placeholder for initialising the construction
    def init_construction(self, construction):
        print('No driver assigned')