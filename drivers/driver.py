import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from inference_toolbox.sampler import Sampler
from inference_toolbox.visualiser import Visualiser
from inference_toolbox.domain import Domain

from data_processing.get_data import get_data
import os

current_directory = os.getcwd()
if current_directory != '/project/':
    os.chdir('/project/')

class Driver():
    def __init__(self, results_name, default_params, data_params, results_path):

        self.results_name = results_name
        self.default_params = default_params
        self.data_params = data_params
        self.results_path = results_path
        
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
    
    def init_construction(self, construction):
        print('No driver assigned')