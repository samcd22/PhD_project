import os
import pandas as pd
import numpy as np

from controllers.inference_controllers.sandbox import Sandbox
from controllers.inference_controllers.generator import Generator
from controllers.inference_controllers.optimiser import Optimiser

from toolboxes.plotting_toolbox.domain import Domain
from toolboxes.inference_toolbox.parameter import Parameter
from toolboxes.inference_toolbox.model import Model
from toolboxes.inference_toolbox.likelihood import Likelihood

current_directory = os.getcwd()
if current_directory != '/project/':
    os.chdir('/project/')

data_params = {
    'data_type': 'dummy',
    'data_path': 'data',
    'sigma': 1,
    'model_select': 'log_gpm_alt_norm',
    'noise_dist': 'gaussian',
    'model': {
        'model_params':{
            'H': 10
        },
        'inference_params':{
            'I_y': 0.1,
            'I_z': 0.1,
            'Q': 3e13,
            'sigma': 1
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
}


default_params = {
    'infered_params':pd.Series({
        'model_params':pd.Series({
            'I_y': Parameter('I_y', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
            'I_z': Parameter('I_z', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
            'Q': Parameter('Q', prior_select = 'gamma', default_value=3e13).add_prior_param('mu', 3e13).add_prior_param('sigma',1e13),
        }),
        'likelihood_params':pd.Series({
            'sigma': Parameter('sigma', prior_select = 'gamma', default_value=1).add_prior_param('mu', 1).add_prior_param('sigma',1)
        })
    }),
    'model':Model('log_gpm_alt_norm').add_model_param('H',10),
    'likelihood': Likelihood('gaussian'),
    'sampler': {
        'n_samples': 10000,
        'n_chains': 3,
        'thinning_rate': 1
    }
}

results_name = 'test'


generator = Generator(results_name=results_name, 
                  data_params=data_params,
                  default_params=default_params)


analysis_iterations = {
    'parameters_1':
    [
        'I_y_mu',
    ],
    'parameters_2':
    [
        'I_y_sigma',
    ],
    'values_1':
    [
        np.array([1e0, 10])
    ],
    'values_2':
    [
        np.array([1e0])
    ]
}

for i in range(len(analysis_iterations['parameters_1'])):
    parameter_1 = analysis_iterations['parameters_1'][i]
    parameter_2 = analysis_iterations['parameters_2'][i]

    print('Working on varying ' + parameter_1 + '...')
    values_1 = analysis_iterations['values_1'][i]
    values_2 = analysis_iterations['values_2'][i]
    inputs = generator.vary_two_parameters(parameter_1, parameter_2, values_1, values_2, scale_1='log', scale_2='log', plot=True)


