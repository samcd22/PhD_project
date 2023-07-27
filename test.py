import os
import pandas as pd
import numpy as np

from drivers.sandbox import Sandbox
from drivers.generator import Generator
from drivers.optimiser import Optimiser

from inference_toolbox.domain import Domain
from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood


current_directory = os.getcwd()
if current_directory != '/project/':
    os.chdir('/project/')

data_params = {
    'data_type': 'gridded',
    'output_header': 'Concentration',
    'log':True,
    'grid_size': [100,100,25],
    'target': False,
    'data_path':'data'
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

results_name = 'gridded_data_100_100_25_real'

# ## Sandbox ##
# 
# Create an instance with the inputted default parameters and visualise it in different ways

sandbox = Sandbox(results_name=results_name, 
                  default_params=default_params, 
                  data_params=data_params)

visualiser = sandbox.run()
visualiser.get_summary()
visualiser.get_traceplot()
visualiser.get_autocorrelations()

domain = Domain('cone_from_source_z_limited', resolution=80)
domain.add_domain_param('r', 1000)
domain.add_domain_param('theta', np.pi/8)
domain.add_domain_param('source', [0,0,10])

visualiser.visualise_results(domain = domain, name = 'small_scale_3D_plots', title='Log Concentration of Droplets', log_results=False)
visualiser.animate(name = 'small_scale_3D_plots')

# ## Generator ##
# 
# Generate different instances by varying the hyperparameters with respect to their defaults

generator = Generator(results_name=results_name, 
                  default_params=default_params, 
                  data_params=data_params)

# Vary two hyperparameters

analysis_iterations = {
    'parameters_1':
    [
        'I_y_mu',
        'I_z_mu',
        'Q_mu',
         
    ],
    'parameters_2':
    [
        'I_y_sigma',
        'I_z_sigma',
        'Q_sigma',
    ],
    'values_1':
    [
        np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
        np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
        np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18]),
    ],
    'values_2':
    [
        np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
        np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]),
        np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18]),    
    ]
}

for i in range(len(analysis_iterations['parameters_1'])):
    parameter_1 = analysis_iterations['parameters_1'][i]
    parameter_2 = analysis_iterations['parameters_2'][i]

    print('Working on varying ' + parameter_1 + '...')
    values_1 = analysis_iterations['values_1'][i]
    values_2 = analysis_iterations['values_2'][i]
    inputs = generator.vary_two_parameters(parameter_1, parameter_2, values_1, values_2, scale_1='log', scale_2='log', plot=True)

# ## Optimiser ##
# 
# Optimise the default construction using Bayesian Optimisation

optimising_parameters = {
                    'I_y_mu': [1e-2, 10],
                    'I_y_sigma': [1e-2, 10],
                    'I_z_mu': [1e-2, 10],
                    'I_z_sigma': [1e-2, 10],
                    'Q_mu': [1e9, 1e18],
                    'Q_sigma': [1e9, 1e18],
                    'sigma_mu':[0.2, 2],
                    'sigma_sigma':[0.2, 2]
                }

optimiser = Optimiser(results_name=results_name, 
                  default_params=default_params, 
                  data_params=data_params)

study = optimiser.run(n_trials=100, optimiser_name='AIC_1', optimising_parameters=optimising_parameters, index_name='aic')
optimiser.get_plots(study)

optimiser.run_best_params(study)


