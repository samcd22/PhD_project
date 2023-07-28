import numpy as np
import pandas as pd
import scipy.stats as stats
import os

from inference_toolbox.domain import Domain
from inference_toolbox.model import Model
from data_processing.utils import data_param_not_exist

current_directory = os.getcwd()
if current_directory != '/project/':
    os.chdir('/project/')


def generate_dummy_data(sigma, model_select, noise_dist = "gaussian", model_params = {}, data_path = 'data', 
                        domain_params={
                            'domain_select': 'cone_from_source_z_limited', 
                            'resolution': 20,
                            'domain_params':{
                                'r': 100,
                                'theta': np.pi/8,
                                'source': [0,0,10]}
                            }, output_header = "Concentration"):

    model = Model(model_select)

    if 'model_params' in model_params:
        for param in model_params['model_params'].keys():
            model.add_model_param(param, model_params['model_params'][param])

    model_func = model.get_model()

    inference_params = pd.Series(model_params['inference_params'])

    domain = Domain(domain_params['domain_select'], resolution = domain_params['resolution'])
    if 'domain_params' in domain_params:
        for param in domain_params['domain_params'].keys():
            domain.add_domain_param(param, domain_params['domain_params'][param])

    points = domain.create_domain()


    mu = model_func(inference_params, points[:,0], points[:,1], points[:,2])
    
    if noise_dist == 'gaussian':
        C = np.array([val + sigma*np.random.normal() for val in mu])
    elif noise_dist == 'gamma':
        a = mu**2/sigma**2
        b = mu/sigma**2
        C = np.array([stats.gamma.rvs(a[i], scale = 1/b[i]) for i in range(mu.size)])
    elif noise_dist == 'no_noise':
        C = mu
    else:
        raise Exception('Noise distribution invalid!')

    data = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2], output_header: C})

    data.to_csv(data_path + '/dummy_data.csv')

    return data
