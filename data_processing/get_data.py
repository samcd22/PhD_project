import pandas as pd
import numpy as np
import os

from data_processing.generate_dummy_data import generate_dummy_data
from data_processing.utils import data_param_not_exist
from data_processing.normaliser import Normaliser
from data_processing.box_gridder import BoxGridder


def get_data(data_type = 'dummy', params = {}):
    if data_type == 'dummy':
        if params == {} or params['data_type'] != 'dummy':
            data_param_not_exist()
        return generate_dummy_data(params['sigma'], params['model_select'], noise_dist = params['noise_dist'], 
                            model_params = params['model'], 
                            data_path = params['data_path'], 
                            domain_params= params['domain'], output_header = params['output_header'])

    elif data_type == 'normalised' or data_type == 'gridded':


        if os.path.exists(params['data_path'] + '/normalised_data.csv'):
            data = pd.read_csv(params['data_path'] + '/normalised_data.csv')
        else:
            all_data = pd.read_csv('data/total_data.csv')
            metadata = pd.read_csv('data/data_summary.csv')
            normaliser = Normaliser(all_data, metadata)
            all_experiments = normaliser.get_experiments_list()
            selected_experiments = np.delete(all_experiments, np.where(all_experiments == 'Control'))
            data = normaliser.normalise_data(selected_experiments)
            data.to_csv(params['data_path'] + '/normalised_data.csv')
 
        if data_type == 'gridded':
            grid_size = params['grid_size']
            target = params['target']
            
            box_gridder = BoxGridder(data, grid_size=grid_size, target = target)
            data = box_gridder.get_averages()
            box_gridder.get_sample_histograms(data)
            box_gridder.visualise_average_data(data, 'Concentration')
            box_gridder.visualise_average_data(data, 'Counts')

            
        else:
            data = data[['x', 'y', 'z', params['output_header']]]

        if params['log']:
            data[params['output_header']] = np.log10(data[params['output_header']])

        return data