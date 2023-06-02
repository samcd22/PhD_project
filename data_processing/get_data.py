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
            if target:
                target_string = 'target'
            else:
                target_string = 'real'
            file_path = params['data_path'] + '/gridded_' + ('_').join(str(x) for x in grid_size) + '_' + target_string +'.csv'
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
            else:
                box_gridder = BoxGridder(data)
                data = box_gridder.get_averages(grid_size, target = target)
                data.to_csv(file_path)
        else:
            data = data[['x', 'y', 'z', params['output_header']]]
            # [200,200,50]

        if params['log']:
            data[params['output_header']] = np.log10(data[params['output_header']])

        return data