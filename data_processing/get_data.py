import pandas as pd
import numpy as np

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

    elif data_type == 'real':
        print('placeholder')

    elif data_type == 'real_gridded':

        # Import and select data.
        all_data = pd.read_csv('data/total_data.csv')

        # Import and select metadata.
        metadata = pd.read_csv('data/data_summary.csv')
        usecols = ['Experiment', 'Wind_Dir', 'WindSpeed', 'boat.lat', 'boat.lon']

        normaliser = Normaliser(all_data, metadata)
        all_experiments = normaliser.get_experiments_list()
        selected_experiments = np.delete(all_experiments, np.where(all_experiments == 'Control'))
        normalised_data = normaliser.normalise_data(selected_experiments)
        
        box_gridder = BoxGridder(normalised_data)

        data = box_gridder.get_averages([200,200,50], target = False)

        if params['log']:
            data['Concentration'] = np.log10(data['Concentration'])
        return data