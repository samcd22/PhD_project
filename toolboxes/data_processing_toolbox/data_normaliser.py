import os
import numpy as np
import json
from numpyencoder import NumpyEncoder
import pandas as pd

from toolboxes.data_processing_toolbox.box_gridder import BoxGridder

class DataNormaliser:
    def __init__(self, data_params, suppress_prints = False):
        self.data_type = 'normalised_data'

        self.data_params = data_params

        self.data_select = data_params['data_select']

        self.output_header = data_params['output_header']

        self.gridding = None
        if 'gridding' in data_params:
            self.gridding = data_params['gridding']

        self.data_path = 'data/' + self.data_type +'/' + self.data_select

        self.suppress_prints = suppress_prints
        self.instance = self.get_instance()
        self.instance_path = self.data_path + '/instance_' + str(self.instance)

    def check_instance_exists(self):
        instance_exists = False
        instance_number = -1
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        x=os.listdir(self.data_path)
        for instance_folder in os.listdir(self.data_path):
            folder_path = self.data_path + '/' + instance_folder
            f = open(folder_path + '/construction.json')
            instance_data_params = json.load(f)
            f.close()
            if self.data_params == instance_data_params:
                instance_exists = True
                instance_number = int(instance_folder.split('_')[1])
        return instance_exists, instance_number

    # Outputs the next available instance number and generates a folder for that instance
    def get_instance(self):
        instance_exists, instance_number = self.check_instance_exists()
        if instance_exists:
            return instance_number
        else:
            instance_folders = os.listdir(self.data_path)
            instances = [int(x.split('_')[1]) for x in instance_folders]
            missing_elements = []
            if len(instances) == 0:
                instance = 1
            else:
                for el in range(1,np.max(instances) + 2):
                    if el not in instances:
                        missing_elements.append(el)
                instance = np.min(missing_elements)

            instance_path = self.data_path + '/instance_' + str(instance)
            if not os.path.exists(instance_path):
                if not self.suppress_prints:
                    print('Creating data instance')
                os.makedirs(instance_path)
            if not os.path.exists(instance_path + '/construction.json'):
                with open(instance_path + '/construction.json', "w") as fp:
                    json.dump(self.data_params,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

            return instance

    def normalise_data(self):
        if not os.path.exists(self.instance_path + '/data.csv'):
            if self.data_params['normaliser_select'] == 'GBR_normaliser':
                data = self.gbr_normaliser()
            else:
                raise Exception('No valid normaliser selected!')
            
            data.to_csv(self.instance_path + '/data.csv')
            self.plot_raw_data(data)
        
        else:
            data = pd.read_csv(self.instance_path + '/data.csv')

        if self.gridding:
            gridding_string = ('_').join([str(x) for x in self.gridding])
            gridding_path = self.instance_path + '/gridding/grid_' + gridding_string
            box_gridder = BoxGridder(data, self.gridding, data_path = gridding_path)
            data = box_gridder.get_averages(input_column_name = self.output_header)
            box_gridder.get_sample_histograms(data)
            box_gridder.visualise_average_data(data, self.data_params['output_header'])
            box_gridder.visualise_average_data(data, 'Counts')

        return data
    
    def gbr_normaliser(self):
        raw_data = pd.read_csv('data/raw_data/' + self.data_params['data_select'] + '.csv')
        meta_data = pd.read_csv('data/raw_data/' + self.data_params['normaliser_params']['meta_data_select'] + '.csv')
        selected_experiments = self.data_params['normaliser_params']['experiments_list']
        
        frames = []
        for experiment in selected_experiments:
            experiment_data = raw_data[raw_data['Experiment'] == experiment].copy()

            experiment_meta_data = meta_data[meta_data['Experiment'] == experiment]

            wind_dir = np.pi/2 - experiment_meta_data['Wind_Dir'].values[0]*np.pi/180
            wind_speed = experiment_meta_data['WindSpeed'].values[0]
            boat_lat = experiment_meta_data['boat.lat'].values[0]
            boat_lon = experiment_meta_data['boat.lon'].values[0]

            experiment_data['x_diff'] = (experiment_data['gps.lon'] - boat_lon)*np.cos(boat_lat*np.pi/180)*40075000/360
            experiment_data['y_diff'] = (experiment_data['gps.lat'] - boat_lat)*110000
            experiment_data['z'] = experiment_data['altitudeRelative']

            experiment_data['x'] =  -(experiment_data['x_diff']*np.cos(wind_dir) + experiment_data['y_diff']*np.sin(wind_dir))
            experiment_data['y'] = -experiment_data['x_diff']*np.sin(wind_dir) + experiment_data['y_diff']*np.cos(wind_dir)

            experiment_data[self.data_params['output_header']] = experiment_data[self.data_params['normaliser_params']['input_header']]*wind_speed*100**3
            
            if self.data_params['log']:
                experiment_data[self.data_params['output_header']] = np.log(experiment_data[self.data_params['output_header']])
            
            frames.append(experiment_data)

        normalised_data = pd.concat(frames)
        normalised_data = normalised_data[normalised_data.notnull().all(axis=1)]
        
        return normalised_data.reset_index()
            
    def plot_raw_data(self, data):
        if not os.path.exists(self.instance_path + '/plot.png'):
            print('Plot data here placeholder')

