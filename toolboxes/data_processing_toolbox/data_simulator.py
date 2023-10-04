import os
import numpy as np
import json
from numpyencoder import NumpyEncoder
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt

from toolboxes.plotting_toolbox.domain import Domain
from toolboxes.inference_toolbox.model import Model
from toolboxes.data_processing_toolbox.box_gridder import BoxGridder

class DataSimulator:
    def __init__(self, data_params, gridding = None, suppress_prints = False):
        self.data_type = 'simulated_data'

        self.data_params = data_params

        self.model_select = data_params['model']['model_select']
        self.model_params = data_params['model']['model_params']
        self.inference_params = data_params['model']['inference_params']
        
        self.domain_select = data_params['domain']['domain_select']
        self.domain_params = data_params['domain']['domain_params']
        if 'resolution' not in data_params['domain']:
            self.domain_resolution = None
        else:
            self.domain_resolution = data_params['domain']['resolution']

        self.noise_dist = data_params['noise_dist']
        self.noise_level = data_params['noise_level']

        self.output_header = data_params['output_header']

        self.gridding = None
        if 'gridding' in data_params:
            self.gridding = data_params['gridding']

        self.data_path = 'data/' + self.data_type + '/' + self.model_select

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

    def generate_data(self):
        if not os.path.exists(self.instance_path + '/data.csv'):

            model = Model(self.model_select)

            for param in self.model_params.keys():
                model.add_model_param(param, self.model_params[param])

            model_func = model.get_model()

            inference_params = pd.Series(self.inference_params)

            domain = Domain(self.domain_select, resolution = self.domain_resolution)
            for param in self.domain_params.keys():
                domain.add_domain_param(param, self.domain_params[param])

            points = domain.create_domain()

            mu = model_func(inference_params, points[:,0], points[:,1], points[:,2])
        
            if self.noise_dist == 'gaussian':
                C = np.array([val + self.noise_level*np.random.normal() for val in mu])
            elif self.noise_dist == 'gamma':
                a = mu**2/self.noise_level**2
                b = mu/self.noise_level**2
                C = np.array([stats.gamma.rvs(a[i], scale = 1/b[i]) for i in range(mu.size)])
            elif self.noise_dist == 'no_noise':
                C = mu
            else:
                raise Exception('Noise distribution invalid!')

            data = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2], self.output_header: C})
            data.to_csv(self.instance_path + '/data.csv')
        
        else:
            data = pd.read_csv(self.instance_path + '/data.csv')
        
        self.plot_raw_data(data)

        if self.gridding:
            gridding_string = ('_').join([str(x) for x in self.gridding])
            gridding_path = self.instance_path + '/gridding/grid_' + gridding_string
            box_gridder = BoxGridder(data, self.gridding, data_path = gridding_path)
            data = box_gridder.get_averages(input_column_name = self.output_header)
            box_gridder.get_sample_histograms(data)
            box_gridder.visualise_average_data(data, 'Concentration')
            box_gridder.visualise_average_data(data, 'Counts')

        return data
            
    def plot_raw_data(self, data):
        if not os.path.exists(self.instance_path + '/plot.png'):
            fig = plt.figure(figsize = (10,10))
            ax = fig.add_subplot(111, projection = '3d')
            colour_output = data[self.output_header]
            p = ax.scatter(data.x, data.y, data.z, c=colour_output, s = 20, cmap='jet', vmin = np.percentile(colour_output,5), vmax = np.percentile(colour_output,95))

            title = 'Data generated from the ' + self.model_select + ' model'

            ax.set_xlabel('Distance Downwind')
            ax.set_ylabel('Distance Crosswind')
            ax.set_zlabel('Altitude')

            ax.set_title(title)
            fig.colorbar(p)
            plt.tight_layout()
            fig.savefig(self.instance_path + '/plot.png')
            plt.close()
