import numpy as np
import pandas as pd

class Normaliser:
    
    def __init__(self, data, meta_data):
        self.data = data
        self.meta_data = meta_data

    def normalise_data(self, experiments):
        frames = []
        for experiment in experiments:
            experiment_data = self.data[self.data['Experiment'] == experiment].copy()

            experiment_meta_data = self.meta_data[self.meta_data['Experiment'] == experiment]

            wind_dir = np.pi/2 - experiment_meta_data['Wind_Dir'].values[0]*np.pi/180
            wind_speed = experiment_meta_data['WindSpeed'].values[0]
            boat_lat = experiment_meta_data['boat.lat'].values[0]
            boat_lon = experiment_meta_data['boat.lon'].values[0]

            experiment_data['x_diff'] = (experiment_data['gps.lon'] - boat_lon)*np.cos(boat_lat*np.pi/180)*40075000/360
            experiment_data['y_diff'] = (experiment_data['gps.lat'] - boat_lat)*110000
            experiment_data['z'] = experiment_data['altitudeRelative']

            experiment_data['x'] =  -(experiment_data['x_diff']*np.cos(wind_dir) + experiment_data['y_diff']*np.sin(wind_dir))
            experiment_data['y'] = -experiment_data['x_diff']*np.sin(wind_dir) + experiment_data['y_diff']*np.cos(wind_dir)

            experiment_data['Norm_Conc'] = experiment_data['Concentration']*wind_speed*100**3
            experiment_data['Log_Norm_Conc'] = np.log(experiment_data['Norm_Conc'])
            frames.append(experiment_data)

        normalised_data = pd.concat(frames)
        normalised_data = normalised_data[normalised_data.notnull().all(axis=1)]
        
        return normalised_data.reset_index()
    
    def get_experiments_list(self):
        return self.data.Experiment.unique()

    def get_overlapping_data(self, experiments):

        unique_distances = []
        for experiment in experiments:

            experiment_data = self.data[self.data['Experiment'] == experiment]

            unique_distances.append(experiment_data.Distance.unique())

        overlapping_distances = set.intersection(*map(set,unique_distances))

        overlapping_data = {}
        for overlapping_distance in overlapping_distances:

            overlapping_distance_data = self.data[self.data['Distance'] == overlapping_distance]

            unique_heights_for_distance = []
            for experiment in experiments:
                experiment_height_data = overlapping_distance_data[overlapping_distance_data.Experiment == experiment]
                unique_heights_for_distance.append(experiment_height_data.Height.unique())

            overlapping_heights_for_distance = set.intersection(*map(set,unique_heights_for_distance))
            overlapping_data[overlapping_distance] = overlapping_heights_for_distance 

        frames = []
        for distance in overlapping_data.keys():
            distance_data = self.data[self.data.Distance == distance]
            frames.append(distance_data[distance_data.Height.isin(overlapping_data[distance])])

        return pd.concat(frames)