from matplotlib import pyplot as plt

# Utility function for traceplots.

def print_all_distances_and_heights(data):
    for experiment in data.Experiment.unique():
        experiment_data = data[data['Experiment'] == experiment]

        selected_data = experiment_data[['Concentration', 'Transect_Num', 'Height', 'Distance', 'Peak_Dist']]
        print('\n',experiment)
        for distance in selected_data.Distance.unique():
            distance_data = selected_data[selected_data.Distance == distance]
            print(distance, distance_data.Height.unique())