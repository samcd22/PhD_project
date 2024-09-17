import os
import numpy as np
import json
from numpyencoder import NumpyEncoder
import pandas as pd

from data_processing.box_gridder import BoxGridder
from data_processing.data_processor import DataProcessor
from sklearn.model_selection import train_test_split

class RawDataProcessor(DataProcessor):
    """
    A class for processing raw data into a form that is usable for the Bayesian Inference tool. This is a subclass of DataProcessor. This class takes raw data and processes it using a specified data processor, using the processor parameters. The processed data is then saved as a csv file and the construction parameters (the config) are saved as a json file.

    Args:
        - raw_data_filename (str): The filename of the raw data. (Just the name, not the path). 
        - processed_data_name (str): The name of the processed data. This will indicate the name of the folder where the data is saved within the data directory, and the name of the folder where the inference results are saved in the results directory.
        - processor_select (str): The type of data processor to use. Options are:
            - 'GBR_processor': Process the data using the GBR processor. This processor also includes a plotting function.
            - A custom data processor function. This function must not take inputs and return a tuple containing two DataFrames (the training and test data).
        - processor_params (dict): The parameters for the data processor. Typically includes (for the GBR processor):
            - 'output_header' (str): The header for the output data.
            - 'log_output_data' (bool): Whether to take the log of the output data.
            - 'gridding' (int): The size of the grid for the GBR processor.
            - 'meta_data_select' (str): The filename of the meta data.
            - 'experiments_list' (list): A list of experiments to process.
        - train_test_split (float, optional): The ratio of training data to test data. Defaults to 0.8.
        - data_path (str, optional): The path to the data directory. Defaults to 'data'.
        - plot_data (bool, optional): Whether to plot the processed data. Defaults to True.
        - plotting_func (callable, optional): A custom plotting function. Defaults to None. Not required if using the GBR processor, which has its own plotting function.

    Attributes:
        - raw_data_filename (str): The filename of the raw data.
        - processed_data_name (str): The name of the processed data. This indicates the name of the folder where the data is saved within the data directory, and the name of the folder where the inference results are saved in the results directory.
        - processor_select (str, callable): The type of data processor to use. Can be a callable function or one of the predefined processor names.
        - processor_params (dict): The parameters for the data processor.
        - train_test_split (float): The ratio of training data to test data
        - data_path (str): The root data path.
        - processed_data (pd.DataFrame): The processed data.
        - plot_data (bool): Whether to plot the processed data.
        - processor_func (callable): The data processor function.
        - plotting_func (callable): The plotting function.
        
    """

    def __init__(self, raw_data_filename: str, processed_data_name: str, processor_select: str, processor_params: dict, train_test_split: float = 0.8, data_path = '/data', plot_data = True, plotting_func = None):
        super().__init__(processed_data_name, processor_params, train_test_split, data_path)
        """
        Initializes the RawDataProcessor class.

        Args:
            - raw_data_filename (str): The filename of the raw data. (Just the name, not the path)
            - processed_data_name (str): The name of the processed data. This processor also includes a plotting function.
            - processor_select (str, callable): The type of data processor to use. Options are:
                - 'GBR_processor': Process the data using the GBR processor.
                - A custom processor function. This function must not take inputs and return a tuple containing two DataFrames (the training and test data).
            - processor_params (dict): The parameters for the data processor.
            - train_test_split (float, optional): The ratio of training data to test data. Defaults to 0.8.
            - data_path (str, optional): The path to the data directory. Defaults to 'data'.
            - plot_data (bool, optional): Whether to plot the processed data. Defaults to True.
            - plotting_func (callable, optional): A custom plotting function. Defaults to None. Not required if using the GBR processor, which has its own plotting function.
        """
        
        self.raw_data_filename = raw_data_filename
        self.processor_select = processor_select
        self.processor_func = None
        if processor_select == 'GBR_processor':
            self.processor_func = self._GBR_processor
            self.plotting_func = self._plot_GBR_data
        elif callable(processor_select):
            self.processor_func = processor_select
            self.plotting_func = plotting_func
        else:
            raise ValueError('RawDataProcessor - Invalid processor_select. Must be a callable function or one of the predefined processor names.')

        self.data_path = data_path
        self.plot_data = plot_data
       
    def get_construction(self) -> dict:
        """
        Gets the construction parameters. The conctruction parameters includes all of the config information used to process the raw data. This is used to check if the processed data already exists. It includes:
            - raw_data_filename: The filename of the raw data.
            - processed_data_name: The name of the processed data.
            - processor_select: The type of data processor to use.
            - processor_params: The parameters for the data processor (e.g. gridding, logging the output data, etc).
            - train_test_split: The ratio of training data to test data.

        Returns:
            - dict: The construction parameters.
        """
        construction = {
            'raw_data_filename': self.raw_data_filename,
            'processed_data_name': self.processed_data_name,
            'processor_select': self.processor_select,
            'processor_params': self.processor_params,
            'train_test_split': self.train_test_split,
        }
        return construction

    def process_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This is the main function of the RawDataProcessor class that processes the raw data using the specified data processor and parameters. If data has already been processed, this function loads the data. If not, it is processed and saved. The processed data is then split into training and test data using the train_test_split parameter and returned. The processed data is then optionally plotted using the plotting function.

        Returns:
            - tuple: A tuple containing two DataFrames, training data and test data.
        """
        if not self._check_data_exists():
            try:
                result = self.processor_func()
                if not isinstance(result, tuple):
                    raise Exception("RawDataProcessor - Processor did not return a tuple")
                
                if len(result) != 2:
                    raise Exception("RawDataProcessor - Processed data tuple does not contain exactly two elements")
                
                if not isinstance(result[0], pd.DataFrame) or not isinstance(result[1], pd.DataFrame):
                    raise Exception("RawDataProcessor - One or both elements of the tuple are not DataFrames")
            
            except Exception as e:
                raise Exception("RawDataProcessor - Error when calling the inputted processor: {e}")
            
            self._save_data(self.processed_data)
        else:
            print('Data loaded from ' + 'data/processed_raw_data/' + self.processed_data_name)
            self.processed_data = pd.read_csv('data/processed_raw_data/' + self.processed_data_name + '/data.csv')

        if self.plot_data:
            if self.plotting_func is not None:
                self.plotting_func(self.processed_data)
            else:
                raise Exception('RawDataProcessor - Invalid data processor selected!')

        train_data, test_data = train_test_split(self.processed_data, test_size=1-self.train_test_split, random_state=42)
        return train_data, test_data

    def _check_data_exists(self):
        """
        Check if the processed data already exists.

        Returns:
            - bool: True if the data exists, False otherwise.
        
        """
        if not os.path.exists('data/processed_raw_data/' + self.processed_data_name):
            return False
        else:
            try:
                with open('data/processed_raw_data/' + self.processed_data_name + '/construction.json', 'r') as f:
                    construction_data = json.load(f)

                if construction_data == self.get_construction():
                    return True
                else:
                    raise Exception('RawDataProcessor - construction.json file under the data_name ' + self.processed_data_name + 'does not match processor parameters')
            except:
                raise FileNotFoundError('RawDataProcessor - construction.json file not found')

    def _save_data(self, processed_data):
        """
        Save the processed data. Saves the data as a csv file and the construction parameters as a json file.

        Args:
            - processed_data (pd.DataFrame): The processed data.
        """
        if not os.path.exists('data/processed_raw_data/' + self.processed_data_name):
            os.makedirs('data/processed_raw_data/' + self.processed_data_name)
        processed_data.to_csv('data/processed_raw_data/' + self.processed_data_name + '/data.csv')
        with open('data/processed_raw_data/' + self.processed_data_name + '/construction.json', 'w') as f:
            json.dump(self.get_construction(), f, cls=NumpyEncoder, separators=(', ',': '), indent=4)

        print('Data saved to ' + 'data/processed_raw_data/' + self.processed_data_name)

    def _plot_GBR_data(self, processed_data):
        """
        Plot the processed GBR data.
        
        Args:
            - processed_data (pd.DataFrame): The processed data.
        """
        box_gridder = BoxGridder(output_path='data/processed_raw_data/' + self.processed_data_name, averages_df=processed_data)
        box_gridder.get_sample_histograms()
        box_gridder.visualise_average_data('values')
        box_gridder.visualise_average_data('counts')


    def _GBR_processor(self):
        """
        Process the data using the GBR processor.

        Returns:
            - pd.DataFrame: The processed data.
        
        Raises:
            - FileNotFoundError: If the raw data file or meta data file is not found.
        """

        try:
            raw_data = pd.read_csv(self.data_path + '/raw_data/' + self.raw_data_filename + '.csv')
        except:
            raise FileNotFoundError('RawDataProcessor - raw data file not found')

        try:
            meta_data = pd.read_csv(self.data_path + '/raw_data/' + self.processor_params['meta_data_select'] + '.csv')
            selected_experiments = self.processor_params['experiments_list']
        except:
            raise FileNotFoundError('RawDataProcessor - invalid data processor parameters')

        # Process GBR data here
        selected_experiments = self.processor_params['experiments_list']
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

            experiment_data[self.processor_params['output_header']] = experiment_data[self.processor_params['input_header']]*wind_speed*100**3

            if self.processor_params['log_output_data']:
                experiment_data[self.processor_params['output_header']] = np.log10(experiment_data[self.processor_params['output_header']])

            frames.append(experiment_data)

        normalised_data = pd.concat(frames)
        normalised_data = normalised_data[normalised_data.notnull().all(axis=1)]

        normalised_data = normalised_data.reset_index()

        if self.processor_params['gridding']:
            box_gridder = BoxGridder(normalised_data, grid_size=self.processor_params['gridding'], 
                                     output_path='data/processed_raw_data/' + self.processed_data_name, 
                                     input_data_logged=self.processor_params['log_output_data'], 
                                     output_data_logged=self.processor_params['log_output_data'])
            processed_data = box_gridder.averages_df
        else:
            processed_data = normalised_data
        
        processed_data = processed_data[processed_data['x'] > 150]

        return processed_data
    # ADD OTHER PROCESSORS HERE!