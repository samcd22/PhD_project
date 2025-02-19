import json
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt  
import matplotlib.gridspec as gridspec
from numpyencoder import NumpyEncoder
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim

from data_processing.box_gridder import BoxGridder
from data_processing.data_processor import DataProcessor

class RawDataProcessor(DataProcessor):
    """
    A class to preprocess raw data for the Bayesian Inference tool.
    It's a subclass of DataProcessor.
    Raw data are processed via a specified processor and its parameters.
    The processed data are saved as a csv file.
    The construction (configuration) parameters are saved as a json file.

    Args:
        - raw_data_filename (str): Filename (not path) of the raw data.
        - processed_data_name (str): The name of the processed data. This will indicate the name of the folder where the data is saved within the data directory, and the name of the folder where the inference results are saved in the results directory.
        - processor_select (str): Type of data processor to use.
            Options are:
            - 'GBR_processor': The Great Barrier Reef processor.
            - A custom data processor function. This function must not take inputs and return a tuple containing two DataFrames (the training and test data).
        - processor_params (dict): The parameters for the data processor. Typically includes (for the Great Barrier Reef processor):
            - 'output_header' (str): The header for the output data.
            - 'log_output_data' (bool): Whether to take the log of the output data.
            - 'gridding' (int): The size of the grid for the Great Barrier Reef processor.
            - 'meta_data_select' (str): The filename of the meta data.
            - 'experiments_list' (list): A list of experiments to process.
        - train_test_split (float, optional): Ratio of training to test data. Defaults to 0.8.
        - data_path (str, optional): The path to the data directory. Defaults to 'data'.
        - plot_data (bool, optional): Whether to plot the processed data. Defaults to True.
        - plotting_func (callable, optional): A custom plotting function. Defaults to None. Not required if using the Great Barrier Reef processor, which has its own plotting function.

    Attributes:
        - raw_data_filename (str): Filename (not path) of the raw data.
        - processed_data_name (str): The name of the processed data. This indicates the name of the folder where the data is saved within the data directory, and the name of the folder where the inference results are saved in the results directory.
        - processor_select (str, callable): Type of data processor to use. Can be a callable function or one of the predefined processor names.
        - processor_params (dict): The parameters for the data processor.
        - train_test_split (float): Ratio of training to test data.
        - data_path (str): The root data path.
        - processed_data (pd.DataFrame): The processed data.
        - plot_data (bool): Whether to plot the processed data.
        - processor_func (callable): The data processor function.
        - plotting_func (callable): The plotting function.
        - dependent_variable (str): The dependent variable.
        - independent_variables (list): The independent variables.
    """

    def __init__(self,
                 raw_data_filename: str,
                 processed_data_name: str,
                 processor_select: str,
                 processor_params: dict,
                 train_test_split: float = 0.8,
                 data_path='/data',
                 plot_data=True,
                 plotting_func=None,
                 slices = None):

        super().__init__(processed_data_name, processor_params,
                         train_test_split, data_path)
        """
        Initialises the RawDataProcessor class.

        Args:
            - raw_data_filename (str): Filename (not path) of the raw data.
            - processed_data_name (str): The name of the processed data.
            - processor_select (str, callable): Type of data processor to use.
                Options are:
                - 'GBR_processor': The Great Barrier Reef processor.
                                   This processor includes a plotting function.
                - 'XYLO_processor': The XYLO processor.
                - A custom processor function. This function must not take inputs and return a tuple containing two DataFrames (the training and test data).
            - processor_params (dict): The parameters for the data processor.
            - train_test_split (float, optional): Ratio of training to test data. Defaults to 0.8.
            - data_path (str, optional): The path to the data directory. Defaults to 'data'.
            - plot_data (bool, optional): Whether to plot the processed data. Defaults to True.
            - plotting_func (callable, optional): A custom plotting function. Defaults to None. Not required if using the Great Barrier Reef processor, which has its own plotting function.
            - slices (dict, optional): A dictionary of slices to plot. Defaults to None.
        """

        self.raw_data_filename = raw_data_filename
        self.processor_select = processor_select
        self.processor_func = None
        if processor_select == 'GBR_processor':
            self.processor_func = self._GBR_processor
        elif processor_select == 'XYLO_processor':
            self.processor_func = self._xylo_processor
        elif callable(processor_select):
            self.processor_func = processor_select
            self.plotting_func = plotting_func
        else:
            raise ValueError('RawDataProcessor - Invalid processor_select. Must be a callable function or one of the predefined processor names.')

        self.plot_data = plot_data
        self.slices = slices
        self.dependent_variable = None
        self.independent_variables = None

    def get_construction(self) -> dict:
        """
        Gets the construction parameters.
        The construction parameters includes all of the config information used to process the raw data.
        This checks if the processed data already exists. It includes:
        - raw_data_filename: Filename of the raw data.
        - processed_data_name: The name of the processed data.
        - processor_select: Type of data processor to use.
        - processor_params: The parameters for the data processor (e.g. gridding, logging the output data, etc).
        - train_test_split: Ratio of training to test data.

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
        This is the main function of the RawDataProcessor class that processes the raw data using the specified data processor and parameters.
        If data has already been processed, this function loads the data.
        If not, it is processed and saved.
        The processed data is then split into training and test data using the train_test_split parameter and returned.
        The processed data is then optionally plotted using the plotting function.

        Returns:
            - tuple: A tuple containing two DataFrames, training data and test data.
        """
        if not self._check_data_exists():
            self.processed_data = self.processor_func()

            self._save_data(self.processed_data)
        else:
            data_path = 'data/processed_raw_data/' + self.processed_data_name
            print('Data loaded from ' + data_path)
            self.processed_data = pd.read_csv(data_path + '/data.csv')
            if 'Unnamed: 0' in self.processed_data.columns:
                self.processed_data = self.processed_data.drop(columns=['Unnamed: 0'])

        self.dependent_variable = self.processor_params['output_header']
        self.independent_variables = [col for col in self.processed_data.columns if col != self.dependent_variable]

        if self.plot_data:
            self.plot_data_values()

        train_data, test_data = train_test_split(
            self.processed_data,
            test_size=1 - self.train_test_split,
            random_state=42)

        return train_data, test_data

    def plot_data_values(self):
        if self.plotting_func is not None:
            self.plotting_func(self.processed_data)
        else:
            if len(self.independent_variables) == 3:
                self._plot_3D_data()
                if self.slices:
                    for slice_axis, slice_value in self.slices.items():
                        self._plot_2D_data_slice(self.processed_data, slice_axis, slice_value)
            elif len(self.independent_variables) == 2:
                self._plot_2D_data(self.processed_data)
            elif len(self.independent_variables) == 1:
                self._plot_1D_data(self.processed_data)
            else:
                raise Exception('RawDataProcessor - Invalid number of independent variables')

    def _check_data_exists(self):
        """
        Check if the processed data already exists.

        Returns:
            - bool: True if the data exists, False otherwise.

        Raises:
            - FileNotFoundError: If the construction.json file is not found.
            - Exception: Mismatched construction.json vs processor parameters.
        """

        data_filepath = 'data/processed_raw_data/' + self.processed_data_name
        if not os.path.exists(data_filepath):
            return False
        else:
            return self._load_data(data_filepath)
            
    def _load_data(self, data_filepath):
        """
        Load the processed data and check if it matches the processor parameters.
        """

        try:
            with open(data_filepath + '/construction.json', 'r') as f:
                construction_data = json.load(f)

            if construction_data == self.get_construction():
                return True
            else:
                raise Exception(
                    'RawDataProcessor - construction.json file under the data_name ' + self.processed_data_name + 'does not match processor parameters')
        except:
            raise FileNotFoundError(
                'RawDataProcessor - construction.json file not found')

    def _save_data(self, processed_data):
        """
        Save processed data as a csv file.
        Save the construction parameters as a json file.

        Args:
            - processed_data (pd.DataFrame): The processed data.
        """

        data_file_path = 'data/processed_raw_data/' + self.processed_data_name
        if not os.path.exists(data_file_path):
            os.makedirs(data_file_path)
        processed_data.to_csv(data_file_path + '/data.csv')
        with open(data_file_path + '/construction.json', 'w') as f:
            json.dump(self.get_construction(), f, cls=NumpyEncoder,
                      separators=(', ', ': '), indent=4)

        print('Data saved to ' + data_file_path)

    def _plot_3D_data(self):
        """
        Plot the processed 3D data.

        Args:
            - processed_data (pd.DataFrame): The processed data.
        """


        colour_output = self.processed_data[self.dependent_variable]

        fig = plt.figure(figsize=(10, 10))

        # Create grid for plot and colour bar.
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 0.2])

        ax = fig.add_subplot(gs[0], projection='3d')
        p = ax.scatter(self.processed_data[self.independent_variables[0]],
                        self.processed_data[self.independent_variables[1]],
                        self.processed_data[self.independent_variables[2]],
                        c=colour_output, s=15, cmap='viridis',
                        vmin=np.percentile(colour_output, 5),
                        vmax=np.percentile(colour_output, 95), alpha=1)
        ax.set_xlabel(self.independent_variables[0], fontsize=15)
        ax.set_ylabel(self.independent_variables[1], fontsize=15)
        ax.set_zlabel(self.independent_variables[2], fontsize=15)
        ax.set_title('Processed 3D data', fontsize=20)
        cbr_ax = fig.add_subplot(gs[1])
        fig.colorbar(p, cax=cbr_ax, orientation='horizontal')
        fig.savefig('data/processed_raw_data/' + self.processed_data_name + '/3D_plot.png')
        plt.close()
    
    def _plot_2D_data_slice(self, processed_data, slice_axis, slice_value, interpolation_range=100, resolution=20):
        """
        Plot a higher-resolution slice of the processed 2D data.

        Args:
            - processed_data (pd.DataFrame): The processed data.
            - slice_axis (str): The axis to slice along.
            - slice_value (float): The value to slice at.
            - interpolation_range (float): Range of data around the slice value.
            - resolution (int): Number of points in the finer grid for higher resolution.
        """

        if slice_axis not in self.independent_variables:
            raise ValueError('RawDataProcessor - Invalid slice axis')

        # Define the x and y axes for slicing
        if slice_axis == self.independent_variables[0]:
            x = self.independent_variables[1]
            y = self.independent_variables[2]
        elif slice_axis == self.independent_variables[1]:
            x = self.independent_variables[0]
            y = self.independent_variables[2]
        else:
            x = self.independent_variables[0]
            y = self.independent_variables[1]

        # Filter data within the interpolation range of the slice axis
        slice_data = processed_data[
            (processed_data[slice_axis] >= slice_value - interpolation_range / 2) &
            (processed_data[slice_axis] <= slice_value + interpolation_range / 2)
        ]

        if slice_data.empty:
            raise ValueError(f'RawDataProcessor - No data found for slice at {slice_axis} = {slice_value}')

        # Group and average data by x and y
        slice_data = slice_data.groupby([x, y]).mean().reset_index()

        # Create the interpolation grid for higher resolution
        x_values = slice_data[x].unique()
        y_values = slice_data[y].unique()

        x_fine = np.linspace(x_values.min(), x_values.max(), resolution)
        y_fine = np.linspace(y_values.min(), y_values.max(), resolution)

        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

        # Perform cubic interpolation for smoother results
        Z_fine = griddata(
            (slice_data[x], slice_data[y]),  # Original data points
            slice_data[self.dependent_variable],  # Values to interpolate
            (X_fine, Y_fine),  # Finer grid for interpolation
            method='cubic'  # Use cubic interpolation for smooth results
        )

        # Create the plot
        fig, ax = plt.subplots()

        # Plot the interpolated data
        c = ax.pcolormesh(x_fine, y_fine, Z_fine, shading='auto', cmap='viridis')

        # Add colorbar and labels
        fig.colorbar(c, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f'Processed 2D data slice at {slice_axis} = {slice_value}')

        # Save the figure
        fig.savefig(f'data/processed_raw_data/{self.processed_data_name}/2D_plot_slice_{slice_axis}_{slice_value}.png')
        plt.close()

    def _plot_2D_data(self, processed_data):
        """
        Plot the processed 2D data.

        Args:
            - processed_data (pd.DataFrame): The processed data.
        """
        raise NotImplementedError('RawDataProcessor - Custom plotting function not implemented')
    
    def _plot_1D_data(self, processed_data):
        """
        Plot the processed 1D data.

        Args:
            - processed_data (pd.DataFrame): The processed data.
        """
        raise NotImplementedError('RawDataProcessor - Custom plotting function not implemented')
    
    def _GBR_processor(self):
        """
        Process the data using the Great Barrier Reef processor.

        Returns:
            - pd.DataFrame: The processed data.

        Raises:
            - FileNotFoundError: If a raw data, or meta data, file isn't found.
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

        # Process Great Barrier Reef data here.
        selected_experiments = self.processor_params['experiments_list']
        frames = []
        for experiment in selected_experiments:
            experiment_data = raw_data[raw_data['Experiment'] == experiment].copy()

            experiment_meta_data = meta_data[meta_data['Experiment'] == experiment]

            wind_dir = np.pi / 2 - experiment_meta_data['Wind_Dir'].values[0] * np.pi / 180
            wind_speed = experiment_meta_data['WindSpeed'].values[0]
            boat_lat = experiment_meta_data['boat.lat'].values[0]
            boat_lon = experiment_meta_data['boat.lon'].values[0]

            experiment_data['x_diff'] = (experiment_data['gps.lon'] - boat_lon) * np.cos(boat_lat * np.pi/180) * 40075000/360
            experiment_data['y_diff'] = (experiment_data['gps.lat'] - boat_lat) * 110000
            experiment_data['z'] = experiment_data['altitudeRelative']

            experiment_data['x'] = -(experiment_data['x_diff'] * np.cos(wind_dir) + experiment_data['y_diff'] * np.sin(wind_dir))
            experiment_data['y'] = -experiment_data['x_diff'] * np.sin(wind_dir) + experiment_data['y_diff'] * np.cos(wind_dir)

            experiment_data[self.processor_params['output_header']] = experiment_data[self.processor_params['input_header']] * wind_speed * 100**3

            if self.processor_params['log_output_data']:
                experiment_data[self.processor_params['output_header']] = np.log10(experiment_data[self.processor_params['output_header']])

            frames.append(experiment_data)

        normalised_data = pd.concat(frames)
        normalised_data = normalised_data[normalised_data.notnull().all(axis=1)]

        normalised_data = normalised_data.reset_index()

        if self.processor_params['gridding']:
            box_gridder = BoxGridder(
                normalised_data,
                grid_size=self.processor_params['gridding'],
                output_path='data/processed_raw_data/' +
                            self.processed_data_name,
                input_data_logarithmised=self.processor_params['log_output_data'],
                output_data_logarithmised=self.processor_params['log_output_data'])
            processed_data = box_gridder.processed_data
            box_gridder.get_sample_histograms()
            box_gridder.visualise_counts()
        else:
            processed_data = normalised_data

        processed_data = processed_data[processed_data['x'] > 150]
        processed_data = processed_data.drop(columns=['counts', 'samples'])
        return processed_data

    def _xylo_processor(self):
        """
        Process the data using the XYLO processor.

        Returns:
            - pd.DataFrame: The processed data.

        Raises:
            - FileNotFoundError: If a raw data, or meta data, file isn't found.
        """

        identifier = self.processor_params['identifier']
        tax_group = self.processor_params['tax_group']
        num_x_cells = self.processor_params['num_x_cells']
        num_y_cells = self.processor_params['num_y_cells']
        timestep = self.processor_params['timestep']

        try:
            raw_data = pd.read_csv(self.data_path + '/raw_data/' + self.raw_data_filename + '.csv')
        except:
            raise FileNotFoundError('RawDataProcessor - raw data file not found')

        def query_data(raw_data, identifier, tax_group = 'scientificname') -> pd.DataFrame:
            
            def _get_nearest_major_city(latitude, longitude):

                geolocator = Nominatim(user_agent="geoapi")

                # Reverse geocoding to find the nearest location details
                location = geolocator.reverse((latitude, longitude), exactly_one=True)

                if location and 'address' in location.raw:
                    address = location.raw['address']
                    # Attempt to get the city and country from the address
                    city = address.get('city') or address.get('town') or address.get('village') or address.get('state')
                    country = address.get('country')
                    return (city if city else "Unknown city", country if country else "Unknown country")

                return ("Unknown city", "Unknown country")

            city_and_country = _get_nearest_major_city(raw_data['decimallatitude'][0], raw_data['decimallongitude'][0])
            location_name = f"{city_and_country[0]}_{city_and_country[1]}"

            self.processor_params['location_name'] = location_name

            if tax_group == 'scientificname':

                input_scientificname = identifier

                species_counts = self.occurance_data['scientificname'].value_counts()
                species_counts.to_csv(f'{self.data_path}/{self.location_name}/species_counts.csv', header=['count'])

                filtered_occurance_data = self.occurance_data[self.occurance_data['scientificname'] == input_scientificname]

            elif tax_group == 'class':

                input_class = identifier
                filtered_occurance_data = raw_data[raw_data['class'] == input_class]

            elif tax_group == 'order':

                input_order = identifier
                filtered_occurance_data = raw_data[raw_data['order'] == input_order]

            elif tax_group == 'family':

                input_family = identifier
                filtered_occurance_data = raw_data[raw_data['family'] == input_family]

            elif tax_group == 'genus':

                input_genus = identifier

                filtered_occurance_data = raw_data[raw_data['genus'] == input_genus]

            elif tax_group == 'phylum':
                
                input_phylum = identifier
                filtered_occurance_data = raw_data[raw_data['phylum'] == input_phylum]
                
            if len(filtered_occurance_data) <= 20:
                raise ValueError("Not enough data! Need at least 20 sightings to model this species.")

            selected_column_data = filtered_occurance_data[['decimallongitude', 'decimallatitude', 'day', 'month', 'year']]

            selected_column_data = selected_column_data.copy()
            selected_column_data['day_number'] = pd.to_datetime(
                selected_column_data[['year', 'month', 'day']]
            ).dt.dayofyear

            selected_column_data = selected_column_data.drop(columns=['day', 'month'])

            return selected_column_data
        
        def grid_data(queried_data, num_x_cells, num_y_cells, timestep, raw_data):
           
            lat_min, lat_max = raw_data['decimallatitude'].min(), raw_data['decimallatitude'].max()
            lon_min, lon_max = raw_data['decimallongitude'].min(), raw_data['decimallongitude'].max()
            lat_cell_size = (lat_max - lat_min) / num_y_cells
            lon_cell_size = (lon_max - lon_min) / num_x_cells
            
            def get_grid_cell(lat, lon):
                try:
                    x_idx = int((lon - lon_min) / lon_cell_size)
                    y_idx = int((lat - lat_min) / lat_cell_size)
                    x_idx = min(x_idx, num_x_cells - 1)
                    y_idx = min(y_idx, num_y_cells - 1)
                except:
                    raise ValueError("Error assigning grid cell")
                return x_idx, y_idx
            
            gridded_data = queried_data.copy()
            gridded_data[['grid_x', 'grid_y']] = queried_data.apply(
                lambda row: pd.Series(get_grid_cell(row['decimallatitude'], row['decimallongitude'])),
                axis=1
            )

            if timestep == 'year':

                # Count the number of points in each grid cell for each year
                yearly_grid_counts = gridded_data.groupby(['year', 'grid_x', 'grid_y']).size().unstack(fill_value=0)

                # Convert yearly_grid_counts into a regular table with specified columns
                yearly_grid_counts_reset = yearly_grid_counts.stack().reset_index()
                yearly_grid_counts_reset.columns = ['year', 'grid_x', 'grid_y', 'counts']

                # Assuming yearly_grid_counts_reset has been created as in your original code
                min_year, max_year = queried_data['year'].min(), queried_data['year'].max()
                years = np.arange(min_year, max_year + 1)
                grid_x_values = np.arange(num_x_cells)
                grid_y_values = np.arange(num_y_cells)

                # Step 1: Create a full grid DataFrame
                full_grid = pd.MultiIndex.from_product([years, grid_x_values, grid_y_values], names=['year', 'grid_x', 'grid_y']).to_frame(index=False)

                # Step 2: Merge the full grid with the observed counts and fill missing values with 0
                yearly_grid_counts_complete = full_grid.merge(yearly_grid_counts_reset, on=['year', 'grid_x', 'grid_y'], how='left').fillna(0)

                # Step 3: Ensure 'counts' column is integer (if it makes sense for your data)
                yearly_grid_counts_complete['counts'] = yearly_grid_counts_complete['counts'].astype(int)

                # Step 4: Recalculate centroids
                lat_min, lat_max = queried_data['decimallatitude'].min(), queried_data['decimallatitude'].max()
                lon_min, lon_max = queried_data['decimallongitude'].min(), queried_data['decimallongitude'].max()
                lat_cell_size = (lat_max - lat_min) / num_y_cells
                lon_cell_size = (lon_max - lon_min) / num_x_cells

                yearly_grid_counts_complete['x'] = yearly_grid_counts_complete['grid_x'] * lon_cell_size + lon_min + (lon_cell_size / 2)
                yearly_grid_counts_complete['y'] = yearly_grid_counts_complete['grid_y'] * lat_cell_size + lat_min + (lat_cell_size / 2)

                gridded_data = yearly_grid_counts_complete

            elif timestep == 'day':
                daily_grid_counts = gridded_data.groupby(['day_number', 'grid_x', 'grid_y']).size().unstack(fill_value=0)
                
                # Convert yearly_grid_counts into a regular table with specified columns
                daily_grid_counts_reset = daily_grid_counts.stack().reset_index()
                daily_grid_counts_reset.columns = ['day_number', 'grid_x', 'grid_y', 'counts']

                # Assuming yearly_grid_counts_reset has been created as in your original code
                min_day, max_day = queried_data['day_number'].min(), queried_data['day_number'].max()
                days = np.arange(min_day, max_day + 1)
                grid_x_values = np.arange(num_x_cells)
                grid_y_values = np.arange(num_y_cells)

                # Step 1: Create a full grid DataFrame
                full_grid = pd.MultiIndex.from_product([days, grid_x_values, grid_y_values], names=['day_number', 'grid_x', 'grid_y']).to_frame(index=False)
                
                daily_grid_counts = gridded_data.groupby(['day_number', 'grid_x', 'grid_y']).size().unstack(fill_value=0)
                # Step 2: Merge the full grid with the observed counts and fill missing values with 0
                daily_grid_counts_complete = full_grid.merge(daily_grid_counts_reset, on=['day_number', 'grid_x', 'grid_y'], how='left').fillna(0)

                # Step 3: Ensure 'counts' column is integer (if it makes sense for your data)
                daily_grid_counts_complete['counts'] = daily_grid_counts_complete['counts'].astype(int)

                # Step 4: Recalculate centroids
                lat_min, lat_max = queried_data['decimallatitude'].min(), queried_data['decimallatitude'].max()
                lon_min, lon_max = queried_data['decimallongitude'].min(), queried_data['decimallongitude'].max()
                lat_cell_size = (lat_max - lat_min) / num_y_cells
                lon_cell_size = (lon_max - lon_min) / num_x_cells

                daily_grid_counts_complete['x'] = daily_grid_counts_complete['grid_x'] * lon_cell_size + lon_min + (lon_cell_size / 2)
                daily_grid_counts_complete['y'] = daily_grid_counts_complete['grid_y'] * lat_cell_size + lat_min + (lat_cell_size / 2)

                gridded_data = daily_grid_counts_complete

            gridded_data = gridded_data.copy()
            return gridded_data
            
        queried_data = query_data(raw_data, identifier, tax_group)
        
        gridded_data = grid_data(queried_data, num_x_cells, num_y_cells, timestep, raw_data)

        gridded_data = gridded_data.drop(columns=['grid_x', 'grid_y'])
        
        return gridded_data
        

    # ADD OTHER PROCESSORS HERE.
