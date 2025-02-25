import numpy as np
# import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.preprocessing import StandardScaler
from jaxlib.xla_extension import ArrayImpl
from sklearn.gaussian_process.kernels import Sum, Product
from sklearn.gaussian_process.kernels import Kernel as GP_Kernel
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel
from sklearn.preprocessing import StandardScaler

import ast


import json
import joblib
import os
# from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.transforms import Bbox
# import imageio
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from shapely.geometry import Point, Polygon, MultiPolygon
# from shapely.strtree import STRtree
# from PIL import Image
# import matplotlib
# from tqdm import tqdm
# from matplotlib.colors import LogNorm
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch

from data_processing.data_processor import DataProcessor
from gaussian_process_toolbox.kernel import Kernel

class GP:
    """
    A class to represent a Gaussian Process model
    
    """

    def __init__(self, 
                 data_processor,
                 kernel: Kernel,
                 uncertainty_method = None,
                 uncertainty_params = None,
                 root_results_path = '/results/gaussian_process_results'):

        if not isinstance(data_processor, DataProcessor):
            raise TypeError("Sampler - data_processor must be an instance of the DataProcessor class")
        if not isinstance(root_results_path, str):
            raise TypeError("Sampler - root_results_path must be a string")
        
        root_results_path = os.getcwd() + root_results_path

        self.data_processor = data_processor

        self.kernel_obj = kernel
        self.kernel = kernel.get_kernel()
        self.kernel_config = kernel.kernel_config
        self.kernel_params = kernel.kernel_params
        
        self.uncertainty_method = uncertainty_method
        self.uncertainty_params = uncertainty_params

        self.training_data, self.testing_data = data_processor.process_data()
        self.data_construction = data_processor.get_construction()

        self.independent_variables = data_processor.independent_variables
        self.dependent_variable = data_processor.dependent_variable

        self.results_path = os.path.join(root_results_path, data_processor.processed_data_name)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path, exist_ok=True)

        self._validate_kernel()

        self.gp_construction = self.get_construction()
        self.instance = self._get_instance_number()
        instance_folder = os.path.join(self.results_path, f'instance_{self.instance}')
        os.makedirs(instance_folder, exist_ok=True)

        self.X_train = self.training_data[self.independent_variables].values
        self.y_train = self.training_data[self.dependent_variable].values

        self.X_test = self.testing_data[self.independent_variables].values
        self.y_test = self.testing_data[self.dependent_variable].values

        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.X_train_scaled = self.X_scaler.fit_transform(self.X_train)
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()

        self.X_test_scaled = self.X_scaler.transform(self.X_test)
        self.y_test_scaled = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()

        self.gp_model = None
        self.trained = False

        self.optimized_params = None
        self.num_params = None

    def _validate_kernel(self):
        # Extract unique covariate identifiers from kernel_config
        num_kernel_covariates = np.unique([i for sublist in [val for val in self.kernel_config.values()] for i in sublist]).size

        # Check if kernel covariates match the independent variables
        if num_kernel_covariates != len(self.independent_variables):
            raise ValueError("Kernel covariates do not match the independent variables")
        
    def get_construction(self):
        construction = {
            'kernel': self.kernel_obj.get_construction(),
            'data_processor': self.data_construction,
            'uncertainty_method': self.uncertainty_method,
            'uncertainty_params': self.uncertainty_params
        }
        return construction

    def _save_run(self, gp_model):
        instance_folder = os.path.join(self.results_path, f'instance_{self.instance}')
        joblib.dump(gp_model, f'{instance_folder}/gaussian_process_model.pkl')
        json_safe_data = self._convert_tuple_keys_to_strings(self.gp_construction)
        with open(f'{instance_folder}/gp_construction.json', 'w') as f:
            json.dump(json_safe_data, f, indent=4)

    def _get_instance_number(self):
        """
        Determine the instance number where the data should be saved.

        Returns:
            int: The instance number where the data is to be saved.
        """

        if not os.path.exists(self.results_path):
            return 1

        instance_folders = os.listdir(self.results_path)
        for instance_folder in instance_folders:
            folder_path = os.path.join(self.results_path, instance_folder)
            gp_path = os.path.join(folder_path, 'gp_construction.json')

            if os.path.exists(gp_path):
                with open(gp_path) as f:
                    json_safe_data = json.load(f)
                instance_hyperparams = self._convert_string_keys_to_tuples(json_safe_data)
                
                print(self._convert_tuples_to_lists(instance_hyperparams))
                print(self._convert_tuples_to_lists(self._convert_arrays_to_lists(self.get_construction())))

                if self._convert_tuples_to_lists(self._convert_arrays_to_lists(self.get_construction())) == self._convert_tuples_to_lists(instance_hyperparams):
                    return int(instance_folder.split('_')[1])

        instances = [int(folder.split('_')[1]) for folder in instance_folders if '_' in folder]
        return 1 if not instances else min(i for i in range(1, max(instances) + 2) if i not in instances)

    def train(self):
        """
        Train the Gaussian Process model on the training data.
        """
        gp_model_file = os.path.join(self.results_path, f'instance_{self.instance}', 'gaussian_process_model.pkl')
        if os.path.exists(gp_model_file):
            print(f'Loading existing GP model from {gp_model_file}')
            gp_model = joblib.load(gp_model_file)
        else:
            gp_model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)
            gp_model.fit(self.X_train_scaled, self.y_train_scaled)
            print(f'Fitted new GP model and saving to {gp_model_file}')
            self._save_run(gp_model)

        results_file = os.path.join(self.results_path, f'instance_{self.instance}', 'results.json')
        results = self._get_training_results(gp_model)
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
        
        self.trained = True
        self.gp_model = gp_model

        return gp_model
    
    def _extract_optimized_kernel_params(self, kernel):
        """
        Recursively extracts optimized hyperparameters from a fitted kernel object.

        Args:
            kernel: A fitted kernel object (e.g., Sum, Product, RBF, Matern, etc.)

        Returns:
            A dictionary representing the structure and parameters of the kernel.
        """
        optimized_params = {"type": type(kernel).__name__}

        # ✅ Handle KernelTransformer explicitly (preserving its dimensions)
        if isinstance(kernel, Kernel._KernelTransformer) or isinstance(kernel, self.kernel_obj._KernelTransformer):
            optimized_params["dims"] = kernel.dims
            optimized_params["base_kernel"] = self._extract_optimized_kernel_params(kernel.base_kernel)
            return optimized_params

        # ✅ Handle composite kernels (Sum, Product) for ANY number of sub-kernels
        if isinstance(kernel, (Sum, Product)):
            optimized_params["operation"] = "Sum" if isinstance(kernel, Sum) else "Product"
            optimized_params["sub_kernels"] = []
            
            # Dynamically find sub-kernels (k1, k2, k3, ...)
            for attr_name in dir(kernel):
                if attr_name.startswith("k") and attr_name[1:].isdigit():  # Matches k1, k2, k3, ...
                    sub_kernel = getattr(kernel, attr_name)
                    if isinstance(sub_kernel, GP_Kernel):  # Ensure it's a valid sub-kernel
                        optimized_params["sub_kernels"].append(self._extract_optimized_kernel_params(sub_kernel))
            
            return optimized_params

        # ✅ Extract parameters from individual base kernels (RBF, Matern, etc.)
        if isinstance(kernel, (RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel)):
            optimized_params["parameters"] = {}

            for param_name, param_value in kernel.get_params().items():
                if isinstance(param_value, np.ndarray):  # Convert NumPy arrays to lists
                    optimized_params["parameters"][param_name] = param_value.tolist()
                elif isinstance(param_value, (float, int)):  # Convert scalars directly
                    optimized_params["parameters"][param_name] = param_value
                elif isinstance(param_value, GP_Kernel):  # Handle nested kernel parameters
                    optimized_params["parameters"][param_name] = self._extract_optimized_kernel_params(param_value)
                else:
                    optimized_params["parameters"][param_name] = str(param_value)  # Fallback for unknown types

            return optimized_params

        raise ValueError(f"Unsupported kernel type: {type(kernel)}")

    def _count_kernel_parameters(self, kernel_dict=None):
        """
        Recursively counts the number of actual hyperparameters in a kernel dictionary.
        Ignores metadata like bounds.

        Args:
            kernel_dict (dict): Dictionary representation of the kernel structure.

        Returns:
            int: Total number of hyperparameters.
        """
        param_count = 0

        if kernel_dict is None:
            kernel_dict = self.optimized_params

        if "parameters" in kernel_dict:
            for key, value in kernel_dict["parameters"].items():
                if "bounds" not in key:  # Ignore bounds
                    if isinstance(value, list):  # Count each element in a list separately
                        param_count += len(value)
                    else:
                        param_count += 1

        if "sub_kernels" in kernel_dict:  # Handle composite kernels (Sum, Product)
            param_count += sum(self._count_kernel_parameters(sub_kernel) for sub_kernel in kernel_dict["sub_kernels"])

        if "base_kernel" in kernel_dict:  # Handle _KernelTransformer
            param_count += self._count_kernel_parameters(kernel_dict["base_kernel"])

        return param_count

    def _get_training_results(self, gp_model):
        self.optimized_params = self._extract_optimized_kernel_params(gp_model.kernel_)
        self.num_params = self._count_kernel_parameters()

        self.training_r2 = gp_model.score(self.X_train_scaled, self.y_train_scaled)
        self.testing_r2 = gp_model.score(self.X_test_scaled, self.y_test_scaled)
        
        # y_pred, y_std = gp_model.predict(self.X_test, return_std=True)
        # y_pred = np.expm1(y_pred)
        # y_test = np.expm1(self.y_test)
        # y_std = np.expm1(y_std)

        # # Calculate the mean absolute error
        # self.mae = np.mean(np.abs(y_test - y_pred))

        # # Calculate the root mean squared error
        # self.rmse = np.sqrt(np.mean((y_test - y_pred)**2))

        # # Calculate normalised root mean squared error
        # self.nrmse = self.rmse / np.std(y_test)

        # # Calculate log likelihood of test data
        # log_likelihoods = (
        #     -0.5 * np.log(2 * np.pi * y_std**2)
        #     - ((y_test - y_pred) ** 2) / (2 * y_std**2)
        # )

        # # Total log-likelihood
        # self.av_log_likelihood = np.mean(log_likelihoods) 

        # Serialize results to JSON
        results = {
            'fitted_kernel_params': self.optimized_params,
            'num_params': self.num_params,
            'training_r2': self.training_r2,
            'testing_r2': self.testing_r2
            # 'mae': self.mae,
            # 'rmse': self.rmse,
            # 'nrmse': self.nrmse,
            # 'av_log_likelihood': self.av_log_likelihood
        }

        return results
        
    def _convert_arrays_to_lists(self, obj):
        """
        Utility function which recursively converts NumPy arrays to lists within a dictionary.

        Args:
            - obj (dict or list or np.ndarray): The input object to convert.

        Returns:
            - dict or list or list: The converted object with NumPy arrays replaced by lists.
        """
        if isinstance(obj, dict):
            # If the object is a dictionary, convert arrays within its values
            return {k: self._convert_arrays_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # If the object is a list, convert arrays within its elements
            return [self._convert_arrays_to_lists(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            # If the object is a NumPy array, convert it to a list
            return obj.tolist()
        elif isinstance(obj, ArrayImpl):
            return np.asarray(obj).tolist()
        else:
            # If the object is neither a dictionary, list, nor NumPy array, return it unchanged
            return obj

    def _convert_tuple_keys_to_strings(self, d):
        """Recursively converts tuple keys in a dictionary to string keys."""
        if isinstance(d, dict):
            return {str(k) if isinstance(k, tuple) else k: self._convert_tuple_keys_to_strings(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._convert_tuple_keys_to_strings(i) for i in d]
        else:
            return d

    def _convert_string_keys_to_tuples(self, d):
        """Converts string keys back to tuples if they were originally tuples."""
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                    try:
                        key = ast.literal_eval(k)
                        if isinstance(key, tuple):  # Ensure it's actually a tuple
                            new_dict[key] = self._convert_string_keys_to_tuples(v)
                            continue  # Skip re-adding the string version
                    except (SyntaxError, ValueError):
                        pass  # If parsing fails, keep the string key
                new_dict[k] = self._convert_string_keys_to_tuples(v)
            return new_dict
        elif isinstance(d, list):
            return [self._convert_string_keys_to_tuples(i) for i in d]
        else:
            return d
        
    def _convert_tuples_to_lists(self, obj):
        """
        Recursively converts all tuples in a nested structure (dict, list, tuple, etc.) to lists.
        
        Args:
            obj (any): The input object (dict, list, tuple, etc.).
            
        Returns:
            any: The same structure with tuples converted to lists.
        """
        if isinstance(obj, dict):
            # Convert tuples in keys, but ensure keys remain hashable (only convert non-keys)
            return {k if not isinstance(k, tuple) else tuple(self._convert_tuples_to_lists(k)): self._convert_tuples_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tuples_to_lists(i) for i in obj]
        elif isinstance(obj, tuple):
            return [self._convert_tuples_to_lists(i) for i in obj]  # Convert tuple to list
        else:
            return obj  # Base case: return the object as is if it's not a tuple, list, or dict

    def predict(self, points):
        """
        Predict the dependent variable at given points.

        Args:
            points (array-like): The points at which to make predictions.

        Returns:
            array-like: The predicted values at the given points.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")

        points_scaled = self.X_scaler.transform(points)

        mean_preds_scaled, std_preds_scaled = self.gp_model.predict(points_scaled, return_std=True)
        mean_preds = self.y_scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
        std_preds = std_preds_scaled * self.y_scaler.scale_[0]  # Only multiply by standard deviation

        return mean_preds, std_preds
        

    #     # Fit the model
    #     X = self.gridded_data[self.covariates].values
    #     y_exp = self.gridded_data[self.dependent_variable].values
    #     y = np.log1p(self.gridded_data[self.dependent_variable].values)

    #     # Scale covariates
    #     X_scaled = self.scaler.fit_transform(X)

    #     # Split the data into training and testing sets randomly
    #     # Create a binary label for stratification: 1 if y_exp > 0, 0 otherwise
    #     y_strat = (y_exp > 0).astype(int)

    #     # Perform stratified train-test split
    #     self.X_train, self.X_test, y_exp_train, y_exp_test = train_test_split(
    #         X_scaled, y_exp, train_size=train_test_ratio, random_state=42, stratify=y_strat
    #     )
    #     self.y_train = np.log1p(y_exp_train)
    #     self.y_test = np.log1p(y_exp_test)

    #     errors_train = [self.zero_count_error if count == 0 else count * self.accuracy for count in y_exp_train]
    #     errors_test = [self.zero_count_error if count == 0 else count * self.accuracy for count in y_exp_test]

    #     ln_error_train = errors_train / (y_exp_train + 1)  # Adjusted relative error for log1p space
    #     ln_error_test = errors_test / (y_exp_test + 1)
    #     alpha_train = ln_error_train**2

    #     if not os.path.exists(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}'):
    #         os.makedirs(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}')

    #     proposed_config = {
    #         'kernel': self.kernel,
    #         'kernel_params': self.kernel_params,
    #         'accuracy': self.accuracy,
    #         'zero_count_error': self.zero_count_error,
    #         'train_test_ratio': train_test_ratio,
    #         'covariates': self.covariates,
    #         'dependent_variable': self.dependent_variable,
    #         'n_restarts_optimizer': self.n_restarts_optimizer
    #     }

    #     def convert_tuples_to_arrays(d):
    #         """
    #         Recursively traverse a dictionary and convert all tuples into numpy arrays.
    #         """
    #         if isinstance(d, dict):
    #             # If it's a dictionary, apply the function to all keys and values
    #             return {key: convert_tuples_to_arrays(value) for key, value in d.items()}
    #         elif isinstance(d, tuple):
    #             # If it's a tuple, convert it to a numpy array
    #             return list(d)
    #         elif isinstance(d, list):
    #             # If it's a list, apply the function to each item in the list
    #             return [convert_tuples_to_arrays(item) for item in d]
    #         else:
    #             # If it's neither, return it as is
    #             return d

    #     # Load the configuration if it exists
    #     config_path = f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/gaussian_process_config.json'
    #     if os.path.exists(config_path):
    #         with open(config_path, 'r') as config_file:
    #             loaded_config = json.load(config_file)

    #         if convert_tuples_to_arrays(proposed_config) != convert_tuples_to_arrays(loaded_config):
    #             raise ValueError('Loaded configuration does not match the proposed configuration')
    #         else:
    #             print('GP already trained with the same configuration')
    #             self.gp = joblib.load(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/gaussian_process_model.pkl')
    #     else:
    #         self.gp = GaussianProcessRegressor(kernel=self.kernel_func, alpha=alpha_train, n_restarts_optimizer=self.n_restarts_optimizer)
    #         self.gp.fit(self.X_train, self.y_train)

    #         # Plot the progress of the optimization
    #         joblib.dump(self.gp, f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/gaussian_process_model.pkl')
    #         with open(config_path, 'w') as config_file:
    #             json.dump(proposed_config, config_file, indent=4)


    #     def kernel_to_dict(kernel):
    #         """
    #         Recursively converts a kernel into a dictionary of its type and parameters.
    #         Handles nested kernels and ensures all parameters are JSON-serializable.
    #         """
    #         if not isinstance(kernel, Kernel):
    #             raise ValueError(f"Expected a Kernel, got {type(kernel)}")

    #         kernel_dict = {
    #             "type": type(kernel).__name__,
    #             "parameters": {}
    #         }

    #         params = kernel.get_params()
    #         for param, value in params.items():
    #             if isinstance(value, np.ndarray):  # Convert numpy arrays to lists
    #                 kernel_dict["parameters"][param] = value.tolist()
    #             elif isinstance(value, Kernel):  # Recursively handle nested kernels
    #                 kernel_dict["parameters"][param] = kernel_to_dict(value)
    #             else:  # Directly add other serializable types
    #                 kernel_dict["parameters"][param] = value

    #         return kernel_dict

    #     # Save the kernel to a dictionary
    #     self.optimized_params = kernel_to_dict(self.gp.kernel_)

    #     self.training_r2 = self.gp.score(self.X_train, self.y_train)
    #     self.testing_r2 = self.gp.score(self.X_test, self.y_test)
        
    #     y_pred, y_std = self.gp.predict(self.X_test, return_std=True)
    #     y_pred = np.expm1(y_pred)
    #     y_test = np.expm1(self.y_test)
    #     y_std = np.expm1(y_std)

    #     # Calculate the mean absolute error
    #     self.mae = np.mean(np.abs(y_test - y_pred))

    #     # Calculate the root mean squared error
    #     self.rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    #     # Calculate normalised root mean squared error
    #     self.nrmse = self.rmse / np.std(y_test)

    #     # Calculate log likelihood of test data
    #     log_likelihoods = (
    #         -0.5 * np.log(2 * np.pi * y_std**2)
    #         - ((y_test - y_pred) ** 2) / (2 * y_std**2)
    #     )

    #     # Total log-likelihood
    #     self.av_log_likelihood = np.mean(log_likelihoods) 

    #     if self.print_results:
    #         print(f'Fitted kernel parameters: {self.optimized_params}')
    #         print(f'Training R^2: {self.training_r2}')
    #         print(f'Testing R^2: {self.testing_r2}')
    #         print(f'Mean Absolute Error: {self.mae}')
    #         print(f'Root Mean Squared Error: {self.rmse}')
    #         print(f'Normalised RMSE: {self.nrmse}')
    #         print(f'Average Log Likelihood: {self.av_log_likelihood}')

    #     # Serialize results to JSON
    #     results = {
    #         'fitted_kernel_params': self.optimized_params,
    #         'training_r2': self.training_r2,
    #         'testing_r2': self.testing_r2,
    #         'mae': self.mae,
    #         'rmse': self.rmse,
    #         'nrmse': self.nrmse,
    #         'av_log_likelihood': self.av_log_likelihood
    #     }

    #     results_path = f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/results.json'
    #     with open(results_path, 'w') as results_file:
    #         json.dump(results, results_file, indent=4)

    #     self.trained = True
    
    # def predict(self, resolution, time = 2020, plot_results = False):
    #     """
    #     Predict the count and probability of one or more counts at a given time and resolution.

    #     Args:
    #         - resolution (int or array-like): Resolution of the prediction grid. If an integer, it will be used for both spatial dimensions. If an array-like object, it should contain the resolution for each spatial dimension.
    #         - time (int or list): Time at which to make the prediction. If a list, it should contain the minimum time, maximum time, and time step. An integer will be treated as a single time step.
    #         - plot_results (bool): Whether to plot the results

    #     Raises:
    #         - ValueError: If the model has not been trained yet
    #         - ValueError: If the resolution or time arguments are invalid

    #     Returns:
    #         - tuple: Tuple containing the lower bounds, means, upper bounds, and probabilities of the predictions
        
    #     This function performs the following steps:
    #         1. Generate a prediction grid for the spatial coordinates
    #         2. Perform predictions timestep by timestep on the grid
    #         3. Calculate the probability of one or more counts and back-transform the predictions
    #         4. Reshape the predictions to match the grid dimensions
    #         5. Save the predictions and probabilities
    #         6. Plot the results if specified
    #     """
    #     if not self.trained:
    #         raise ValueError('Model has not been trained yet')

    #     if isinstance(resolution, int):
    #         resolution = [resolution] * len(self.spatial_covariates)
    #     elif len(resolution) != len(self.spatial_covariates):
    #         raise ValueError('Resolution must be an integer or an array-like object with length equal to the number of covariates')

    #     if isinstance(time, int):
    #         min_t = max_t = time
    #         t_step = 1
    #     elif isinstance(time, (list, tuple)) and len(time) == 3:
    #         min_t, max_t, t_step = time
    #     else:
    #         raise ValueError('Time must be an integer or an array-like object with three elements (min_t, max_t, t_step)')

    #     self.time = time
    #     # Generate a linspace for each spatial coordinate
    #     self.spatial_grids = [np.linspace(self.gridded_data[cov].min(), self.gridded_data[cov].max(), res) for cov, res in zip(self.spatial_covariates, resolution)]

    #     # Define the prediction grid for coordinates
    #     self.x_pred_coord = np.linspace(self.gridded_data[self.spatial_covariates[0]].min(), self.gridded_data[self.spatial_covariates[0]].max(), resolution[0])
    #     self.y_pred_coord = np.linspace(self.gridded_data[self.spatial_covariates[1]].min(), self.gridded_data[self.spatial_covariates[1]].max(), resolution[1])
        
    #     self.t_pred = np.arange(min_t, max_t + t_step, t_step)

    #     # Initialize prediction storage
    #     pred_means = []
    #     pred_lowers = []
    #     pred_uppers = []
    #     probs = []

    #     X = self.gridded_data[self.covariates].values
    #     X_scaled = self.scaler.fit_transform(X)

    #     # Perform predictions timestep by timestep on the grid
    #     for t in self.t_pred:
    #         X_pred = np.array(np.meshgrid(*self.spatial_grids, [t])).T.reshape(-1, len(self.spatial_covariates) + 1)
            
    #         # Scale prediction data
    #         X_pred_scaled = self.scaler.transform(X_pred)
    #         y_pred_mean, y_pred_std = self.gp.predict(X_pred_scaled, return_std=True)
            
    #         # Calculate probability of one or more counts and back-transform
    #         prob_1_or_more = 1 - np.exp(-np.expm1(y_pred_mean + y_pred_std**2 / 2))
    #         y_pred_lower = np.maximum(np.expm1(y_pred_mean - 1.96 * y_pred_std), 0)
    #         y_pred_upper = np.maximum(np.expm1(y_pred_mean + 1.96 * y_pred_std), 0)
    #         y_pred_mean = np.maximum(np.expm1(y_pred_mean), 0)
            
    #         # Reshape the predictions to match the grid dimensions
    #         probs.append(prob_1_or_more.reshape(resolution[0], resolution[1]).T)
    #         pred_means.append(y_pred_mean.reshape(resolution[0], resolution[1]).T)
    #         pred_lowers.append(y_pred_lower.reshape(resolution[0], resolution[1]).T)
    #         pred_uppers.append(y_pred_upper.reshape(resolution[0], resolution[1]).T)

    #     self.pred_means = pred_means
    #     self.pred_lowers = pred_lowers
    #     self.pred_uppers = pred_uppers
    #     self.probs = probs

    #     self.min_c = np.percentile(np.array(pred_lowers), 5)
    #     self.max_c = np.percentile(np.array(pred_uppers), 95)

    #     if plot_results:
    #         self.plot()

    #     return pred_lowers, pred_means, pred_uppers, probs

    # def plot(self):
    #     """
    #     Plot the predictions and probabilities of the Gaussian Process model.

    #     Raises:
    #         - ValueError: If the model has not been trained yet
        
    #     This function performs the following steps:
    #         1. Create a land mask for the coordinate grids
    #         2. Determine whether to create an animation or a single plot
    #         3. Create a directory for the predictions if it does not exist
    #         4. Create a directory for the frames if an animation is being created
    #         5. Create frames for each timestep if an animation is being created and a single plot if an animation is not being created
    #             - Overlay the predictions on a map with land and water features
    #             - Add gridlines and map features
    #             - Overlay the predictions on the map
    #             - Add a colorbar and title
    #             - Save the frame
    #         6. Save the plot

    #     """

    #     if not os.path.exists(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions'):
    #         os.makedirs(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions')

    #     self.land_mask = self._create_land_mask()

    #     if len(self.t_pred) == 1:
    #         self.animation = False
    #     else:
    #         self.animation = True
        
    #     if self.animation:
    #         if not os.path.exists(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions/frames'):
    #             os.makedirs(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions/frames')
    #         frames = []
    #         animation_filename = f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions/{self.temporal_covariates[0]}_animation.gif'
    #         if os.path.exists(animation_filename):
    #             print(f'Animation already exists in {animation_filename}')
    #         else:
    #             print('Creating animation...')
    #             for i in range(len(self.t_pred)):
    #                 frame_filename = f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions/frames/{self.temporal_covariates[0]}_{str(self.t_pred[i])}.png'
    #                 if os.path.exists(frame_filename):
    #                     print(f'Frame {i} already exists')
    #                 else:
    #                     frame = self._plot_timestep(i)
    #                     frame.savefig(frame_filename, bbox_inches='tight')
                
    #             with imageio.get_writer(animation_filename, mode='I', fps=0.5, loop =0) as writer:
    #                 for i in range(len(self.t_pred)):
    #                     frame = imageio.imread(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions/frames/{self.temporal_covariates[0]}_{str(self.t_pred[i])}.png')
    #                     writer.append_data(frame)
    #     else:
    #         file_name = f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions/{self.temporal_covariates[0]}_{str(self.t_pred[0])}.png'
    #         if os.path.exists(file_name):
    #             print('File already exists')
    #         else:
    #             fig = self._plot_timestep(0)
    #             fig.savefig(f'{self.results_path}/{self.location}/{self.identifier}/{self.name}/predictions/{self.temporal_covariates[0]}_{str(self.t_pred[0])}.png', bbox_inches='tight')
            
    #         plt.close(fig)
    #         return fig

    # def _create_land_mask(self):
    #     """
    #     Create a land mask for given coordinate grids using Cartopy's Natural Earth features,
    #     optimized for speed with spatial indexing.
    #     """
    #     mask = np.zeros((len(self.y_pred_coord), len(self.x_pred_coord)), dtype=bool)  # Initialize mask as all False
    #     land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m')

    #     # Collect all polygons from land geometries
    #     land_polygons = []
    #     for geom in land_feature.geometries():
    #         if isinstance(geom, Polygon):
    #             land_polygons.append(geom)
    #         elif isinstance(geom, MultiPolygon):
    #             land_polygons.extend(geom.geoms)

    #     # Create a spatial index for the land polygons
    #     polygon_tree = STRtree(land_polygons)

    #     # Create a grid of points
    #     grid_x, grid_y = np.meshgrid(self.x_pred_coord, self.y_pred_coord)
    #     grid_points = [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]

    #     # Use the spatial index to find polygons containing points
    #     for idx, point in enumerate(grid_points):
    #         # Query the spatial index to get indices of potential matching polygons
    #         potential_matches_indices = polygon_tree.query(point)
    #         # Check if the point is contained in any of the actual polygons
    #         if any(land_polygons[i].contains(point) for i in potential_matches_indices):
    #             j, i = divmod(idx, len(self.x_pred_coord))  # Map the flat index back to 2D grid indices
    #             mask[j, i] = True

    #     return mask

    # def _plot_timestep(self, step_num):
    #     pred_lower = self.pred_lowers[step_num]
    #     pred_mean = self.pred_means[step_num]
    #     pred_upper = self.pred_uppers[step_num]
    #     prob = self.probs[step_num]

    #     fig = plt.figure(figsize=(24, 7))
    #     gs = fig.add_gridspec(4, 5, width_ratios = [1,1,1,0.01,1], height_ratios=[2, 7, 2, 1], hspace=0.2, wspace=0.25)  # Add extra space with wspace and extra column
        
    #     title_ax = fig.add_subplot(gs[0, :])  # For the title
    #     axes = [fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree()) for j in range(3)]
    #     axes.append(fig.add_subplot(gs[1, 4], projection=ccrs.PlateCarree()))  # Fourth plot in the fifth column
    #     legend_ax = fig.add_subplot(gs[2, :])  # For the legend
    #     metrics_ax = fig.add_subplot(gs[3, :])  # For the metrics

    #     # count_scatter_data = self.gridded_data[self.gridded_data[self.temporal_covariates[0]] == self.t_pred[step_num]]
    #     # occurance_scatter_data = self.queried_data[self.queried_data[self.temporal_covariates[0]] == self.t_pred[step_num]]

    #     # Titles for each subplot
    #     titles = [
    #         f'Lower Bound - {self.temporal_covariates[0].capitalize()} {self.t_pred[step_num]}',
    #         f'Predicted Counts (Mean) - {self.temporal_covariates[0].capitalize()} {self.t_pred[step_num]}',
    #         f'Upper Bound - {self.temporal_covariates[0].capitalize()} {self.t_pred[step_num]}',
    #         f'Probability of ≥1 Count - {self.temporal_covariates[0].capitalize()} {self.t_pred[step_num]}'
    #     ]

    #     # Data to overlay on each subplot, with masking applied
    #     data_layers = [pred_lower, pred_mean, pred_upper, prob]
    #     masked_layers = [np.where(self.land_mask, data, np.nan) for data in data_layers]

    #     x_min, x_max = min(self.gridded_data[self.spatial_covariates[0]]), max(self.gridded_data[self.spatial_covariates[0]])
    #     y_min, y_max = min(self.gridded_data[self.spatial_covariates[1]]), max(self.gridded_data[self.spatial_covariates[1]])

    #     # Color range limits
    #     vmin_counts, vmax_counts =self.min_c, self.max_c
    #     vmin_prob, vmax_prob = 0, 1

    #     resolution = pred_lower.shape[0]

    #     # Use x_pred_coord and y_pred_coord as the coordinate axes for pcolormesh
    #     x_pred = np.linspace(x_min, x_max, resolution)
    #     y_pred = np.linspace(y_min, y_max, resolution)

    #     X, Y = np.meshgrid(x_pred, y_pred)

    #     # Initialize the heatmaps for colorbars
    #     heatmaps = []

    #     for ax, data, title in zip(axes, masked_layers, titles):
    #         # Add map features
    #         ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
    #         ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
    #         ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle='--', alpha=1)
    #         ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, alpha=1)
    #         ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.7, linewidth=0.5)
    #         ax.add_feature(cfeature.LAKES, edgecolor='blue', facecolor='lightblue', alpha=1)
    #         ax.add_feature(cfeature.STATES, edgecolor='black', linestyle='-', linewidth=0.5, alpha=1)

    #         # Set gridlines and extent
    #         gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray', alpha=1)
    #         gl.top_labels = False
    #         gl.right_labels = False
    #         gl.xlabel_style = {'size': 10, 'color': 'gray'}
    #         gl.ylabel_style = {'size': 10, 'color': 'gray'}
    #         ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())

    #         # Overlay heatmap
    #         vmin, vmax = (vmin_counts, vmax_counts) if 'Probability' not in title else (vmin_prob, vmax_prob)
    #         if vmax > self.log_transform_threshold:
    #             norm = LogNorm(vmin=vmin+1, vmax=vmax)
    #             colorbar_label = 'Log Scale'
    #             hm = ax.pcolormesh(X, Y, data, cmap='coolwarm', shading='nearest', transform=ccrs.PlateCarree(), alpha=0.8, norm=norm)
    #         else:
    #             norm = None
    #             colorbar_label = 'Linear Scale'
    #             hm = ax.pcolormesh(X, Y, data, cmap='coolwarm', shading='nearest', transform=ccrs.PlateCarree(), alpha=0.8, vmin=vmin, vmax=vmax)
    #         heatmaps.append(hm)  # Keep track of the heatmap for colorbar

    #         # # Scatter observed data
    #         # if not count_scatter_data.empty:
    #         #     if count_scatter_data['counts'].max() > self.log_transform_threshold:
    #         #         size = np.log(count_scatter_data['counts'] + 1) * 10
    #         #     else:
    #         #         size = count_scatter_data['counts'] * 10
    #         #     ax.scatter(
    #         #         count_scatter_data[self.spatial_covariates[0]], count_scatter_data[self.spatial_covariates[1]], 
    #         #         s=size, edgecolor='k', c='g', alpha=0.8,
    #         #         transform=ccrs.PlateCarree(), zorder=101, label='Observed Counts'
    #         #     )
    #         #     # Annotate scatter points with their size
    #         #     for _, row in count_scatter_data.iterrows():
    #         #         if row['counts'] > 0:
    #         #             ax.text(
    #         #                 row[self.spatial_covariates[0]], row[self.spatial_covariates[1]], 
    #         #                 f"{int(row['counts'])}",  # Annotate with the size value
    #         #                 color='black', fontsize=6, ha='center', va='center', 
    #         #                 transform=ccrs.PlateCarree(), zorder=102
    #         #             )

    #         # if not occurance_scatter_data.empty:
    #         #     ax.scatter(
    #         #         occurance_scatter_data['decimallongitude'], occurance_scatter_data['decimallatitude'], 
    #         #         s=5, c='r', alpha=0.8, transform=ccrs.PlateCarree(), zorder=100, label='Occurrences'
    #         #     )
    #         ax.set_title(title, fontsize=10)
    #     # Shared colorbar for the first three subplots
    #     cbar_counts = fig.colorbar(
    #         heatmaps[0], 
    #         ax=axes[:3], 
    #         location='bottom', 
    #         fraction=0.02,  # Thickness remains small and consistent
    #         pad=0.1,      # Space between plots and colorbar
    #         aspect=60      # Controls the aspect ratio (makes it wider)
    #     )
    #     cbar_counts.set_label(f'{colorbar_label} - Number of Occurrences', fontsize=8)

    #     # Colorbar for the fourth subplot (Probability)
    #     cbar_prob = fig.colorbar(
    #         heatmaps[3], 
    #         ax=axes[3], 
    #         location='bottom', 
    #         fraction=0.02, 
    #         pad=0.1
    #     )
    #     cbar_prob.set_label('Probability of ≥1 Occurrence', fontsize=8)

    #     # Add boxes around the first three subplots and the fourth subplot, including colorbars
    #     x_padding = 0.035  # Add padding around the boxes
    #     y_padding = 0.07  # Add padding to account for colorbars
    #     bbox1 = Bbox.union([ax.get_position() for ax in axes[:3]])
    #     bbox2 = axes[3].get_position()
        
    #     # Adjust boxes with padding
    #     box1 = Rectangle(
    #         (bbox1.x0 - x_padding, bbox1.y0 - y_padding - 0.05), 
    #         bbox1.width + 1.3 * x_padding, bbox1.height + 2*y_padding + 0.05,  # Extra height for the colorbar
    #         transform=fig.transFigure, color="black", linewidth=2, fill=False
    #     )
    #     box2 = Rectangle(
    #         (bbox2.x0 - x_padding, bbox2.y0 - y_padding - 0.05), 
    #         bbox2.width + 1.3 * x_padding, bbox2.height + 2*y_padding + 0.05,  # Extra height for the colorbar
    #         transform=fig.transFigure, color="black", linewidth=2, fill=False
    #     )
    #     fig.patches.extend([box1, box2])

    #     title_ax.axis('off')
    #     title_ax.text(
    #         0.5, 0.5,  # Adjust horizontal and vertical position
    #         f'Predictions for {self.identifier.capitalize()} in {self.location.capitalize()} - {self.temporal_covariates[0].capitalize()} {self.t_pred[step_num]}',  # Add title with timestep
    #         fontsize=20,
    #         fontweight='bold',
    #         ha='center',  # Center align text
    #         va='center',
    #     )

    #     # Create custom legend handles
    #     legend_elements = [
    #         Patch(facecolor='lightgray', edgecolor='black', label='Land'),
    #         Patch(facecolor='lightblue', edgecolor='none', label='Ocean'),
    #         Line2D([0], [0], color='black', linestyle='--', label='Borders'),
    #         Line2D([0], [0], color='black', linewidth=0.8, label='Coastline'),
    #         Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.7, label='Rivers'),
    #         Patch(facecolor='lightblue', edgecolor='blue', label='Lakes'),
    #         Line2D([0], [0], color='black', linestyle='-', linewidth=0.5, label='States'),
    #     ]

    #     # Add the legend to legend_ax
    #     legend_ax.legend(
    #         handles=legend_elements,
    #         loc='lower center',  # Position the legend at the top
    #         ncol=7,  # Arrange in 4 columns for horizontal layout
    #         frameon=True,  # Remove legend frame
    #         fontsize=12
    #     )

    #     # Hide the legend_ax borders and ticks (optional, for a cleaner look)
    #     legend_ax.axis('off')

    #     categorized_metrics = self._check_metrics()

    #     metrics_ax.axis('off')

    #     # Utility function for styling text with color categories
    #     def style_metric_with_marker_side_by_side(name, value, category):
    #         # Define the markers and their associated colors
    #         color_marker = {'green': '✔', 'orange': '⚠', 'red': '✘'}.get(category, '')
    #         return f"{name}: {value:.2f} ({color_marker})"  # Add marker below the text

    #     # Create plain text for metrics with markers side by side
    #     metric_list = [
    #         f"Mean Absolute Error: {self.mae:.2f} ",  # No marker, but align format
    #         f"Root Mean Squared Error: {self.rmse:.2f} ",  # No marker, but align format
    #         style_metric_with_marker_side_by_side("Normalised RMSE", self.nrmse, categorized_metrics['nrmse']),
    #         style_metric_with_marker_side_by_side("Training R²", self.training_r2, categorized_metrics['train_r2']),
    #         style_metric_with_marker_side_by_side("Testing R²", self.testing_r2, categorized_metrics['test_r2']),
    #         f"Log Likelihood: {self.av_log_likelihood:.2f} "  # No marker, but align format
    #     ]

    #     # Combine metrics with pipes and line breaks for markers
    #     plain_text = " | ".join(metric_list)

    #     # Render the plain text in the metrics axis
    #     metrics_ax.text(
    #         0.5, 0.5,  # Adjust horizontal and vertical position
    #         plain_text,
    #         fontsize=12,
    #         ha='center',  # Center align text
    #         va='center',
    #         transform=metrics_ax.transAxes,  # Use axes coordinates for positioning
    #         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")  # Add a box around the text
    #     )

    #     return fig
    
    # def _check_metrics(self):
    #     """
    #     Categorize model performance metrics into 'green', 'orange', or 'red'
    #     based on defined bounds for good, medium, and bad performance.
    #     """
    #     # Get the metrics to evaluate
    #     metrics = {
    #         'nrmse': self.nrmse,
    #         'train_r2': self.training_r2,
    #         'test_r2': self.testing_r2
    #     }

    #     # Categorize metrics
    #     categorized_metrics = {}
    #     for metric, value in metrics.items():
    #         if self.metric_bounds[metric]['orange'] >= self.metric_bounds[metric]['green']:
    #             # Small values are better
    #             if value < self.metric_bounds[metric]['green']:
    #                 categorized_metrics[metric] = 'green'
    #             elif value < self.metric_bounds[metric]['orange']:
    #                 categorized_metrics[metric] = 'orange'
    #             else:
    #                 categorized_metrics[metric] = 'red'
    #         else:
    #             # Large values are better
    #             if value > self.metric_bounds[metric]['green']:
    #                 categorized_metrics[metric] = 'green'
    #             elif value > self.metric_bounds[metric]['orange']:
    #                 categorized_metrics[metric] = 'orange'
    #             else:
    #                 categorized_metrics[metric] = 'red'

    #     return categorized_metrics
