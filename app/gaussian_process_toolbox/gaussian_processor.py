import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from jaxlib.xla_extension import ArrayImpl
from sklearn.gaussian_process.kernels import Sum, Product
from sklearn.gaussian_process.kernels import Kernel as GP_Kernel
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel
from sklearn.preprocessing import StandardScaler
import ast
import json
import joblib
import os


from data_processing.data_processor import DataProcessor
from gaussian_process_toolbox.kernel import Kernel
from gaussian_process_toolbox.transformation import Transformation

class GP:
    """
    A class to represent a Gaussian Process model
    
    Attributes:
        - data_processor (DataProcessor): An instance of the DataProcessor class.
        - kernel (sklearn.gaussian_process.kernels.Kernel): The kernel to use in the Gaussian Process model.
        - kernel_config (dict): A dictionary where keys are kernel instances and values are lists of dimensions they apply to.
        - kernel_params (dict): A dictionary where keys are kernel instances and values are dictionaries of their parameters.
        - uncertainty_method (str): The method to calculate uncertainties.
        - uncertainty_params (dict): Parameters for the uncertainty method.
        - transformation (Transformation): An instance of the Transformation class.
        - root_results_path (str): The root path to save the results.
        - training_data (pd.DataFrame): The training data.
        - testing_data (pd.DataFrame): The testing data.
        - data_construction (dict): The construction of the data.
        - independent_variables (list): The independent variables.
        - dependent_variable (str): The dependent variable.
        - results_path (str): The path to save the results.
        - gp_construction (dict): The construction of the Gaussian Process model.
        - instance (int): The instance number where the data is saved.
        - X_train (np.array): The scaled training independent variables.
        - y_train (np.array): The scaled training dependent variable.
        - X_test (np.array): The scaled testing independent variables.
        - y_test (np.array): The scaled testing dependent variable.
        - y_train_transformed (np.array): The transformed training dependent variable.
        - y_test_transformed (np.array): The transformed testing dependent variable.
        - X_scaler (StandardScaler): The StandardScaler for the independent variables.
        - y_scaler (StandardScaler): The StandardScaler for the dependent variable.
        - X_train_scaled (np.array): The scaled training independent variables.
        - y_train_scaled (np.array): The scaled training dependent variable.
        - X_test_scaled (np.array): The scaled testing independent variables.
        - y_test_scaled (np.array): The scaled testing dependent variable.
        - gp_model (GaussianProcessRegressor): The Gaussian Process model.
        - trained (bool): Whether the model has been trained.
        - optimized_params (dict): The optimized parameters of the model.
        - num_params (int): The number of parameters in the model.

    Methods:

        - get_construction(): Get the construction of the Gaussian Process model.
        - train(): Train the Gaussian Process model on the training data.
        - predict(points): Predict the dependent variable at given points.
    """

    def __init__(self, 
                 data_processor,
                 kernel: Kernel,
                 uncertainty_method = None,
                 uncertainty_params = None,
                 transformation: Transformation = Transformation('identity'),
                 root_results_path = '/results/gaussian_process_results'):

        """
        Initialize the Gaussian Process model.

        Args:
            - data_processor (DataProcessor): An instance of the DataProcessor class.
            - kernel (Kernel): An instance of the Kernel class.
            - uncertainty_method (str): The method to calculate uncertainties. Default is None.
            - uncertainty_params (dict): Parameters for the uncertainty method. Default is None.
            - transformation (Transformation): An instance of the Transformation class. Default is 'identity'.
            - root_results_path (str): The root path to save the results. Default is '/results/gaussian_process_results'.
        """

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

        self.transformation = transformation

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

        self.y_train_transformed = self.transformation.transform(self.y_train)
        self.y_test_transformed = self.transformation.transform(self.y_test)

        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.X_train_scaled = self.X_scaler.fit_transform(self.X_train)
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train_transformed.reshape(-1, 1)).flatten()

        self.X_test_scaled = self.X_scaler.transform(self.X_test)
        self.y_test_scaled = self.y_scaler.transform(self.y_test_transformed.reshape(-1, 1)).flatten()

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
        """
        Get the construction of the Gaussian Process model. The construction parameters include all of the config information needed to construct the Gaussian Process model. It includes:
            - kernel: The construction of the kernel.
            - data_processor: The construction of the data.
            - uncertainty_method: The method to calculate uncertainties.
            - uncertainty_params: Parameters for the uncertainty method.
            - transformation: The type of transformation applied to the dependent variable.

        Returns:
            dict: The construction of the Gaussian Process model.
        """

        construction = {
            'kernel': self.kernel_obj.get_construction(),
            'data_processor': self.data_construction,
            'uncertainty_method': self.uncertainty_method,
            'uncertainty_params': self.uncertainty_params,
            'transformation': self.transformation.transformation_type
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

                if self._convert_tuples_to_lists(self._convert_arrays_to_lists(self.get_construction())) == self._convert_tuples_to_lists(instance_hyperparams):
                    return int(instance_folder.split('_')[1])

        instances = [int(folder.split('_')[1]) for folder in instance_folders if '_' in folder]
        return 1 if not instances else min(i for i in range(1, max(instances) + 2) if i not in instances)

    def _get_uncertainties(self):
        """
        Calculate the uncertainties of the Gaussian Process model.

        Returns:
            np.array: An array containing the uncertainties of the model.
        """

        if self.uncertainty_method is None:
            return 1e-6*np.ones_like(self.y_train)

        if self.uncertainty_method == 'precision':
            uncertainty = self._precision_uncertainties()
        elif self.uncertainty_method == 'constant':
            uncertainty = self._constant_uncertainties()
        elif self.uncertainty_method == 'XYLO_uncertainty':
            uncertainty = self._xylo_uncertainties()
        else:
            raise ValueError(f'Unsupported uncertainty method: {self.uncertainty_method}')
        transformed_uncertainty = self.transformation.transform_alpha(self.y_train, uncertainty)
        scaled_uncertainty = transformed_uncertainty / self.y_scaler.scale_[0]**2
        return scaled_uncertainty
    
    def _precision_uncertainties(self):
        """
        Calculate the precision uncertainties of the Gaussian Process model.
        """
        precision_error = self.uncertainty_params['precision_error']
        uncertainties = self.training_data*precision_error
        return uncertainties
    
    def _constant_uncertainties(self):
        """
        Calculate the constant uncertainties of the Gaussian Process model.
        """
        uncertainties = self.uncertainty_params['constant_error']
        return uncertainties

    def _xylo_uncertainties(self):
        """
        Calculate the XYLO uncertainties of the Gaussian Process model.
        """
        if 'precision_error' not in self.uncertainty_params or 'zero_count_error' not in self.uncertainty_params:
            raise ValueError('Precision error and zero count error must be provided for XYLO uncertainty')

        precision_error = self.uncertainty_params['precision_error']
        zero_count_error = self.uncertainty_params['zero_count_error']
        uncertainties = np.zeros_like(self.y_train)
        for i in range(len(self.y_train)):
            if self.y_train[i] == 0:
                uncertainties[i] = zero_count_error
            else:
                uncertainties[i] = precision_error*self.y_train[i]
        return uncertainties

    def train(self):
        """
        Train the Gaussian Process model on the training data.
        """
        gp_model_file = os.path.join(self.results_path, f'instance_{self.instance}', 'gaussian_process_model.pkl')
        if os.path.exists(gp_model_file):
            print(f'Loading existing GP model from {gp_model_file}')
            self.gp_model = joblib.load(gp_model_file)
            self.trained = True
        else:
            uncertainties = self._get_uncertainties()
            self.gp_model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha = uncertainties)
            self.gp_model.fit(self.X_train_scaled, self.y_train_scaled)
            print(f'Fitted new GP model and saving to {gp_model_file}')
            self._save_run(self.gp_model)
            self.trained = True

        results_file = os.path.join(self.results_path, f'instance_{self.instance}', 'results.json')
        results = self._get_training_results()
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
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

    def _get_training_results(self):
        if not self.trained:
            raise ValueError("Model has not been trained yet")

        self.optimized_params = self._extract_optimized_kernel_params(self.gp_model.kernel_)
        self.num_params = self._count_kernel_parameters()

        training_r2 = self.gp_model.score(self.X_train_scaled, self.y_train_scaled)
        testing_r2 = self.gp_model.score(self.X_test_scaled, self.y_test_scaled)
        
        y_pred, y_std, _, _ = self.predict(self.X_test)

        # Calculate the mean absolute error
        mae = np.mean(np.abs(self.y_test - y_pred))

        # Calculate the root mean squared error
        rmse = np.sqrt(np.mean((self.y_test - y_pred)**2))

        # Calculate normalised root mean squared error
        nrmse = rmse / np.std(self.y_test)

        # Calculate log likelihood of test data
        log_likelihoods = (
            -0.5 * np.log(2 * np.pi * y_std**2)
            - ((self.y_test - y_pred) ** 2) / (2 * y_std**2)
        )

        # Total log-likelihood
        av_log_likelihood = np.mean(log_likelihoods) 

        # Serialize results to JSON
        results = {
            'fitted_kernel_params': self.optimized_params,
            'num_params': self.num_params,
            'training_r2': training_r2,
            'testing_r2': testing_r2,
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'av_log_likelihood': av_log_likelihood
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
            mean_preds (array-like): The mean predictions at the given points.
            std_preds (array-like): The standard deviation of the predictions at the given points.
            lower_preds (array-like): The lower bound of the predictions at the given points.
            upper_preds (array-like): The upper bound of the predictions at the given points.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")

        points_scaled = self.X_scaler.transform(points)

        mean_preds_scaled, std_preds_scaled = self.gp_model.predict(points_scaled, return_std=True)
        mean_preds_transformed = self.y_scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
        std_preds_transformed = std_preds_scaled * self.y_scaler.scale_[0]  # Only multiply by standard deviation

        lower_preds_transformed = mean_preds_transformed - 1.96 * std_preds_transformed
        upper_preds_transformed = mean_preds_transformed + 1.96 * std_preds_transformed

        lower_preds = self.transformation.inverse_transform(lower_preds_transformed)
        upper_preds = self.transformation.inverse_transform(upper_preds_transformed)

        mean_preds, std_preds = self.transformation.inverse_transform(mean_preds_transformed, std_preds_transformed)

        return mean_preds, std_preds, lower_preds, upper_preds
        