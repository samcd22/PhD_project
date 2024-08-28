import jax.numpy as jnp
import pandas as pd
from numpyencoder import NumpyEncoder
import re

from typing import Union, List
import json
import os
ModelInput = Union[str, int, float]



class Model:
    """
    A class representing the Model used for predicting a system's behaviour. The parameters of this model are optimised using Bayesian Inference

    Attributes:
    - fixed_model_params (pd.Series): A pandas Series object to store model parameters.
    - model_select (str): The selected model.
    - model_func_string (str): The model function string.
    - independent_variables (list): A list of independent variables.
    - dependent_variables (list): A list of dependent variables.
    - all_param_names (list): A list of all parameter names.

    Methods:
    - __init__(self, model_select: str): Initializes the Model class.
    - add_model_param(self, name: str, val: ModelInput) -> 'Model': Adds a named parameter to the Model class.
    - get_model(self) ->: callable Generates the selected model function using the model parameters.
    """

    def __init__(self, model_select: str):
        """
        Initializes the Model class.

        Args:
        - model_select (str): The selected model. Options are:
        """
        self.fixed_model_params = pd.Series({}, dtype='float64')
        self.model_select = model_select

        models_file = os.path.join(os.path.dirname(__file__), "models.json")
        if not os.path.exists(models_file):
            raise Exception('Model - models.json file does not exist!')
        with open(models_file, 'r') as f:
            models_data = json.load(f)

        if model_select not in models_data:
            raise Exception('Model - model does not exist in models.json file! Please add the model first.')
        
        model_func = models_data[self.model_select]["model"]
        self.independent_variables = models_data[self.model_select]["independent_variables"]
        self.dependent_variables = models_data[self.model_select]["dependent_variables"]
        self.all_param_names = models_data[self.model_select]["all_param_names"]

        mathematical_words = ['pi', 'exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 
                     'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
                     'arcsinh', 'arccosh', 'arctanh', 'abs', 'ceil', 'floor']

        for word in mathematical_words:
            model_func = re.sub(r'\b'+word+r'\b', 'jnp.'+word, model_func)

        for i in self.all_param_names:
            model_func = re.sub(r'\b'+i+r'\b', 'self.'+i, model_func)
        self.model_func_string = model_func        

    def add_fixed_model_param(self, name: str, val) -> 'Model':
        """
        Adds a named parameter to the Model class.

        Args:
        - name (str): The name of the parameter.
        - val: The value of the parameter.

        Returns:
        - self: The Model object.
        """
        self.fixed_model_params[name] = val
        return self
        
    def _set_params(self, inference_model_params):
        """
        Set the parameters of the model.

        Args:
            inference_model_params (dict): A dictionary containing the inference model parameters.

        Raises:
            Exception: If any required parameter is missing for the model.

        """
        self.inference_params = inference_model_params
        for param_name in self.all_param_names:
            if param_name in inference_model_params:
                setattr(self, param_name, inference_model_params[param_name])
            elif param_name in self.fixed_model_params:
                setattr(self, param_name, self.fixed_model_params[param_name]) 
            else:
                raise Exception('Model - missing parameters for the model!')
            

    def get_model(self):
        """
        Generates the selected model function using the model parameters.

        Returns:
        - _model_func: The model function.
        """
        def _model_func(inferance_model_params, independent_variables):
            self._set_params(inferance_model_params)
            if len(independent_variables) != len(self.independent_variables):
                raise Exception('Model - incorrect number of independent variables for this model!')
            for i in range(len(self.independent_variables)):
                globals()[self.independent_variables[i]] = independent_variables[i]

            return eval(self.model_func_string)
        
        return _model_func
        
def add_model(model_name: str, model_str: str, independent_variables: List[str], dependent_variables: List[str], all_param_names: List[str]):
    """
    Adds a model to the model.json.

    Args:
    - model_name (str): The name of the model.
    - model_str (str): The model string.
    - independent_variables (list): A list of independent variables.
    - dependent_variables (list): A list of dependent variables.
    - all_param_names (list): A list of all parameter names.

    Raises:
    - Exception: If the model string contains any mathematical words.
    - Exception: If the model already exists in the models.json file but the model string does not match.
    - Exception: If the model already exists in the models.json file.

    Mathematical expressions:
    - pi: The mathematical constant pi.
    - exp: The exponential function.
    - log: The natural logarithm function.
    - log10: The logarithm function (base 10).
    - sqrt: The square root function. 
    - sin: The sine function.
    - cos: The cosine function.
    - tan: The tangent function.
    - arcsin: The inverse sine function.
    - arccos: The inverse cosine function.
    - arctan: The inverse tangent function.
    - sinh: The hyperbolic sine function.
    - cosh: The hyperbolic cosine function.
    - tanh: The hyperbolic tangent function.
    - arcsinh: The inverse hyperbolic sine function.
    - arccosh: The inverse hyperbolic cosine function.
    - arctanh: The inverse hyperbolic tangent function.
    - abs: The absolute value function.
    - ceil: The ceiling function.
    - floor: The floor function.

    """
    mathematical_words = ['pi', 'exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 
                     'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
                     'arcsinh', 'arccosh', 'arctanh', 'abs', 'ceil', 'floor']
    
    models_file = os.path.join(os.path.dirname(__file__), "models.json")
    if os.path.exists(models_file):
        with open(models_file, 'r') as f:
            models_data = json.load(f)
        if model_name in models_data:
            if models_data[model_name] != model_str:
                raise Exception('Model - model already exists under '+model_name+', but model string does not match the model name in models.json')
            else:
                raise Exception('Model - model already exists!')
    else:
        raise Exception('Model - models.json file does not exist!')

    model_func_vars = re.findall(r'\b(?:[a-zA-Z]+\d*\w*|\d+[a-zA-Z]+\w*)\b', model_str)
    model_func_vars = [var for var in model_func_vars if var not in mathematical_words]

    missing_vars = [var for var in model_func_vars if var not in independent_variables and var not in dependent_variables and var not in all_param_names]
    if missing_vars:
        raise Exception(f"Model function contains variables that are not accounted for: {', '.join(missing_vars)}")
    
    models_data[model_name] = {}
    models_data[model_name]['model'] = model_str
    models_data[model_name]["independent_variables"] = independent_variables
    models_data[model_name]["dependent_variables"] = dependent_variables
    models_data[model_name]["all_param_names"] = all_param_names

    with open(models_file, 'w') as f:
        json.dump(models_data, f, cls=NumpyEncoder, separators=(', ', ': '), indent=4)

def delete_model(model_name: str):
    """
    Deletes a model from model.json.

    Args:
    - model_name (str): The name of the model.

    """
    models_file = os.path.join(os.path.dirname(__file__), "models.json")
    if os.path.exists(models_file):
        with open(models_file, 'r') as f:
            models_data = json.load(f)
        if model_name in models_data:
            del models_data[model_name]
            with open(models_file, 'w') as f:
                json.dump(models_data, f, cls=NumpyEncoder, separators=(', ', ': '), indent=4)
        else:
            raise Exception('Model - model does not exist!')
    else:
        raise Exception('Model - models.json file does not exist!')