import jax.numpy as jnp
import pandas as pd
from numpyencoder import NumpyEncoder
import sympy as sp
import re

from typing import Union, List
import json
import os
ModelInput = Union[str, int, float]

class Model:
    """
    A class representing the Model used for predicting a system's behaviour. The parameters of this model are optimised using Bayesian Inference. The Sampler class requires a Model object to generate the sampled posterior.

    Args:
        - model_select (str): The selected model. Default options are:
            - 'line': y = m * x + c.
            - 'curve': y = m * x^2 + c.
            - Add more models using the add_model function.

    Attributes:
        - fixed_model_params (pd.Series): A pandas Series object to store model parameters.
        - model_select (str): The selected model. This is the key used to select the model from the models.json file.
        - model_func_string (str): The model function string.
        - independent_variables (list): A list of independent variable names.
        - dependent_variables (list): A list of dependent variable names.
        - all_param_names (list): A list of all parameter names.
    """

    def __init__(self, model_select: str):
        """
        Initializes the Model class.

        Args:
            - model_select (str): The selected model. Options are:
                - 'line': y = m * x + c.
                - 'curve': y = m * x^2 + c.
                - Add more models using the add_model function.
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
        
        self.model_string_expr = models_data[self.model_select]["model"]
        self.independent_variables = models_data[self.model_select]["independent_variables"]
        self.dependent_variables = models_data[self.model_select]["dependent_variables"]
        self.all_param_names = models_data[self.model_select]["all_param_names"]


        self.variables = {}

        for param in self.independent_variables:
            self.variables[param] = sp.symbols(param)
        for param in self.all_param_names:
           self. variables[param] = sp.symbols(param)

        # Define the mathematical words to be used in sympify
        mathematical_words = {
            'pi': sp.pi, 
            'exp': sp.exp, 
            'log': sp.log, 
            'log10': lambda x: sp.log(x, 10),
            'sqrt': sp.sqrt, 
            'sin': sp.sin, 
            'cos': sp.cos, 
            'tan': sp.tan, 
            'asin': sp.asin, 
            'acos': sp.acos, 
            'atan': sp.atan, 
            'sinh': sp.sinh, 
            'cosh': sp.cosh, 
            'tanh': sp.tanh,
            'asinh': sp.asinh, 
            'acosh': sp.acosh, 
            'atanh': sp.atanh
        }

        # Convert the string to a SymPy expression
        try:
            self.sum_expr = sp.sympify(self.model_string_expr, locals={**self.variables, **mathematical_words})
        except (sp.SympifyError, ValueError) as e:
            raise Exception('Model - invalid model string!')
   

    def add_fixed_model_param(self, name: str, val) -> 'Model':
        """
        Adds a named parameter to the Model class. This parameter is fixed and not optimised during the inference process.

        Args:
            - name (str): The name of the parameter.
            - val: The value of the parameter.

        """
        self.fixed_model_params[name] = val
        return self

    def get_model(self) -> callable:
        """
        Generates the selected model function using the model parameters. The model function is used to predict the system's behaviour.

        Returns:
            - callable: The model function. This function takes the list of infered model parameters and a value list of independent variables as inputs and returns the predicted dependent variable values.
        
        """
        
        inference_param_vars = [self.variables[param] for param in self.all_param_names if param not in self.fixed_model_params]

        self.sum_expr = self.sum_expr.subs([(self.variables[param], self.fixed_model_params[param]) for param in self.fixed_model_params.index])
        a = [*[self.variables[indep_var] for indep_var in self.independent_variables], *inference_param_vars]
        self.expr_func = sp.lambdify([*[self.variables[indep_var] for indep_var in self.independent_variables], *inference_param_vars], self.sum_expr, modules='jax')        
        def _model_func(inference_model_params, independent_variables):
            ordered_params = [inference_model_params[param] for param in self.all_param_names if param not in self.fixed_model_params]
            ordered_independent_variables = [jnp.array(independent_variables[indep_var]) for indep_var in self.independent_variables]
            return self.expr_func(*ordered_independent_variables, *ordered_params)
        
        return _model_func
    
    def get_construction(self) -> dict:
        """
        Gets the construction parameters. The conctruction parameters includes all of the config information used to construct the model object. It includes:
            - model_select: The selected model.
            - fixed_model_params: The fixed model parameters.
            - independent_variables: A list of independent variable names.
            - dependent_variables: A list of dependent variable names.
            - all_param_names: A list of all parameter names.
            - model_func_string: The model function string.
        """
        construction = {
            'model_select': self.model_select,
            'fixed_model_params': self.fixed_model_params.to_dict(),
            'independent_variables': self.independent_variables,
            'dependent_variables': self.dependent_variables,
            'all_param_names': self.all_param_names,
            'model_func_string': self.model_string_expr
        }
        return construction

def add_model(model_name: str, model_str: str, independent_variables: List[str], dependent_variables: List[str], all_param_names: List[str]):
    """
    Adds a model to the model.json. The model string should be a mathematical expression that can be evaluated using SymPy. The model string should contain the independent variables, dependent variables, and all parameter names. A list of how mathematical expressions in the model string should be formatted is provided below.

    Args:
        - model_name (str): The name of the model. This name should be unique.
        - model_str (str): The model string. This should be a mathematical expression that can be evaluated using SymPy.
        - independent_variables (list): A list of independent variable names.
        - dependent_variables (list): A list of dependent variable names.
        - all_param_names (list): A list of all parameter names.

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
    Deletes a model from model.json using the model name.

    Args:
        - model_name (str): The name of the model. This is the key used to select the model from the models.json file.

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
    