import pandas as pd
import numpyro
from typing import Union
LikelihoodInput = Union[str, int, float]
import os
import json

class Likelihood:
    """
    A class representing the Likelihood function used to describe the relationship between the modeled and measured data.
    
    Attributes:
     - fixed_likelihood_params (pd.Series): A pandas Series object to store the fixed likelihood parameters.
     - likelihood_select (str): The selected likelihood function.
    
    Methods:
     - __init__(self, likelihood_select: str): Initializes the Likelihood class.
     - add_likelihood_param(self, name: str, val: LikelihoodInput) -> 'Likelihood': Saves a named parameter to the Likelihood class before generating the likelihood function.
     - get_likelihood_function(self) -> callable: Generates the selected likelihood function using the likelihood parameters.
     - get_construction(self): Get the construction parameters of the likelihood.
    """
    def __init__(self, likelihood_select: str):
        """
        Initializes the Likelihood class.
        
        Args:
        -  likelihood_select (str): The selected likelihood function. Options are:
            - 'gaussian': Gaussian likelihood function.
            - 'gaussian_fixed_sigma': Gaussian likelihood function without estimating sigma.
        """
        self.fixed_likelihood_params = pd.Series({}, dtype='float64')
        self.likelihood_select = likelihood_select  
    
    def get_construction(self):
        """
        Get the construction parameters.
        
        Returns:
            dict: The construction parameters.
        """
        construction = {
            'likelihood_select': self.likelihood_select,
            'fixed_likelihood_params': self.fixed_likelihood_params.to_dict()
        }
        return construction
    
    def add_likelihood_param(self, name: str, val: LikelihoodInput) -> 'Likelihood':
        """
        Saves a named parameter to the Likelihood class before generating the likelihood function.
        
        Args:
        - name (str): The name of the parameter.
        - val (int or float): The value of the parameter.
        
        Returns:
        - Likelihood: The Likelihood object.
        """
        self.fixed_likelihood_params[name] = val
        return self

    
    def get_likelihood_function(self) -> callable:
        """
        Generates the selected likelihood function using the likelihood parameters.
        
        Returns:
        - function: The selected likelihood function.
        """

        def _gaussian_likelihood_fixed_sigma(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Gaussian likelihood function without estimating sigma.
            
            Args:
            - mu (float): Mean value.
            - params (dict): Dictionary of parameters.
                
            Returns:
            - numpyro.distributions.Normal: Gaussian likelihood distribution.
            """
            return numpyro.distributions.Normal(mu, self.fixed_likelihood_params.sigma)
        
        def _gaussian_likelihood(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Gaussian likelihood function.
            
            Args:
            - mu (float): Mean value.
            - params (dict): Dictionary of parameters.
                
            Returns:
            - numpyro.distributions.Normal: Gaussian likelihood distribution.
            """
            return numpyro.distributions.Normal(mu, params['sigma'])
        
        if self.likelihood_select == 'gaussian_fixed_sigma':
            return _gaussian_likelihood_fixed_sigma        
        elif self.likelihood_select == 'gaussian':
            return _gaussian_likelihood
        else:
            raise ValueError('Invalid likelihood function selected!')
        
    def _set_params(self, inference_likelihood_params: dict):
        """
        Set the parameters of the likelihood function.
        
        Args:
        - inference_likelihood_params (dict): A dictionary containing the likelihood parameters.
        
        Raises:
        - Exception: If any required parameter is missing for the likelihood function.
        
        """
        for param_name in self.all_param_names:
            if param_name in inference_likelihood_params:
                setattr(self, param_name, inference_likelihood_params[param_name])
            elif param_name in self.fixed_likelihood_params:
                setattr(self, param_name, self.fixed_likelihood_params[param_name]) 
            else:
                raise Exception('Likelihood - missing parameters for the likelihood function!')