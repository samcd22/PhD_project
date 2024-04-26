import pandas as pd
import numpyro
from typing import Union
LikelihoodInput = Union[str, int, float]


class Likelihood:
    """
    A class representing the Likelihood function used to describe the relationship between the modeled and measured data.
    
    Attributes:
     - likelihood_params (pd.Series): A pandas Series object to store the likelihood parameters.
     - likelihood_select (str): The selected likelihood function.
    
    Methods:
     - __init__(self, likelihood_select: str): Initializes the Likelihood class.
     - add_likelihood_param(self, name: str, val: LikelihoodInput) -> 'Likelihood': Saves a named parameter to the Likelihood class before generating the likelihood function.
     - get_likelihood_function(self) -> callable: Generates the selected likelihood function using the likelihood parameters.
    """
    def __init__(self, likelihood_select: str):
        """
        Initializes the Likelihood class.
        
        Args:
        -  likelihood_select (str): The selected likelihood function. Options are:
            - 'gaussian_fixed_sigma': Gaussian likelihood function without estimating sigma.
            - 'gaussian': Gaussian likelihood function.
            - 'gamma_fixed_sigma': Gamma likelihood function without estimating sigma.
            - 'gamma': Gamma likelihood function.
        """
        self.likelihood_params = pd.Series({}, dtype='float64')
        self.likelihood_select = likelihood_select

    def _alpha(self, mu: float, sigma: float) -> float:
        """
        Utility function for converting mean and standard deviation to alpha in a Gamma distribution.
        
        Args:
            mu (float): Mean value.
            sigma (float): Standard deviation value.
        
        Returns:
            float: The alpha value.
        """
        return mu**2 / sigma**2

    
    def _beta(self, mu: float, sigma: float) -> float:
        """
        Utility function for converting mean and standard deviation to beta in a Gamma distribution.
        
        Args:
            mu (float): Mean value.
            sigma (float): Standard deviation value.
        
        Returns:
            float: The beta value.
        """
        return mu / sigma**2

    
    def add_likelihood_param(self, name: str, val: LikelihoodInput) -> 'Likelihood':
        """
        Saves a named parameter to the Likelihood class before generating the likelihood function.
        
        Args:
            name (str): The name of the parameter.
            val (int or float): The value of the parameter.
        
        Returns:
            Likelihood: The Likelihood object.
        """
        self.likelihood_params[name] = val
        return self

    
    def get_likelihood_function(self) -> callable:
        """
        Generates the selected likelihood function using the likelihood parameters.
        
        Returns:
            function: The selected likelihood function.
        """

        def _gaussian_likelihood_fixed_sigma(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Gaussian likelihood function without estimating sigma.
            
            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.
                
            Returns:
                numpyro.distributions.Normal: Gaussian likelihood distribution.
            """
            return numpyro.distributions.Normal(mu, self.likelihood_params.sigma)
        
        def _gaussian_likelihood(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Gaussian likelihood function.
            
            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.
                
            Returns:
                numpyro.distributions.Normal: Gaussian likelihood distribution.
            """
            return numpyro.distributions.Normal(mu, params['sigma'])

        def _gamma_likelihood_fixed_sigma(mu: float, params: dict) -> numpyro.distributions.Gamma:
            """
            Gamma likelihood function without estimating sigma.
            
            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.
                
            Returns:
                numpyro.distributions.Gamma: Gamma likelihood distribution.
            """
            return numpyro.distributions.Gamma(self._alpha(mu, self.likelihood_params.sigma), 1/self._beta(mu, self.likelihood_params.sigma))
        
        def _gamma_likelihood(mu: float, params: dict) -> numpyro.distributions.Gamma:
            """
            Gamma likelihood function.
            
            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.
                
            Returns:
                numpyro.distributions.Gamma: Gamma likelihood distribution.
            """
            sigma = params['sigma']
            return numpyro.distributions.Gamma(self._alpha(mu, sigma), 1/self._beta(mu, sigma))
        
        if self.likelihood_select == 'gaussian_fixed_sigma':
            return _gaussian_likelihood_fixed_sigma        
        elif self.likelihood_select == 'gaussian':
            return _gaussian_likelihood
        elif self.likelihood_select == 'gamma_fixed_sigma':
            return _gamma_likelihood_fixed_sigma
        elif self.likelihood_select == 'gamma':
            return _gamma_likelihood
        else:
            raise ValueError('Invalid likelihood function selected!')