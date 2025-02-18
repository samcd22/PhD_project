import pandas as pd
import numpyro
from typing import Union
import jax.numpy as jnp
from inference_toolbox.tr_norm import TruncatedNormal

LikelihoodInput = Union[str, int, float]

class Likelihood:
    """
    A class representing the Likelihood function used to describe the relationship between the modeled and measured data. The likelihood function is used to estimate the parameters of the model, and is a key component of the Bayesian inference process. The Sampler class requires a Likelihood object to generate the sampled posterior.

    Args:
        - likelihood_select (str): The selected likelihood function. Options are:
            - 'gaussian': Gaussian likelihood function.
            - 'gaussian_fixed_sigma': Gaussian likelihood function without estimating sigma.
            - 'gaussian_percentage_error': Gaussian likelihood function using percentage error.
            - Other likelihood functions can be added as needed.

    Attributes:
        - fixed_likelihood_params (pd.Series): A pandas Series object to store the fixed likelihood parameters.
        - likelihood_select (str): The selected likelihood function.
    """

    def __init__(self, likelihood_select: str):
        """
        Initializes the Likelihood class.

        Args:
            - likelihood_select (str): The selected likelihood function. Options are:
                - 'gaussian': Gaussian likelihood function.
                - 'gaussian_fixed_sigma': Gaussian likelihood function without estimating sigma.
                - 'gaussian_percentage_error': Gaussian likelihood function using percentage error.
                - Other likelihood functions can be added as needed.
        """
        self.fixed_likelihood_params = pd.Series({}, dtype='float64')
        self.likelihood_select = likelihood_select

    def get_construction(self) -> dict:
        """
        Gets the construction parameters. The conctruction parameters includes all of the config information used to construct the likelihood object. It includes:
            - likelihood_select: The selected likelihood function.
            - fixed_likelihood_params: The fixed likelihood parameters.

        Returns:
            - dict: A dictionary containing the construction parameters.
        """
        construction = {
            'likelihood_select': self.likelihood_select,
            'fixed_likelihood_params': self.fixed_likelihood_params.to_dict()
        }
        return construction

    def add_likelihood_param(self, name: str, val: LikelihoodInput) -> 'Likelihood':
        """
        Saves a named parameter to the Likelihood class before generating the likelihood function. This parameter is fixed and not estimated during the inference process.

        Args:
            - name (str): The name of the parameter.
            - val (str, int, float): The value of the parameter, can be an int, float, or str.
        """
        self.fixed_likelihood_params[name] = val
        return self

    def get_likelihood_function(self) -> callable:
        """
        Generates the selected likelihood function using the likelihood parameters.

        """
        def _gaussian_likelihood_fixed_sigma(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Gaussian likelihood function without estimating sigma.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """
            if 'sigma' not in self.fixed_likelihood_params:
                raise ValueError('Likelihood - fixed sigma parameter not set!')
            return numpyro.distributions.Normal(mu, self.fixed_likelihood_params.sigma)

        def _gaussian_likelihood_percentage_error(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Gaussian likelihood function using percentage error.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """

            try :
                return numpyro.distributions.Normal(mu, jnp.abs(mu) * params['error'])
            except:
                print('Error in Gaussian likelihood function')
                print('mu:', mu)
                print('params:', params)

            return numpyro.distributions.Normal(mu, jnp.abs(mu) * params['error'])

        def _gaussian_likelihood(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Standard Gaussian likelihood function.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """
            return numpyro.distributions.Normal(mu, params['sigma'])
        
        def _gamma_likelihood(mu: float, params: dict) -> numpyro.distributions.Gamma:
            """
            Gamma likelihood function.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """  
            alpha, beta = mean_to_alpha_beta(mu, params['sigma'])
            return numpyro.distributions.Gamma(alpha, beta)
        
        def _gamma_likelihood_fixed_sigma(mu: float, params: dict) -> numpyro.distributions.Gamma:
            """
            Gamma likelihood function without estimating sigma.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """    
            if 'sigma' not in self.fixed_likelihood_params:
                raise ValueError('Likelihood - fixed sigma parameter not set!')
            alpha, beta = mean_to_alpha_beta(mu, self.fixed_likelihood_params.sigma)
            return numpyro.distributions.Gamma(alpha, beta)

        def _gamma_likelihood_percentage_error(mu: float, params: dict) -> numpyro.distributions.Gamma:
            """
            Gamma likelihood function using percentage error.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """    
            alpha, beta = mean_to_alpha_beta(mu, mu * params['error'])
            return numpyro.distributions.Gamma(alpha, beta)
        
        def _truncate_normal_likelihood_fixed_sigma(mu: float, params: dict) -> TruncatedNormal:
            """
            Truncated normal likelihood function.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """    
            return TruncatedNormal(mu, self.fixed_likelihood_params.sigma)
        
        def _truncate_normal_likelihood(mu: float, params: dict) -> TruncatedNormal:
            """
            Truncated normal likelihood function without estimating sigma.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """    
            return TruncatedNormal(mu, params['sigma'])
        
        def _truncate_normal_likelihood_percentage_error(mu: float, params: dict) -> TruncatedNormal:
            """
            Truncated normal likelihood function using percentage error.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """    
            return TruncatedNormal(mu, mu * params['error'])
        
        def uniform_likelihood(mu: float, params: dict) -> numpyro.distributions.Uniform:
            """
            Uniform likelihood function.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """
            return numpyro.distributions.Uniform(mu - params['sigma']*jnp.sqrt(12)/2, mu + params['sigma']*jnp.sqrt(12)/2)

        def uniform_likelihood_fixed_sigma(mu: float, params: dict) -> numpyro.distributions.Uniform:
            """
            Uniform likelihood function without estimating sigma.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """
            return numpyro.distributions.Uniform(mu - self.fixed_likelihood_params.sigma*jnp.sqrt(12)/2, mu + self.fixed_likelihood_params.sigma*jnp.sqrt(12)/2)
        
        def uniform_likelihood_percentage_error(mu: float, params: dict) -> numpyro.distributions.Uniform:
            """
            Uniform likelihood function using percentage error.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """              
            return numpyro.distributions.Uniform(mu - mu*params['error']*jnp.sqrt(12)/2, mu + mu*params['error']*jnp.sqrt(12)/2)

        def mean_to_alpha_beta(mean, sigma):
            alpha = mean ** 2 / sigma**2
            beta = mean / sigma**2
            return alpha, beta

        if self.likelihood_select == 'gaussian_fixed_sigma':
            return _gaussian_likelihood_fixed_sigma
        elif self.likelihood_select == 'gaussian_percentage_error':
            return _gaussian_likelihood_percentage_error
        elif self.likelihood_select == 'gaussian':
            return _gaussian_likelihood
        elif self.likelihood_select == 'gamma_fixed_sigma':
            return _gamma_likelihood_fixed_sigma
        elif self.likelihood_select == 'gamma':
            return _gamma_likelihood
        elif self.likelihood_select == 'gamma_percentage_error':
            return _gamma_likelihood_percentage_error
        elif self.likelihood_select == 'truncated_normal_fixed_sigma':
            return _truncate_normal_likelihood_fixed_sigma
        elif self.likelihood_select == 'truncated_normal':
            return _truncate_normal_likelihood
        elif self.likelihood_select == 'truncated_normal_percentage_error':
            return _truncate_normal_likelihood_percentage_error
        elif self.likelihood_select == 'uniform_fixed_sigma':
            return uniform_likelihood_fixed_sigma
        elif self.likelihood_select == 'uniform':
            return uniform_likelihood
        elif self.likelihood_select == 'uniform_percentage_error':
            return uniform_likelihood_percentage_error
        else:
            raise ValueError('Invalid likelihood function selected!')

    def _set_params(self, inference_likelihood_params: dict):
        """
        Set the parameters of the likelihood function.

        Args:
            inference_likelihood_params (dict): A dictionary containing the likelihood parameters.

        """
        for param_name in self.all_param_names:
            if param_name in inference_likelihood_params:
                setattr(self, param_name, inference_likelihood_params[param_name])
            elif param_name in self.fixed_likelihood_params:
                setattr(self, param_name, self.fixed_likelihood_params[param_name])
            else:
                raise Exception('Likelihood - missing parameters for the likelihood function!')
