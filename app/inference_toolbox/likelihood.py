import pandas as pd
import numpyro
from typing import Union

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
            return numpyro.distributions.Normal(mu, self.fixed_likelihood_params.sigma**2)

        def _gaussian_likelihood_percentage_error(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Gaussian likelihood function using percentage error.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """
            return numpyro.distributions.Normal(mu, (mu * params['error'])**2)

        def _gaussian_likelihood(mu: float, params: dict) -> numpyro.distributions.Normal:
            """
            Standard Gaussian likelihood function.

            Args:
                mu (float): Mean value.
                params (dict): Dictionary of parameters.

            """
            return numpyro.distributions.Normal(mu, params['sigma']**2)

        if self.likelihood_select == 'gaussian_fixed_sigma':
            return _gaussian_likelihood_fixed_sigma
        elif self.likelihood_select == 'gaussian_percentage_error':
            return _gaussian_likelihood_percentage_error
        elif self.likelihood_select == 'gaussian':
            return _gaussian_likelihood
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