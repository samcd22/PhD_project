import numpy as np
import pandas as pd
import numpyro
import jax.numpy as jnp
from jax import random
from scipy.optimize import newton, root_scalar
from numpyro import distributions
from typing import Union, Optional, List

class Parameter:
    """
    A class representing an inferred parameter of the model by the sampler.

    Attributes:
    - prior_params (pd.Series): A series to store the prior hyperparameters.
    - name (str or list): The name or list of names of the parameter/joint parameters.
    - prior_select (str): The type of prior distribution to use.
    - multivar (bool): Indicates whether the parameter is multivariate or not.
    - multi_mode (bool): Indicates whether the parameter is multimodal or not.
    - joined_name (str): The combined name of the parameter/joint parameters.
    - order (int): The order of magnitude of the parameter - input hyperparameter values generally between 0.1 and 10, then specify the order of magnitude using "order".
    - n_dims (int): The number of dimensions of the parameter.
    
    Methods:
    - add_prior_param: Saves a named prior hyperparameter to the Parameter class before generating the parameter's prior distribution.
    - get_prior_function: Generates the selected prior distribution using the prior hyperparameters.
    - sample_param: Generates a sample for this parameter.
    """
    def __init__(self, name: Union[str, List[str]], prior_select: str = "gaussian", order: int = 0, multi_mode: Optional[bool] = False):
        """
        Initializes the Parameter class.

        Args:
        - name (str or list): The name(s) of the parameter.
        - prior_select (str, optional): The type of prior distribution to use. Defaults to "gaussian". Options are:
            - "gaussian": Gaussian distribution 
                - Can handle single and multivariate cases.
                - Can handle single and multimodal cases.
                - Requires 'mu' and 'sigma' hyperparameters.
            - "gamma": Gamma distribution 
                - Can only handle single variate cases.
                - Can only handle single mode cases
                - Requires either 'mu' and 'sigma' or 'alpha' and 'beta' hyperparameters.
            - "uniform": Uniform distribution.
                - Can only handle single variate cases.
                - Can only handle single mode cases
                - Requires 'low' and 'high' hyperparameters.
            - "log_norm": Log-normal distribution.
                - Can handle single and multivariate cases.
                - Can handle single and multimodal cases.
                - Requires either 'mu' and 'sigma' or 'loc' and 'scale' hyperparameters.

        - order (int, optional): The order of the parameter. Defaults to 0.
        - multi_mode (bool, optional): Indicates whether the parameter is multimodal or not. Defaults to False.
        """

        if not isinstance(name, (str, list)):
            raise TypeError("name must be a string or a list")
        if not isinstance(prior_select, str):
            raise TypeError("prior_select must be a string")
        if not isinstance(order, (int, float)):
            raise TypeError("order must be an integer")
        if prior_select not in ["gaussian", "gamma", "uniform", "log_norm", "multi_mode_gaussian", "multi_mode_log_norm"]:
            raise ValueError("Invalid prior distribution selected!")
        
        self.prior_params = pd.Series({}, dtype='float64')
        self.name = name
        self.prior_select = prior_select
        self.multivar = False
        self.multi_mode = multi_mode
        self.prior_func = None
        if isinstance(self.name, list) and len(self.name) >= 2:
            self.multivar = True
            self.joined_name = '_and_'.join(self.name)
            self.n_dims = len(self.name)
        else:
            self.joined_name = self.name 
            self.n_dims = 1        
        self.order = order

    def get_construction(self):
        """
        Get the construction parameters of the parameter.

        Returns:
        - dict: The construction parameters.
        """
        construction = {
            'name': self.name,
            'prior_select': self.prior_select,
            'prior_params': self.prior_params.to_dict(),
            'order': self.order,
            'multi_mode': self.multi_mode
        }
        return construction

    def _alpha_gamma(self, mu: Union[float, int, np.ndarray], sigma: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
        """
        Utility function for converting mean and stdev to alpha in a Gamma distribution.

        Args:
        - mu (float or array-like): The mean value(s).
        - sigma (float or array-like): The standard deviation value(s).

        Returns:
        - float or array-like: The alpha value(s).
        """
        if sigma == 0:
            raise ValueError("'Parameter - 'sigma' cannot be zero.")
        
        return mu**2 / sigma**2

    def _beta_gamma(self, mu: Union[float, int, np.ndarray], sigma: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
        """
        Utility function for converting mean and stdev to beta in a Gamma distribution.

        Args:
        - mu (float or array-like): The mean value(s).
        - sigma (float or array-like): The standard deviation value(s).

        Returns:
        - float or array-like: The beta value(s).
        """
        if sigma == 0:
            raise ValueError("'Parameter - 'sigma' cannot be zero.")

        return mu / sigma**2

    def _mean_stdev_to_loc_log_norm(self, mu: Union[float, int, np.ndarray], sigma: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
        """
        Utility function for converting mean and stdev to location in a Log Normal distribution.

        Args:
        - mu (float or array-like): The mean value(s).
        - sigma (float or array-like): The standard deviation value(s).

        Returns:
        - float or array-like: The loc value(s).
        """

        if np.isscalar(mu):
            if mu == 0:
                raise ValueError("Parameter - 'mu' cannot be zero.")
            if sigma == 0:
                raise ValueError("Parameter - 'sigma' cannot be zero.")
            return np.log(mu) - 0.5 * np.log(1 + sigma**2 / mu**2)
        else:
            mu = np.array(mu)
            sigma = np.array(sigma)
            alpha = np.zeros(len(mu))
            for i in range(len(mu)):
                if mu[i] == 0:
                    raise ValueError("Parameter - the elements of 'mu' cannot be zero.")
                if sigma[i, i] == 0:
                    raise ValueError("Parameter - the diagonal elements of 'sigma' cannot be zero.")
                alpha[i] = np.log(mu[i]) - 1 / 2 * np.log(sigma[i, i]**2 / mu[i]**2 + 1)
            return np.array(alpha)

    def _mean_stdev_to_scale_log_norm(self, mu: Union[float, int, np.ndarray], sigma: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
        """
        Utility function for converting mean and stdev to scale in a Log Normal distribution.

        Args:
        - mu (float or array-like): The mean value(s).
        - sigma (float or array-like): The standard deviation value(s).

        Returns:
        - float or array-like: The scale value(s).
        """
        if np.isscalar(mu):
            return np.sqrt(np.log(1 + sigma**2 / mu**2))
        else:
            mu = np.array(mu)
            sigma = np.array(sigma)
            beta = np.zeros((len(mu), len(mu)))
            for i in range(len(mu)):
                for j in range(len(mu)):
                    beta[i, j] = np.log(sigma[i, j]**2 / (mu[i] * mu[j]) + 1)
            return beta
        
    def _peak_to_loc(self, peak: Union[float, int, np.ndarray], scale: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
        """
        Utility function for converting peak to location in a Log Normal distribution.

        Args:
        - peak (float or array-like): The peak value(s).

        Returns:
        - float or array-like: The loc value(s).
        """
        if np.isscalar(peak):
            return np.log(peak) + scale
        
        else:
            peak = np.array(peak)
            loc = np.zeros(len(peak))
            if np.isscalar(scale):
                for i in range(len(peak)):
                    loc[i] = np.log(peak[i]) + scale
                return loc
            else:
                scale = np.array(scale)
                for i in range(len(peak)):
                    loc[i] = np.log(peak[i]) + scale[i,i]
                return loc

    def add_prior_param(self, name: str, val: Union[float, int, np.ndarray, list]) -> 'Parameter':
        """
        Saves a named prior hyperparameter to the Parameter class before generating the parameter's prior distribution.

        Args:
        - name (str): The name of the prior hyperparameter.
        - val (float, int or np.ndarray): The value of the prior hyperparameter.

        Returns:
        - Parameter: The Parameter instance.
        """

        if isinstance(val, list):
            val = np.array(val)

        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(val, (float, int, np.ndarray)):
            raise TypeError("val must be a float or np.ndarray")

        self.prior_params[name] = val
        return self

    def _hyperparam_checks(self, prior_params, checks):
        """
        Checks the validity of the prior hyperparameters.
        """
        if 'gaussian_check' in checks:
            if 'mu' not in prior_params or 'sigma' not in prior_params:
                raise ValueError("Parameter - class must contain the hyperparameters 'mu' and 'sigma'")                
            if set(prior_params.keys()) - {'mu', 'sigma'}:
                    raise ValueError("Parameter - class must only contain the hyperparameters 'mu' and 'sigma'")

            if self.multi_mode:
                try: 
                    mu = np.array(prior_params.mu)
                    mu.shape[0]
                except: raise ValueError("Parameter - 'mu' must be an array-like object")
                try: 
                    sigma = np.array(prior_params['sigma'])
                    sigma.shape[0]
                except: raise ValueError("Parameter - 'sigma' must be an array-like object")

                if mu.shape[0] != sigma.shape[0]:
                    raise ValueError("Parameter - 'mu' and 'sigma' indicate the same number of modes")
                for i in range(mu.shape[0]):
                    if self.multivar:
                        if mu[i].ndim != 1 or sigma[i].ndim != 2:
                            raise ValueError("Parameter - 'mu' must be a 1D array and 'sigma' must be a 2D array within each mode")
                        if mu[i].shape[0] != sigma[i].shape[0] or sigma[i].shape[0] != sigma[i].shape[1]:
                            raise ValueError("Parameter - 'mu' and 'sigma' must have the same length and 'sigma' must be a square matrix within each mode")
                        if mu[i].shape[0] != len(self.name):
                            raise ValueError("Parameter - 'mu' and 'sigma' must have the same length as the number of parameters within each mode")
                    else:
                        if not np.isscalar(prior_params.mu[i]) or not np.isscalar(prior_params['sigma'][i]):
                            raise ValueError("Parameter - 'mu' and 'sigma' must be scalar values within each mode")
            else:
                if self.multivar:
                    
                    try: 
                        mu = np.array(prior_params.mu)
                        mu.shape[0]
                    except: raise ValueError("Parameter - 'mu' must be an array-like object")
                    try: 
                        sigma = np.array(prior_params['sigma'])
                        mu.shape[0]
                    except: raise ValueError("Parameter - 'sigma' must be an array-like object")
                    if mu.ndim != 1 or sigma.ndim != 2:
                        raise ValueError("Parameter - 'mu' must be a 1D array and 'sigma' must be a 2D array")
                    if mu.shape[0] != sigma.shape[0] or sigma.shape[0] != sigma.shape[1]:
                        raise ValueError("Parameter - 'mu' and 'sigma' must have the same length and 'sigma' must be a square matrix")
                    if mu.shape[0] != len(self.name):
                        raise ValueError("Parameter - 'mu' and 'sigma' must have the same length as the number of parameters")
                else:
                    if not np.isscalar(prior_params.mu) or not np.isscalar(prior_params['sigma']):
                        raise ValueError("Parameter - 'mu' and 'sigma' must be scalar values")

        if 'gamma_check' in checks:
            if ('mu' not in prior_params or 'sigma' not in prior_params) and ('alpha' not in prior_params or 'beta' not in prior_params):
                raise ValueError("Parameter - class must contain the hyperparameters 'mu' and 'sigma' or 'alpha' and 'beta'")

            if set(prior_params.keys()) - {'mu', 'sigma'} and set(prior_params.keys()) - {'alpha', 'beta'}:
                raise ValueError("Parameter - class must only contain the hyperparameters 'mu' and 'sigma' or 'alpha' and 'beta'")

            if self.multi_mode:
                # TO BE IMPLEMENTED
                raise Exception('Parameter - multimodal gamma distribution not supported!')
            else:
                if self.multivar:
                    # TO BE IMPLEMENTED
                    raise Exception('Parameter - multivariate gamma distribution not supported!')
                else:
                    if 'mu' in prior_params and 'sigma' in prior_params:
                        if not np.isscalar(prior_params.mu) or not np.isscalar(prior_params['sigma']):
                            raise ValueError("Parameter - 'mu' and 'sigma' must be scalar values")
                    elif 'alpha' in prior_params and 'beta' in prior_params:
                        if not np.isscalar(prior_params.alpha) or not np.isscalar(prior_params.beta):
                            raise ValueError("Parameter - 'alpha' and 'beta' must be scalar values")

        if 'uniform_check' in checks:
            if 'low' not in prior_params or 'high' not in prior_params:
                raise ValueError("Parameter - class must contain the hyperparameters 'low' and 'high'")
            if set(prior_params.keys()) - {'low', 'high'}:
                raise ValueError("Parameter - class must only contain the hyperparameters 'low' and 'sigma'")
            
            if self.multi_mode:
                # TO BE IMPLEMENTED
                raise Exception('Parameter - multimodal uniform distribution not supported!')
            else:
                if self.multivar:
                    # TO BE IMPLEMENTED
                    raise Exception('Parameter - multivariate uniform distribution not supported!')
                else:
                    if not np.isscalar(prior_params.low) or not np.isscalar(prior_params.high):
                        raise ValueError("Parameter - 'low' and 'high' must be scalar values")

        if 'log_norm_check' in checks:
            if ('mu' not in prior_params or 'sigma' not in prior_params) and ('loc' not in prior_params or 'scale' not in prior_params) and ('peak' not in prior_params or 'scale' not in prior_params) and ('peak' not in prior_params or 'overall_scale' not in prior_params):
                raise ValueError(
                    "Parameter - class must contain the hyperparameters 'mu' and 'sigma' or 'loc' and 'scale' or 'peak' and 'scale'")
            if set(prior_params.keys()) - {'mu', 'sigma'} and set(prior_params.keys()) - {'loc', 'scale'} and set(prior_params.keys()) - {'peak', 'scale'} and set(prior_params.keys()) - {'peak', 'overall_scale'}:
                raise ValueError("Parameter - class must only contain the hyperparameters 'mu' and 'sigma' or 'loc' and 'scale' or 'peak' and 'scale' or 'peak' and 'overall_scale'")


            if 'mu' in prior_params and 'sigma' in prior_params:
                if self.multi_mode:
                    try:
                        mu = np.array(prior_params.mu)
                        mu.shape[0]
                    except: raise ValueError("Parameter - 'mu' must be an array-like object")
                    try:
                        sigma = np.array(prior_params['sigma'])
                        sigma.shape[0]
                    except: raise ValueError("Parameter - 'sigma' must be an array-like object")
                    if mu.shape[0] != sigma.shape[0]:
                        raise ValueError("Parameter - 'mu' and 'sigma' indicate the same number of modes")
                    for i in range(mu.shape[0]):
                        if self.multivar:
                            if mu[i].ndim != 1 or sigma[i].ndim != 2:
                                raise ValueError("Parameter - 'mu' must be a 1D array and 'sigma' must be a 2D array within each mode")
                            if mu[i].shape[0] != sigma[i].shape[0] or sigma[i].shape[0] != sigma[i].shape[1]:
                                raise ValueError("Parameter - 'mu' and 'sigma' must have the same length and 'sigma' must be a square matrix within each mode")
                            if mu[i].shape[0] != len(self.name):
                                raise ValueError("Parameter - 'mu' and 'sigma' must have the same length as the number of parameters within each mode")
                        else:
                            if not np.isscalar(prior_params.mu[i]) or not np.isscalar(prior_params['sigma'][i]):
                                raise ValueError("Parameter - 'mu' and 'sigma' must be scalar values within each mode")
                            
                else:
                    if self.multivar:
                        try:
                            mu = np.array(prior_params.mu)
                            mu.shape[0]
                        except: raise ValueError("Parameter - 'mu' must be an array-like object")
                        try:
                            sigma = np.array(prior_params['sigma'])
                            sigma.shape[0]
                        except: raise ValueError("Parameter - 'sigma' must be an array-like object")
                        if mu.ndim != 1 or sigma.ndim != 2:
                            raise ValueError("Parameter - 'mu' must be a 1D array and 'sigma' must be a 2D array")
                        if mu.shape[0] != sigma.shape[0] or sigma.shape[0] != sigma.shape[1]:
                            raise ValueError("Parameter - 'mu' and 'sigma' must have the same length and 'sigma' must be a square matrix")
                        if mu.shape[0] != len(self.name):
                            raise ValueError("Parameter - 'mu' and 'sigma' must have the same length as the number of parameters")
                    else:
                        if not np.isscalar(prior_params.mu) or not np.isscalar(prior_params['sigma']):
                            raise ValueError("Parameter - 'mu' and 'sigma' must be scalar values")
                        
            elif 'loc' in prior_params and 'scale' in prior_params:
                if self.multi_mode:
                    try:
                        loc = np.array(prior_params['loc'])
                        loc.shape[0]
                    except:
                        raise ValueError("Parameter - 'loc' must be an array-like object")
                    try:
                        scale = np.array(prior_params['scale'])
                        scale.shape[0]
                    except:
                        raise ValueError("Parameter - 'scale' must be an array-like object")
                    if loc.shape[0] != scale.shape[0]:
                        raise ValueError("Parameter - 'loc' and 'scale' indicate the same number of modes")
                    for i in range(loc.shape[0]):
                        if self.multivar:
                            if loc[i].ndim != 1 or scale[i].ndim != 2:
                                raise ValueError("Parameter - 'loc' must be a 1D array and 'scale' must be a 2D array within each mode")
                            if loc[i].shape[0] != scale[i].shape[0] or scale[i].shape[0] != scale[i].shape[1]:
                                raise ValueError("Parameter - 'loc' and 'scale' must have the same length and 'scale' must be a square matrix within each mode")
                            if loc[i].shape[0] != len(self.name):
                                raise ValueError("Parameter - 'loc' and 'scale' must have the same length as the number of parameters within each mode")
                        else:
                            if not np.isscalar(prior_params['loc'][i]) or not np.isscalar(prior_params['scale'][i]):
                                raise ValueError("Parameter - 'loc' and 'scale' must be scalar values within each mode")
                else:
                    if self.multivar:
                        try:
                            loc = np.array(prior_params['loc'])
                            loc.shape[0]
                        except:
                            raise ValueError("Parameter - 'loc' must be an array-like object")
                        try:
                            scale = np.array(prior_params['scale'])
                            scale.shape[0]
                        except:
                            raise ValueError("Parameter - 'scale' must be an array-like object")
                        if loc.ndim != 1 or scale.ndim != 2:
                            raise ValueError("Parameter - 'loc' must be a 1D array and 'scale' must be a 2D array")
                        if loc.shape[0] != scale.shape[0] or scale.shape[0] != scale.shape[1]:
                            raise ValueError("Parameter - 'loc' and 'scale' must have the same length and 'scale' must be a square matrix")
                        if loc.shape[0] != len(self.name):
                            raise ValueError("Parameter - 'loc' and 'scale' must have the same length as the number of parameters")
                    else:
                        if not np.isscalar(prior_params['loc']) or not np.isscalar(prior_params['scale']):
                            raise ValueError("Parameter - 'loc' and 'scale' must be scalar values")
                
            elif 'peak' in prior_params and 'scale' in prior_params:
                if self.multi_mode:
                    try:
                        peak = np.array(prior_params['peak'])
                        peak.shape[0]
                    except:
                        raise ValueError("Parameter - 'peak' must be an array-like object")
                    try:
                        scale = np.array(prior_params['scale'])
                        scale.shape[0]
                    except:
                        raise ValueError("Parameter - 'scale' must be an array-like object")
                    if peak.shape[0] != scale.shape[0]:
                        raise ValueError("Parameter - 'peak' and 'scale' indicate the same number of modes")
                    for i in range(peak.shape[0]):
                        if self.multivar:
                            if peak[i].ndim != 1 or scale[i].ndim != 2:
                                raise ValueError("Parameter - 'peak' must be a 1D array and 'scale' must be a 2D array within each mode")
                            if peak[i].shape[0] != scale[i].shape[0] or scale[i].shape[0] != scale[i].shape[1]:
                                raise ValueError("Parameter - 'peak' and 'scale' must have the same length and 'scale' must be a square matrix within each mode")
                            if peak[i].shape[0] != len(self.name):
                                raise ValueError("Parameter - 'peak' and 'scale' must have the same length as the number of parameters within each mode")
                        else:
                            if not np.isscalar(prior_params['peak'][i]) or not np.isscalar(prior_params['scale'][i]):
                                raise ValueError("Parameter - 'peak' and 'scale' must be scalar values within each mode")
                else:
                    if self.multivar:
                        try:
                            peak = np.array(prior_params['peak'])
                            peak.shape[0]
                        except:
                            raise ValueError("Parameter - 'peak' must be an array-like object")
                        try:
                            scale = np.array(prior_params['scale'])
                            scale.shape[0]
                        except:
                            raise ValueError("Parameter - 'scale' must be an array-like object")
                        if peak.ndim != 1 or scale.ndim != 2:
                            raise ValueError("Parameter - 'peak' must be a 1D array and 'scale' must be a 2D array")
                        if peak.shape[0] != scale.shape[0] or scale.shape[0] != scale.shape[1]:
                            raise ValueError("Parameter - 'peak' and 'scale' must have the same length and 'scale' must be a square matrix")
                        if peak.shape[0] != len(self.name):
                            raise ValueError("Parameter - 'peak' and 'scale' must have the same length as the number of parameters")
                    else:
                        if not np.isscalar(prior_params['peak']) or not np.isscalar(prior_params['scale']):
                            raise ValueError("Parameter - 'peak' and 'scale' must be scalar values")
                
            elif 'peak' in prior_params and 'overall_scale' in prior_params:
                if self.multi_mode:
                    try:
                        peak = np.array(prior_params['peak'])
                        peak.shape[0]
                    except:
                        raise ValueError("Parameter - 'peak' must be an array-like object")
                    
                    if not isinstance(prior_params['overall_scale'], (float, int)):
                        raise ValueError("Parameter - 'overall_scale' must be a float or integer")

                    for i in range(peak.shape[0]):
                        if self.multivar:
                            if peak[i].ndim != 1:
                                raise ValueError("Parameter - 'peak' must be a 1D array within each mode")
                            if peak[i].shape[0] != len(self.name):
                                raise ValueError("Parameter - 'peak' must have the same length as the number of parameters within each mode")
                        else:
                            if not np.isscalar(prior_params['peak'][i]):
                                raise ValueError("Parameter - 'peak' must be a scalar value within each mode")
                else:
                    raise ValueError("Parameter - using 'peak' and 'overall_scale' requires 'multi_mode' to be 'True'")
            else:
                raise ValueError("Parameter - class must contain the hyperparameters 'mu' and 'sigma' or 'loc' and 'scale' or 'peak' and 'scale'")
                          
    def get_prior_function(self) -> distributions.Distribution:
        """
        Generates the selected prior distribution using the prior hyperparameters.

        Returns:
        - numpyro.distributions.Distribution: The prior distribution.
        """

        def gaussian_prior(prior_params):
            self._hyperparam_checks(prior_params, ['gaussian_check'])
            if self.multi_mode:
                dists = []
                mixture_size = np.shape(prior_params.mu)[0]
                mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                for i in range(np.shape(prior_params.mu)[0]):
                    if self.multivar:
                        multinorm = distributions.MultivariateNormal(prior_params.mu[i], prior_params['sigma'][i]**2)
                        dists.append(multinorm)
                    else:
                        multinorm = distributions.Normal(prior_params.mu[i], prior_params['sigma'][i]**2)
                        dists.append(multinorm)
                return distributions.MixtureGeneral(mixing_dist, dists)
            else:
                if self.multivar:
                    return distributions.MultivariateNormal(prior_params.mu, prior_params['sigma']**2)
                else:
                    return distributions.Normal(prior_params.mu, prior_params['sigma']**2)

        def gamma_prior(prior_params):
            self._hyperparam_checks(prior_params, ['gamma_check'])
            if 'mu' in prior_params and 'sigma' in prior_params:
                return distributions.Gamma(self._alpha_gamma(prior_params.mu, prior_params['sigma']),
                                           self._beta_gamma(prior_params.mu, prior_params['sigma']))
            elif 'alpha' in prior_params and 'beta' in prior_params:
                return distributions.Gamma(prior_params.alpha, prior_params.beta)
            else:
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'sigma' or 'alpha' and 'beta'")

        def uniform_prior(prior_params):
            self._hyperparam_checks(prior_params, ['uniform_check'])
            return distributions.Uniform(prior_params.low, prior_params.high)
        
        def log_norm_prior(prior_params):
            self._hyperparam_checks(prior_params, ['log_norm_check'])         

            if 'mu' in prior_params and 'sigma' in prior_params:
                if self.multi_mode:
                    dists = []
                    mixture_size = np.shape(prior_params.mu)[0]
                    mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                    for i in range(mixture_size):
                        multinorm = distributions.MultivariateNormal(self._mean_stdev_to_loc_log_norm(prior_params.mu[i], prior_params['sigma'][i]),
                                                                        self._mean_stdev_to_scale_log_norm(prior_params.mu[i], prior_params['sigma'][i]))
                        dists.append(distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform()))
                    return distributions.MixtureGeneral(mixing_dist, dists)
                else:
                    if self.multivar:
                        multinorm = distributions.MultivariateNormal(self._mean_stdev_to_loc_log_norm(self.prior_params.mu, self.prior_params['sigma']),
                                                                    self._mean_stdev_to_scale_log_norm(self.prior_params.mu, self.prior_params['sigma']))
                        return distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
                    else:
                        return distributions.LogNormal(self._mean_stdev_to_loc_log_norm(self.prior_params.mu, self.prior_params['sigma']),
                                                    np.sqrt(self._mean_stdev_to_scale_log_norm(self.prior_params.mu, self.prior_params['sigma'])))
                    
            elif 'loc' in prior_params and 'scale' in prior_params:
                if self.multi_mode:
                    dists = []
                    mixture_size = np.shape(prior_params['loc'])[0]
                    mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                    for i in range(np.shape(prior_params['loc'])[0]):
                        multinorm = distributions.MultivariateNormal(prior_params['loc'][i], prior_params['scale'][i])
                        dists.append(distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform()))
                    return distributions.MixtureGeneral(mixing_dist, dists)
                else:
                    if self.multivar:
                        multinorm = distributions.MultivariateNormal(prior_params['loc'], prior_params['scale'])
                        return distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
                    else:
                        return distributions.LogNormal(prior_params['loc'], np.sqrt(prior_params['scale']))
                    
            elif 'peak' in prior_params and 'scale' in prior_params:
                if self.multi_mode:
                    dists = []
                    mixture_size = np.shape(prior_params['peak'])[0]
                    mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                    for i in range(np.shape(prior_params['peak'])[0]):
                        loc = self._peak_to_loc(prior_params['peak'][i], prior_params['scale'][i])
                        multinorm = distributions.MultivariateNormal(loc, prior_params['scale'][i])                        
                        dists.append(distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform()))
                    return distributions.MixtureGeneral(mixing_dist, dists)
                else:
                    if self.multivar:
                        loc = self._peak_to_loc(prior_params['peak'], prior_params['scale'])
                        multinorm = distributions.MultivariateNormal(loc, prior_params['scale'])
                        return distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
                    else:
                        return distributions.LogNormal(self._peak_to_loc(prior_params['peak'], prior_params['scale']), np.sqrt(prior_params['scale']))
            
            elif 'peak' in prior_params and 'overall_scale' in prior_params:
                if self.multi_mode:
                    dists = []
                    max_vals = []
                    mixture_size = np.shape(prior_params['peak'])[0]
                    mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                    for i in range(mixture_size):
                        loc = self._peak_to_loc(prior_params['peak'][i], prior_params['overall_scale'])
                        multinorm = distributions.MultivariateNormal(loc, prior_params['overall_scale']*np.eye(len(loc)))
                        dist = distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
                        max_vals.append(np.exp(dist.log_prob(prior_params['peak'][i])))
                    target_max_pdf = min(max_vals)  # Use the maximum pdf value among all distributions as the target
                    for i in range(mixture_size):
                        mode_peak = prior_params['peak'][i]
                        adjusted_scale = root_scalar(lambda scale: scale + np.log(2*np.pi*mode_peak[0]*mode_peak[1]*scale*target_max_pdf), x0=prior_params['overall_scale']).root
                        if np.isnan(adjusted_scale):
                            raise ValueError("Parameter - unable to find a valid scale for the distribution")
                        loc = self._peak_to_loc(prior_params['peak'][i], adjusted_scale)
                        multinorm = distributions.MultivariateNormal(loc, adjusted_scale*np.eye(len(loc)))
                        transformed_dist = distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
                        dists.append(transformed_dist)
                    return distributions.MixtureGeneral(mixing_dist, dists)
                else:
                    raise ValueError("Parameter - using 'peak' and 'overall_scale' requires 'multi_mode' to be 'True'")
            else:
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'sigma' or 'loc' and 'scale' or 'peak' and 'scale' or 'peak' and 'overall_scale'")
                
        
        if self.prior_select == 'gaussian':
            dist = gaussian_prior(self.prior_params)
            return dist
        
        elif self.prior_select == 'gamma':
            return gamma_prior(self.prior_params)

        elif self.prior_select == 'uniform':
            return uniform_prior(self.prior_params)
        
        elif self.prior_select == 'log_norm':  
            return log_norm_prior(self.prior_params)        
        else:
            raise ValueError("Invalid prior distribution selected!")

    def sample_param(self) -> list[float]|float:
        """
        Generates a sample for this parameter.

        Returns:
        - float or list[float]: The sampled parameter value.
        """
        if self.prior_func == None:
            self.prior_func = self.get_prior_function()
        sample_val = numpyro.sample(self.joined_name, self.prior_func)
        return sample_val
