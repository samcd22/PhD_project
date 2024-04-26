import numpy as np
import pandas as pd
import numpyro
import jax.numpy as jnp

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
    - jointed_name (str): The combined name of the parameter/joint parameters.
    - order (int): The order of magnitude of the parameter.

    Methods:
    - add_prior_param: Saves a named prior hyperparameter to the Parameter class before generating the parameter's prior distribution.
    - get_prior_function: Generates the selected prior distribution using the prior hyperparameters.
    - sample_param: Generates a sample for this parameter.
    """
    def __init__(self, name: Union[str, List[str]], prior_select: str = "gaussian", order: int = 1):
        """
        Initializes the Parameter class.

        Args:
        - name (str or list): The name(s) of the parameter.
        - prior_select (str, optional): The type of prior distribution to use. Defaults to "gaussian". Options are:
            - "gaussian": Gaussian distribution 
                - Can handle single and multivariate cases.
                - Requires 'mu' and 'cov' hyperparameters.
            - "gamma": Gamma distribution 
                - Can only handle single variate cases.
                - Requires either 'mu' and 'cov' or 'alpha' and 'beta' hyperparameters.
            - "uniform": Uniform distribution.
                - Can only handle single variate cases.
                - Requires 'low' and 'high' hyperparameters.
            - "log_norm": Log-normal distribution.
                - Can handle single and multivariate cases.
                - Requires either 'mu' and 'cov' or 'loc' and 'scale' hyperparameters.
            - "multi_mode_gaussian": Multimodal Gaussian distribution.
                - Can handle single and multivariate cases.
                - Requires 'mu' and 'cov' hyperparameters.
            - "multi_mode_log_norm": Multimodal Log-normal distribution.
                - Can handle single and multivariate cases.
                - Requires either 'mu' and 'cov' or 'loc' and 'scale' hyperparameters.
        - order (int, optional): The order of the parameter. Defaults to 1.
        """
        self.prior_params = pd.Series({}, dtype='float64')
        self.name = name
        self.prior_select = prior_select
        self.multivar = False
        self.multi_mode = False
        if isinstance(self.name, list) and len(self.name) >= 2:
            self.multivar = True
            self.joined_name = '_and_'.join(self.name)
        else:
            self.name = self.name
        self.order = order

    def _alpha_gamma(self, mu: Union[float, np.ndarray], cov: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Utility function for converting mean and stdev to alpha in a Gamma distribution.

        Args:
            mu (float or array-like): The mean value(s).
            cov (float or array-like): The covariance value(s).

        Returns:
            float or array-like: The alpha value(s).
        """
        return mu**2 / cov

    def _beta_gamma(self, mu: Union[float, np.ndarray], cov: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Utility function for converting mean and stdev to beta in a Gamma distribution.

        Args:
            mu (float or array-like): The mean value(s).
            cov (float or array-like): The covariance value(s).

        Returns:
            float or array-like: The beta value(s).
        """
        return mu / cov

    def _loc_log_norm(self, mu: Union[float, np.ndarray], cov: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Utility function for converting mean and stdev to location in a Log Normal distribution.

        Args:
            mu (float or array-like): The mean value(s).
            cov (float or array-like): The covariance value(s).

        Returns:
            float or array-like: The loc value(s).
        """
        if np.isscalar(mu):
            return np.log(mu) - 0.5 * np.log(1 + cov / mu**2)
        else:
            mu = np.array(mu)
            cov = np.array(cov)
            alpha = np.zeros(len(mu))
            for i in range(len(mu)):
                alpha[i] = np.log(mu[i]) - 1 / 2 * np.log(cov[i, i] / mu[i]**2 + 1)
            return np.array(alpha)

    def _scale_log_norm(self, mu: Union[float, np.ndarray], cov: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Utility function for converting mean and stdev to scale in a Log Normal distribution.

        Args:
            mu (float or array-like): The mean value(s).
            cov (float or array-like): The covariance value(s).

        Returns:
            float or array-like: The scale value(s).
        """
        if np.isscalar(mu):
            return np.sqrt(np.log(1 + cov / mu**2))
        else:
            mu = np.array(mu)
            cov = np.array(cov)
            beta = np.zeros((len(mu), len(mu)))
            for i in range(len(mu)):
                for j in range(len(mu)):
                    beta[i, j] = np.log(cov[i, j] / (mu[i] * mu[j]) + 1)
            return beta

    def add_prior_param(self, name: str, val: Union[float, np.ndarray]) -> 'Parameter':
        """
        Saves a named prior hyperparameter to the Parameter class before generating the parameter's prior distribution.

        Args:
            name (str): The name of the prior hyperparameter.
            val (float): The value of the prior hyperparameter.

        Returns:
            Parameter: The Parameter instance.
        """
        self.prior_params[name] = val
        return self

    def _hyperparam_checks(self, prior_params, checks):
        """
        Checks the validity of the prior hyperparameters.
        """
        if 'gaussian_params_check' in checks:
            if 'mu' not in prior_params or 'cov' not in prior_params:
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'cov'")

            if self.multivar:
                try: mu = np.array(prior_params.mu)
                except: raise ValueError("Parameter 'mu' must be an array-like object")
                try: cov = np.array(prior_params.cov)
                except: raise ValueError("Parameter 'cov' must be an array-like object")
                if mu.ndim != 1 or cov.ndim != 2:
                    raise ValueError("Parameter 'mu' must be a 1D array and 'cov' must be a 2D array")
                if mu.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
                    raise ValueError("Parameter 'mu' and 'cov' must have the same length and 'cov' must be a square matrix")
                if mu.shape[0] != len(self.names):
                    raise ValueError("Parameter 'mu' and 'cov' must have the same length as the number of parameters")
            else:
                if not np.isscalar(prior_params.mu) or not np.isscalar(prior_params.cov):
                    raise ValueError("Parameter 'mu' and 'cov' must be scalar values")

        if 'gamma_check' in checks:
            if ('mu' not in prior_params or 'cov' not in prior_params) and ('alpha' not in prior_params or 'beta' not in prior_params):
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'cov' or 'alpha' and 'beta'")

            if self.multivar:
                raise Exception('Multivariate Gamma distribution not supported!')
            else:
                if 'mu' in prior_params and 'cov' in prior_params:
                    if not np.isscalar(prior_params.mu) or not np.isscalar(prior_params.cov):
                        raise ValueError("Parameter 'mu' and 'cov' must be scalar values")
                elif 'alpha' in prior_params and 'beta' in prior_params:
                    if not np.isscalar(prior_params.alpha) or not np.isscalar(prior_params.beta):
                        raise ValueError("Parameter 'alpha' and 'beta' must be scalar values")

        if 'uniform_check' in checks:
            if 'low' not in prior_params or 'high' not in prior_params:
                raise ValueError("Parameter class must contain the hyperparameters 'low' and 'high'")
            if self.multivar:
                raise Exception('Multivariate Uniform distribution not supported!')
            else:
                if not np.isscalar(prior_params.low) or not np.isscalar(prior_params.high):
                    raise ValueError("Parameter 'low' and 'high' must be scalar values")

        if 'log_norm_check' in checks:
            if ('mu' not in prior_params or 'cov' not in prior_params) and (
                    'loc' not in prior_params or 'scale' not in prior_params):
                raise ValueError(
                    "Parameter class must contain the hyperparameters 'mu' and 'cov' or 'loc' and 'scale'")
            if self.multivar:
                if 'mu' in prior_params or 'cov' in prior_params:
                    try:
                        mu = np.array(prior_params.mu)
                    except:
                        raise ValueError("Parameter 'mu' must be an array-like object")
                    try:
                        cov = np.array(prior_params.cov)
                    except:
                        raise ValueError("Parameter 'cov' must be an array-like object")
                    if mu.ndim != 1 or cov.ndim != 2:
                        raise ValueError("Parameter 'mu' must be a 1D array and 'cov' must be a 2D array")
                    if mu.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
                        raise ValueError("Parameter 'mu' and 'cov' must have the same length and 'cov' must be a square matrix")
                    if mu.shape[0] != len(self.names):
                        raise ValueError("Parameter 'mu' and 'cov' must have the same length as the number of parameters")

                elif 'loc' in prior_params or 'scale' in prior_params:
                    try:
                        loc = np.array(prior_params.loc)
                    except:
                        raise ValueError("Parameter 'loc' must be an array-like object")
                    try:
                        scale = np.array(prior_params.scale)
                    except:
                        raise ValueError("Parameter 'scale' must be an array-like object")
                    if loc.ndim != 1 or scale.ndim != 2:
                        raise ValueError("Parameter 'loc' must be a 1D array and 'scale' must be a 2D array")
                    if loc.shape[0] != scale.shape[0] or scale.shape[0] != scale.shape[1]:
                        raise ValueError("Parameter 'loc' and 'scale' must have the same length and 'scale' must be a square matrix")
                    if loc.shape[0] != len(self.names):
                        raise ValueError("Parameter 'loc' and 'scale' must have the same length as the number of parameters")
            else:
                if 'mu' in prior_params or 'cov' in prior_params:
                    if not np.isscalar(prior_params.mu) or not np.isscalar(prior_params.cov):
                        raise ValueError("Parameter 'mu' and 'cov' must be scalar values")
                elif 'loc' in prior_params or 'scale' in prior_params:
                    if not np.isscalar(prior_params.loc) or not np.isscalar(prior_params.scale):
                        raise ValueError("Parameter 'loc' and 'scale' must be scalar values")
                    
        if 'multi_mode_gaussian_params_check' in checks:
            if 'mu' not in prior_params or 'cov' not in prior_params:
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'cov'")
            if np.shape(prior_params.mu)[0] != np.shape(prior_params.cov)[0]:
                    raise Exception('Parameters must indicate the same number of modes!')
            if self.multivar:
                for i in range(len(prior_params.mu)):
                    try: mu = np.array(prior_params.mu[i])
                    except: raise ValueError("Parameter 'mu' must be an array-like object within each mode")
                    try: cov = np.array(prior_params.cov[i])
                    except: raise ValueError("Parameter 'cov' must be an array-like object within each mode")
                    if mu.ndim != 1 or cov.ndim != 2:
                        raise ValueError("Parameter 'mu' must be a 1D array and 'cov' must be a 2D array within each mode")
                    if mu.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
                        raise ValueError("Parameter 'mu' and 'cov' must have the same length and 'cov' must be a square matrix within each mode")
                    if mu.shape[0] != len(self.names):
                        raise ValueError("Parameter 'mu' and 'cov' must have the same length as the number of parameters within each mode")
            else:
                for i in range(len(prior_params.mu)):
                    if not np.isscalar(prior_params.mu[i]) or not np.isscalar(prior_params.cov[i]):
                        raise ValueError("Parameter 'mu' and 'cov' must be scalar values within each mode")
            
    def get_prior_function(self) -> distributions.Distribution:
        """
        Generates the selected prior distribution using the prior hyperparameters.

        Returns:
            numpyro.distributions.Distribution: The prior distribution.
        """


        def _gaussian_prior(prior_params):
            self._hyperparam_checks(prior_params, ['gaussian_params_check'])
            return distributions.MultivariateNormal(prior_params.mu, prior_params['cov'])

        def _gamma_prior(prior_params):
            self._hyperparam_checks(prior_params, ['gamma_check'])
            if 'mu' in prior_params and 'cov' in prior_params:
                return distributions.Gamma(self._alpha_gamma(prior_params.mu, prior_params['cov']),
                                           self._beta_gamma(prior_params.mu, prior_params['cov']))
            elif 'alpha' in prior_params and 'beta' in prior_params:
                return distributions.Gamma(prior_params.alpha, prior_params.beta)
            else:
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'cov' or 'alpha' and 'beta'")

        def _uniform_prior(prior_params):
            self._hyperparam_checks(prior_params, ['uniform_check'])
            if self.multivar:
                raise Exception('Multivariate Uniform distribution not supported!')
            return distributions.Uniform(prior_params.low, prior_params.high)
        
        def _log_norm_prior(prior_params):
            self._hyperparam_checks(prior_params, ['log_norm_check'])
            
            if 'mu' in prior_params and 'cov' in prior_params:
                if self.multivar:
                    multinorm = distributions.MultivariateNormal(self._loc_log_norm(self.prior_params.mu, self.prior_params['cov']),
                                                                self._scale_log_norm(self.prior_params.mu, self.prior_params['cov']))
                    return distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
                else:
                    return distributions.LogNormal(self._loc_log_norm(self.prior_params.mu, self.prior_params['cov']),
                                                self._scale_log_norm(self.prior_params.mu, self.prior_params['cov']))
            elif 'loc' in prior_params and 'scale' in prior_params:
                if self.multivar:
                    multinorm = distributions.MultivariateNormal(prior_params.loc, prior_params.scale)
                    return distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
                else:
                    return distributions.LogNormal(prior_params.loc, prior_params.scale)
            else:
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'cov' or 'loc' and 'scale'")

        def _multi_mode_gaussian_prior(prior_params):
            self.multi_mode = True
            self._hyperparam_checks(prior_params, ['multi_mode_gaussian_params_check'])
            dists = []
            mixture_size = np.shape(prior_params.mu)[0]
            mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
            for i in range(np.shape(prior_params.mu)[0]):
                dists.append(distributions.MultivariateNormal(prior_params.mu[i], prior_params.cov[i]))
            return distributions.MixtureGeneral(mixing_dist, dists)
        
        def _multi_mode_log_norm_prior(prior_params):
            self.multi_mode = True
            self._hyperparam_checks(prior_params, ['multi_mode_log_norm_params_check'])
            dists = []
            if 'mu' in prior_params and 'cov' in prior_params: 
                mixture_size = np.shape(prior_params.mus)[0]
                mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                for i in range(np.shape(prior_params.mu)[0]):
                    multinorm = distributions.MultivariateNormal(self._loc_log_norm(prior_params.mu[i], prior_params.cov[i]),
                                                                    self._scale_log_norm(prior_params.mu[i], prior_params.cov[i]))
                    dists.append(distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform()))
                return distributions.MixtureGeneral(mixing_dist, dists)
            elif 'loc' in prior_params and 'scale' in prior_params:
                mixture_size = np.shape(prior_params.loc)[0]
                mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                for i in range(np.shape(prior_params.loc)[0]):
                    multinorm = distributions.MultivariateNormal(prior_params.loc[i], prior_params.scale[i])
                    dists.append(distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform()))
                return distributions.MixtureGeneral(mixing_dist, dists)
            else:
                raise ValueError("Parameter class must contain the hyperparameters 'mu' and 'cov' or 'loc' and 'scale'")

        if self.prior_select == 'gaussian':
            return _gaussian_prior(self.prior_params)
        
        elif self.prior_select == 'gamma':
            return _gamma_prior(self.prior_params)

        elif self.prior_select == 'uniform':
            return _uniform_prior(self.prior_params)
        
        elif self.prior_select == 'log_norm':  
            return _log_norm_prior(self.prior_params)
        
        elif self.prior_select == 'multi_mode_gaussian':
            return _multi_mode_gaussian_prior(self.prior_params)

        elif self.prior_select == 'multi_mode_log_norm':
            return _multi_mode_log_norm_prior(self.prior_params)
        
        else:
            raise ValueError("Invalid prior distribution selected!")

    def sample_param(self) -> tuple:
        """
        Generates a sample for this parameter.

        Returns:
            tuple: A tuple containing the sampled parameter value and its order.
        """
        prior_func = self.get_prior_function()
        a = numpyro.sample(self.name, prior_func)
        return a, self.order