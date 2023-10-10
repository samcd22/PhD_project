import numpy as np
import pandas as pd
import numpyro

# Parameter class - used for generating the infered parameters in our sampler
class Parameter:
    # Initialises the Likelihood class saving all relevant variables
    def __init__(self, name, prior_select = "gaussian", default_value = 'NaN'):
        self.prior_params = pd.Series({},dtype='float64')
        self.name = name
        self.prior_select = prior_select
        self.default_value = default_value
        
    # Utility function for converting mean and stdev to alpha in a Gamma distribution 
    def alpha_gamma(self,mu,sigma): return mu**2/sigma**2

    # Utility function for converting mean and stdev to beta in a Gamma distribution 
    def beta_gamma(self,mu,sigma): return mu/(sigma**2)

    # Utility function for converting mean and stdev to alpha in a Log Normal distribution 
    def alpha_log_norm(self,mu,sigma): return np.log(mu) - 0.5*np.log(1+sigma**2/mu)

    # Utility function for converting mean and stdev to beta in a Log Normal distribution 
    def beta_log_norm(self,mu,sigma): return np.sqrt(np.log(1+sigma**2/mu))

    # Saves a named prior hyperparameters to the Parameter class before generating the parameter's prior distribution
    def add_prior_param(self, name, val):
        self.prior_params[name] = val
        return self

    # Generates the selected prior distribution using the prior hyperparameters
    def get_prior_function(self):
        # Gaussian Prior
        if self.prior_select == 'gaussian':
            return numpyro.distributions.Normal(self.prior_params.mu, self.prior_params.sigma)
        
        # Gamma Prior
        elif self.prior_select == 'gamma':
            return numpyro.distributions.Gamma(self.alpha_gamma(self.prior_params.mu,self.prior_params.sigma), self.beta_gamma(self.prior_params.mu,self.prior_params.sigma))
    
        # Uniform
        elif self.prior_select == 'uniform':
            return numpyro.distributions.Uniform(self.prior_params.low, self.prior_params.high)
        
        # Log Normal
        if self.prior_select == 'log_norm':
            return numpyro.distributions.LogNormal(self.alpha_log_norm(self.prior_params.mu,self.prior_params.sigma), self.beta_log_norm(self.prior_params.mu,self.prior_params.sigma))
        
    # Generates a sample for this parameter
    def sample_param(self):
        prior_func = self.get_prior_function()
        a = numpyro.sample(self.name, prior_func)
        return a
    
    # Copy function for the Parameter Class
    def copy(self):
        return Parameter(self.val, self.step_select, self.step_size, self.prior_select)