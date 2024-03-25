import pandas as pd
import numpyro

# Likelihood class - used for generating a likelihood function
class Likelihood:
    # Initialises the Likelihood class saving all relevant variables
    def __init__(self, likelihood_select):
        self.likelihood_params = pd.Series({},dtype='float64')
        self.likelihood_select = likelihood_select

    # Utility function for converting mean and stdev to alpha in a Gamma distribution 
    def alpha(self,mu,sigma): return mu**2/sigma**2

    # Utility function for converting mean and stdev to beta in a Gamma distribution 
    def beta(self,mu,sigma): return mu/sigma**2

    # Saves a named parameter to the Likelihood class before generating the likelihood function
    def add_likelihood_param(self,name,val):
        self.likelihood_params[name] = val
        return self

    # Generates the selected likelihood function using the likelihood parameters
    def get_likelihood_function(self):

        # Gaussian likelihood with non-inferred sigma
        if self.likelihood_select == 'gaussian_fixed_sigma':
            def gaussian_likelihood_fixed_sigma(mu, params):
                return numpyro.distributions.Normal(mu, self.likelihood_params.sigma)
            return gaussian_likelihood_fixed_sigma
        
        # Gaussian likelihood with inferred sigma
        elif self.likelihood_select == 'gaussian':
            def gaussian_likelihood(mu, params):
                return numpyro.distributions.Normal(mu, params['sigma'])
            return gaussian_likelihood
        
        # Gamma likelihood with non-infered sigma
        elif self.likelihood_select == 'gamma_fixed_sigma':
            def gamma_likelihood_fixed_sigma(mu, params):
                return numpyro.distributions.Gamma(self.alpha(mu,self.likelihood_params.sigma), 1/self.beta(mu, self.likelihood_params.sigma))
            return gamma_likelihood_fixed_sigma
        
        # Gamma likelihood with infered sigma
        elif self.likelihood_select == 'gamma':
            def gamma_likelihood(mu, params):
                sigma = params['sigma']
                return numpyro.distributions.Gamma(self.alpha(mu,sigma), 1/self.beta(mu,sigma))
            return gamma_likelihood