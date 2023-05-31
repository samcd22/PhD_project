import pandas as pd
import numpyro

# Likelihood Function
class Likelihood:
    def __init__(self, likelihood_select):
        self.likelihood_params = pd.Series({},dtype='float64')
        self.likelihood_select = likelihood_select

    def alpha(self,mu,sigma): return mu**2/sigma**2

    def beta(self,mu,sigma): return mu/sigma**2

    def add_likelihood_param(self,name,val):
        self.likelihood_params[name] = val
        return self

    def get_likelihood_function(self):
        if self.likelihood_select == 'gaussian_fixed_sigma':
            def gaussian_likelihood_fixed_sigma(mu, params):
                return numpyro.distributions.Normal(mu, self.likelihood_params.sigma)
            return gaussian_likelihood_fixed_sigma
        
        elif self.likelihood_select == 'gaussian':
            def gaussian_likelihood(mu, params):
                return numpyro.distributions.Normal(mu, params['sigma'])
            return gaussian_likelihood
        
        elif self.likelihood_select == 'gamma_fixed_sigma':
            def gamma_likelihood_fixed_sigma(mu, params):
                return numpyro.distributions.Gamma(self.alpha(mu,self.likelihood_params.sigma), self.beta(mu, self.likelihood_params.sigma))
            return gamma_likelihood_fixed_sigma
        
        elif self.likelihood_select == 'gamma':
            def gamma_likelihood(mu, params):
                sigma = params['sigma']
                return numpyro.distributions.Gamma(self.alpha(mu,sigma), 1/self.beta(mu,sigma))
            return gamma_likelihood