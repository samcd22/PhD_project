import numpy as np
import pandas as pd
import numpyro

class Parameter:

    def __init__(self, name, init_val, prior_select = "gaussian"):
        self.prior_params = pd.Series({},dtype='float64')
        self.val = 0        
        self.name = name
        self.prior_select = prior_select
        self.init_val = init_val
        
    def alpha(self,mu,sigma): return mu**2/sigma**2

    def beta(self,mu,sigma):
        return mu/(sigma**2)

    def add_prior_param(self, name, val):
        self.prior_params[name] = val

    def get_prior_function(self):
        if self.prior_select == 'gaussian':
            return numpyro.distributions.Normal(self.prior_params.mu, self.prior_params.sigma)
        
        elif self.prior_select == 'gamma':
            return numpyro.distributions.Gamma(self.alpha(self.prior_params.mu,self.prior_params.sigma), self.beta(self.prior_params.mu,self.prior_params.sigma))
        
    def sample_param(self):
        prior_func = self.get_prior_function()
        a = numpyro.sample(self.name, prior_func)
        return a
        
    # # Step Function
    # def get_step_function(self):
    #     # Probability of step
    #     def log_p_step_multivariate_gaussian(val, mu):
    #         return stats.multivariate_normal.logpdf(val, mean=mu, cov=self.step_size**2)
        
    #     def log_p_step_gamma(val, mu):
    #         beta = mu/self.step_size**2
    #         a = mu**2/self.step_size**2
    #         return stats.gamma.logpdf(val, a, scale=1/beta)
        
    #     # The step itself
    #     def step_multivariate_positive_gaussian(mu):
    #         stepped_val = -1
    #         while stepped_val <= 0:
    #             stepped_val = stats.multivariate_normal.rvs(mean=mu,cov=self.step_size**2)
    #         return stepped_val
        
    #     def step_multivariate_gaussian(mu):
    #         return stats.multivariate_normal.rvs(mean=mu,cov=self.step_size**2)
        
    #     def step_gamma(mu):
    #         beta = mu/self.step_size**2
    #         a = mu**2/self.step_size**2
    #         return stats.gamma.rvs(a,scale=1/beta)
        
        
    #     if self.step_select == "positive gaussian":
    #         return log_p_step_multivariate_gaussian, step_multivariate_positive_gaussian
        
    #     elif self.step_select == 'gamma':
    #         return log_p_step_gamma, step_gamma
        
    #     elif self.step_select == 'gaussian':
    #         return log_p_step_multivariate_gaussian, step_multivariate_gaussian
            

    # Priors
    def get_log_prior(self):
        def log_gaussian_prior(val):
            return -(val-self.prior_params.mu)**2/(2*self.prior_params.sigma**2)
        def log_gamma_prior(val):
            return (self.prior_params.k - 1)*np.log(val)-val/self.prior_params.theta
        def no_prior(val):
            return 0

        if self.prior_select == "gaussian":
            return log_gaussian_prior
        elif self.prior_select == "gamma":
            return log_gamma_prior
        elif self.prior_select == "no prior":
            return no_prior
        
    def copy(self):
        return Parameter(self.val, self.step_select, self.step_size, self.prior_select)