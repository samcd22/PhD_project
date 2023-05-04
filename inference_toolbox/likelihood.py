import numpy as np
import pandas as pd
import scipy.stats as stats

# Likelihood Function
class Likelihood:
    def __init__(self, likelihood_select):
        self.likelihood_params = pd.Series({},dtype='float64')
        self.likelihood_select = likelihood_select

    def add_likelihood_param(self,name,val):
        self.likelihood_params[name] = val
    
    def get_log_likelihood_func(self):
        def gaussian_log_likelihood_fixed_sigma(modeled_vals, measured_vals):
            # cum_sum = 0
            # for i in range(len(modeled_vals)): 
                # in_exp = (modeled_vals[i]-measured_vals[i])**2/(2*self.likelihood_params.sigma**2)
                # cum_sum +=in_exp
            return -np.sum((modeled_vals-measured_vals)**2)/(2*self.likelihood_params.sigma**2)#- modeled_vals.size*np.log(np.sqrt(2*np.pi)*self.likelihood_params.sigma)

        def gaussian_log_likelihood_hetroscedastic_fixed_sigma(modeled_vals, measured_vals):
            res = abs(modeled_vals-measured_vals)
            trans_res = ((res+self.likelihood_params.lambda_2)**self.likelihood_params.lambda_1-1)/self.likelihood_params.lambda_1
            return -sum(trans_res**2)/(2*self.likelihood_params.lambda_1**2*self.likelihood_params.sigma**2)
        
        def gamma_log_likelihood_fixed_sigma(modeled_vals, measured_vals):
            log_likelihood = 0
            for i in range(len(modeled_vals)):

                mu = modeled_vals.values[i]
                val = measured_vals.values[i]
                beta = mu/self.likelihood_params.sigma**2
                a = mu**2/self.likelihood_params.sigma**2
                llhood = stats.gamma.logpdf(val, a, scale=1/beta)
                log_likelihood += llhood
            return log_likelihood

        if self.likelihood_select == "gaussian_fixed_sigma":
            return gaussian_log_likelihood_fixed_sigma
        
        if self.likelihood_select == "gaussian_hetroscedastic_fixed_sigma":
            return gaussian_log_likelihood_hetroscedastic_fixed_sigma
        
        if self.likelihood_select == "gamma_fixed_sigma":
            return gamma_log_likelihood_fixed_sigma