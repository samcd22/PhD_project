import numpy as np
import pandas as pd

# Likelihood Function
class Likelihood:
    def __init__(self, likelihood_select):
        self.likelihood_params = pd.Series({},dtype='float64')
        self.likelihood_select = likelihood_select

    def add_likelihood_param(self,name,val):
        self.likelihood_params[name] = val
    
    def get_log_likelihood_func(self):
        def gaussian_log_likelihood_fixed_sigma(modeled_vals, measured_vals):
            return -np.sum((modeled_vals-measured_vals)**2/(2*self.likelihood_params.sigma**2)) - modeled_vals.size*np.log(np.sqrt(2*np.pi)*self.likelihood_params.sigma)

        def gaussian_log_likelihood_hetroscedastic_fixed_sigma(modeled_vals, measured_vals):
            res = abs(modeled_vals-measured_vals)
            trans_res = ((res+self.likelihood_params.lambda_2)**self.likelihood_params.lambda_1-1)/self.likelihood_params.lambda_1
            return -sum(trans_res**2)/(2*self.likelihood_params.lambda_1**2*self.likelihood_params.sigma**2)
        
        if self.likelihood_select == "gaussian_fixed_sigma":
            return gaussian_log_likelihood_fixed_sigma
        
        if self.likelihood_select == "gaussian_hetroscedastic_fixed_sigma":
            return gaussian_log_likelihood_hetroscedastic_fixed_sigma
        
        def log_gaussian_log_likelihood_fixed_sigma(model_vals,measured_vals):
            return 0