import numpy as np
import pandas as pd
import scipy.stats as stats


class Parameter:
    prior_params = pd.Series({},dtype='float64')
    val = 0
    def __init__(self, init_val, step_select = "" ,step_size = 1, prior_select = ""):
        self.val = init_val
        self.step_select = step_select
        self.step_size = step_size
        self.prior_select = prior_select
        
    def add_prior_param(self, name, val):
        self.prior_params[name] = val
        
    # Step Function
    def get_step_function(self):
        # Probability of step
        def log_p_step_multivariate_gaussian(val, mu):
            return stats.multivariate_normal.logpdf(val, mean=mu, cov=self.step_size)

        # The step itself
        def step_multivariate_positive_gaussian(mu):
            stepped_val = -1
            while stepped_val <= 0:
                stepped_val = stats.multivariate_normal.rvs(mean=mu,cov=self.step_size)
            return stepped_val
        
        if self.step_select == "positive gaussian":
            return log_p_step_multivariate_gaussian, step_multivariate_positive_gaussian
        
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

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        
class Model:
    model_params = pd.Series({},dtype='float64')
    def add_model_param(self,name,val):
        self.model_params[name] = val
    
    # Model Function
    def get_model(self,model_select):
        def GPM(params, x, y, z):
            a = params.a.val
            b = params.b.val
            Q = params.Q.val
            H = self.model_params.H
            u = self.model_params.u
            tmp = 2*a*x**b
            
            return Q / (tmp*np.pi*u)*(np.exp(-(y**2)/tmp))*(np.exp(-(z-H)**2/tmp)+np.exp(-(z+H)**2/tmp))

        if model_select == "GPM":
            return GPM
        
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



# Likelihood Function
class Likelihood:
    likelihood_params = pd.Series({},dtype='float64')
    def add_likelihood_param(self,name,val):
        self.likelihood_params[name] = val
    
    def get_log_likelihood_func(self, likelihood_select):
        def gaussian_log_likelihood_fixed_sigma(modeled_vals, measured_vals):
            return -np.sum((modeled_vals-measured_vals)**2/(2*self.likelihood_params.sigma**2)) - modeled_vals.size*np.log(np.sqrt(2*np.pi)*self.likelihood_params.sigma)

        def gaussian_log_likelihood_hetroscedastic_fixed_sigma(modeled_vals, measured_vals):
            res = abs(modeled_vals-measured_vals)
            trans_res = ((res+self.likelihood_params.lambda_2)**self.likelihood_params.lambda_1-1)/self.likelihood_params.lambda_1
            return -sum(trans_res**2)/(2*self.likelihood_params.lambda_1**2*self.likelihood_params.sigma**2)
        
        if likelihood_select == "gaussian fixed sigma":
            return gaussian_log_likelihood_fixed_sigma
        
        if likelihood_select == "gaussian hetroscedastic fixed sigma":
            return gaussian_log_likelihood_hetroscedastic_fixed_sigma
        
        def log_gaussian_log_likelihood_fixed_sigma(model_vals,measured_vals):
            return 0
    
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

class Sampler:
    def __init__(self, params, model_func, likelihood_func, data, joint_pdf = True):
        self.current_params = params
        self.proposed_params = self.copy_params(params)
        self.model_func = model_func
        self.likelihood_func = likelihood_func
        self.data = data
        self.joint_pdf = joint_pdf
        
    def copy_params(self,params):
        new_params = params.copy()
        for ind in params.index:
            new_params[ind] = new_params[ind].copy()
        return new_params
    
    def accept_params(self, current_log_priors, proposed_log_priors):
        # Calculate the log posterior of the current parameters
        curr_modelled_vals = self.model_func(self.current_params,self.data[:,1],self.data[:,2],self.data[:,3])
        curr_log_lhood = self.likelihood_func(curr_modelled_vals, self.data[:,0])
        curr_log_posterior = curr_log_lhood + current_log_priors
        
         # Calculate the log posterior of the proposed parameters
        prop_modelled_vals = self.model_func(self.proposed_params,self.data[:,1],self.data[:,2],self.data[:,3])
        prop_log_lhood = self.likelihood_func(prop_modelled_vals, self.data[:,0])
        prop_log_posterior = curr_log_lhood + proposed_log_priors
        
        
#             # Calculating the probability of stepping (for the Metropolis Hastings Acceptance Criteria)
#             log_p_step_back, log_p_step_forward = self.get_p_step_back_forward(param_select, current_params, proposed_params)       
        
        # Acceptance criteria
        alpha = np.exp(prop_log_posterior - curr_log_posterior)# + log_p_step_back - log_p_step_forward)

        # Acceptance criteria.
        if np.random.uniform(low = 0, high = 1) < np.min([1,alpha]):
            self.current_params = self.copy_params(self.proposed_params)
            return self.copy_params(self.proposed_params), 1
        else:
            return self.copy_params(self.current_params), 0

    
    def sample_one(self):
        current_log_priors = []
        proposed_log_priors = []

        for i in range(self.current_params.size):
            # Define current parameter
            current_param = self.current_params[i]
            proposed_param = current_param.copy()
            
            # Get functions
            step_log_prob, step_function = current_param.get_step_function()
            log_prior_func = current_param.get_log_prior()
            
            # Step to proposed parameter
            proposed_param.val = step_function(current_param.val)
            
            # Add to series of proposed parameters
            self.proposed_params[i] = proposed_param
                        
            # Create a list of log prior probabilities from each current and proposed parameter
            current_log_priors.append(log_prior_func(current_param.val))
            proposed_log_priors.append(log_prior_func(proposed_param.val))
            
            # Can include non joint PDF here
            
        if self.joint_pdf:
            return self.accept_params(sum(current_log_priors), sum(proposed_log_priors))
            
            
    def sample_all(self, n_samples):
        samples = []
        samples_means = []
        for i in range(1,n_samples+1):
            if (i % 1000 == 0):
                print('Running sample ' + str(i) + '...')    # Print progress every 1000th sample.
            sample, accept = self.sample_one()
            samples.append(sample)
        samples = pd.DataFrame(samples)

        return samples
    
    def get_mean_samples(self,samples):
        means = pd.Series({},dtype='float64')
        for col in samples.columns:
            mean = samples[col].apply(lambda x: x.val).mean()
            means[col] = Parameter(mean)
        return means
    