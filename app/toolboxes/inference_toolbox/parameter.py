import numpy as np
import pandas as pd
import numpyro
import jax.numpy as jnp

from numpyro import distributions

# Parameter class - used for generating the infered parameters in our sampler
class Parameter:
    # Initialises the Likelihood class saving all relevant variables
    def __init__(self, name, name_2 = None, prior_select = "gaussian", order = 1):
        self.prior_params = pd.Series({},dtype='float64')
        self.name_1 = name
        self.name_2 = name_2
        self.prior_select = prior_select
        self.multivar = False
        if self.name_2:
            self.multivar = True
            self.name = self.name_1 + '_and_' + self.name_2
        else:
            self.name = self.name_1
        self.order = order

        
    # Utility function for converting mean and stdev to alpha in a Gamma distribution 
    def alpha_gamma(self,mu,cov): return mu**2/cov

    # Utility function for converting mean and stdev to beta in a Gamma distribution 
    def beta_gamma(self,mu,cov): return mu/cov

    # Utility function for converting mean and stdev to alpha in a Log Normal distribution 
    def alpha_log_norm(self,mu,cov): 
        if np.isscalar(mu): return np.log(mu) - 0.5*np.log(1+cov/mu**2)
        else:
            mu = np.array(mu)
            cov = np.array(cov)
            alpha = np.zeros(len(mu))
            for i in range(len(mu)):
                alpha[i] = np.log(mu[i]) - 1/2*np.log(cov[i,i]/mu[i]**2 +1)
            return np.array(alpha)

    # Utility function for converting mean and stdev to beta in a Log Normal distribution 
    def beta_log_norm(self,mu,cov):         
        if np.isscalar(mu): return np.sqrt(np.log(1+cov/mu**2))
        else:
            mu = np.array(mu)
            cov = np.array(cov)
            beta = np.zeros((len(mu), len(mu)))
            for i in range(len(mu)):
                for j in range(len(mu)):
                    beta[i,j] = np.log(cov[i,j]/(mu[i]*mu[j])+1)
            return beta

    # Saves a named prior hyperparameters to the Parameter class before generating the parameter's prior distribution
    def add_prior_param(self, name, val):
        self.prior_params[name] = val
        return self

    # Generates the selected prior distribution using the prior hyperparameters
    def get_prior_function(self):
        # Gaussian Prior
        if self.prior_select == 'gaussian':
            if self.multivar:
                raise Exception('Distribution not supported!')
            else:
                return distributions.Normal(self.prior_params.mu, self.prior_params['cov'])
        
        # Gamma Prior
        elif self.prior_select == 'gamma':
            if self.multivar:
                raise Exception('Distribution not supported!')
            else:
                return distributions.Gamma(self.alpha_gamma(self.prior_params.mu,self.prior_params['cov']), self.beta_gamma(self.prior_params.mu,self.prior_params['cov]']))
    
        # Uniform
        elif self.prior_select == 'uniform':
            if self.multivar:
                raise Exception('Distribution not supported!')
            else:
                return distributions.Uniform(self.prior_params.low, self.prior_params.high)
        
        # Log Normal
        elif self.prior_select == 'log_norm':
            if self.multivar:
                multinorm = distributions.MultivariateNormal(self.alpha_log_norm(self.prior_params.mu, self.prior_params['cov']), self.beta_log_norm(self.prior_params.mu, self.prior_params['cov']))
                return distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform())
            else:
                return distributions.LogNormal(self.alpha_log_norm(self.prior_params.mu,self.prior_params['cov']), self.beta_log_norm(self.prior_params.mu,self.prior_params['cov']))
        
        elif self.prior_select == 'multi_mode_log_norm':
            if self.multivar:
                if len(self.prior_params.mus) != len(self.prior_params.covs):
                    raise Exception('Invalid prior parameters!')
                dists = []
                mixture_size = len(self.prior_params.mus)
                mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                for i in range(len(self.prior_params.mus)):
                    multinorm = distributions.MultivariateNormal(self.alpha_log_norm(self.prior_params.mus[i], self.prior_params.covs[i]), self.beta_log_norm(self.prior_params.mus[i], self.prior_params.covs[i]))
                    dists.append(distributions.TransformedDistribution(multinorm, distributions.transforms.ExpTransform()))
                return distributions.MixtureGeneral(mixing_dist, dists)
            else:
                if len(self.prior_params.mus) != len(self.prior_params.covs):
                    raise Exception('Invalid prior parameters!')
                
                dists = []
                mixture_size = len(self.prior_params.mus)
                mixing_dist = distributions.Categorical(probs=jnp.ones(mixture_size) / mixture_size)
                for i in range(len(self.prior_params.mus)):
                    dists.append(distributions.LogNormal(self.alpha_log_norm(self.prior_params.mus[i],self.prior_params.covs[i]), self.beta_log_norm(self.prior_params.mus[i],self.prior_params.covs[i])))
                return distributions.MixtureGeneral(mixing_dist, dists)
        
    # Generates a sample for this parameter
    def sample_param(self):
        prior_func = self.get_prior_function()
        a = numpyro.sample(self.name, prior_func)
        return a, self.order