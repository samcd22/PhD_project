import numpy as np
import pandas as pd
import os
import json
from inference_toolbox.parameter import Parameter
import numpyro
import jax.numpy as jnp
from jax import random

class Sampler:
    def __init__(self, params, model, likelihood, data, show_sample_info = False):
        self.params = params
        self.model = model
        self.model_func = model.get_model()
        self.likelihood = likelihood
        self.likelihood_func = likelihood.get_likelihood_function()
        self.data = data
        self.show_sample_info = show_sample_info
        self.sample_info_rows = []
        self.instance = -1

        # if self.show_sample_info:
        # sample_info_row = {}
        # sample_info_row['current_params'] = [x.val for x in self.current_params]
        # sample_info_row['proposed_params'] = [x.val for x in self.proposed_params]
        # sample_info_row['current_log_likelihood'] = curr_log_lhood
        # sample_info_row['proposed_log_likelihood'] = prop_log_lhood
        # sample_info_row['step_forward_log_probs'] = step_forward_log_prob
        # sample_info_row['step_backward_log_probs'] = step_backward_log_prob
        # sample_info_row['current_log_posterior'] = curr_log_posterior
        # sample_info_row['proposed_log_posterior'] = prop_log_posterior
        # sample_info_row['alpha'] = alpha
        # sample_info_row['accepted'] = accepted

        # self.sample_info_rows.append(pd.Series(sample_info_row))

    def sample_all(self, n_samples, n_warmup = -1, n_chains = 1, thinning_rate = 1, rng_key = random.PRNGKey(2120)):
        if n_warmup == -1:
            n_warmup = int(0.25*n_samples)
        data_exists = self.check_data_exists()
        sample_info_file_name = 'most_recent_sample_info.csv'

        if os.path.exists(sample_info_file_name):
            os.remove(sample_info_file_name)
        if not data_exists:
            kernel = numpyro.infer.NUTS(self.sample_one)
            mcmc_obj = numpyro.infer.MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains, thinning=thinning_rate)
            mcmc_obj.run(rng_key=rng_key)
            samples = mcmc_obj.get_samples(group_by_chain=True)
            return self.format_samples(samples)
        else:
            return [], []
        
    def format_samples(self, samples):
        chain_new_samples = pd.DataFrame({}, dtype='float64')
        for param in self.params.index:
            sample_chains = np.array(samples[param])
            chains = range(sample_chains.shape[0])
            chain_array = np.array([])
            chain_new_samples_array = np.array([])
            sample_index_array = np.array([])
            
            for chain in chains:
                chain_new_samples_array = np.concatenate((chain_new_samples_array, sample_chains[chain]))
                chain_array = np.concatenate((chain_array, (chain+1)*np.ones(sample_chains[chain].shape)))
                sample_index_array = np.concatenate((sample_index_array, np.array(range(sample_chains.shape[1]))+1))
            chain_new_samples[param] = chain_new_samples_array
        chain_new_samples['chain'] = chain_array
        chain_new_samples['sample_index'] = sample_index_array

        return chain_new_samples.groupby('sample_index').mean().drop(columns = ['chain']), chain_new_samples
    
    def sample_one(self):
        current_params_sample ={}
        for param_ind in self.params.index:
            s = self.params[param_ind].sample_param()
            current_params_sample[param_ind] = s

        mu = self.model_func(current_params_sample, self.data.x, self.data.y, self.data.z)
        numpyro.deterministic('mu', mu)
        observations_modelled = numpyro.sample('obs', self.likelihood_func(mu, current_params_sample), obs=jnp.array(self.data.Concentration))

    def get_hyperparams(self):
        self.hyperparams = {}
        # Adding Parameter related hyperparameters
        self.hyperparams['params'] = {}
        for param_ind in self.params.index:
            self.hyperparams['params'][param_ind] = {}
            self.hyperparams['params'][param_ind]['prior_func'] = self.params[param_ind].prior_select
            self.hyperparams['params'][param_ind]['prior_params'] = {}
            for prior_param_ind in self.params[param_ind].prior_params.index:
                self.hyperparams['params'][param_ind]['prior_params'][prior_param_ind] = self.params[param_ind].prior_params[prior_param_ind]

        # Adding Model related hyperparameters
        self.hyperparams['model'] = {}
        self.hyperparams['model']['model_func'] = self.model.model_select
        self.hyperparams['model']['model_params'] = {}
        for model_param_ind in self.model.model_params.index:
            self.hyperparams['model']['model_params'][model_param_ind] = self.model.model_params[model_param_ind]
        
        # Adding Likelihood related hyperparameters
        self.hyperparams['likelihood'] = {}
        self.hyperparams['likelihood']['likelihood_func'] = self.likelihood.likelihood_select
        self.hyperparams['likelihood']['likelihood_params'] = {}
        for likelihood_param_ind in self.likelihood.likelihood_params.index:
            self.hyperparams['likelihood']['likelihood_params'][likelihood_param_ind] = self.likelihood.likelihood_params[likelihood_param_ind]
        
        return self.hyperparams
        
    def copy_params(self,params):
        new_params = params.copy()
        for ind in params.index:
            new_params[ind] = new_params[ind].copy()
        return new_params


            
    def check_data_exists(self):
        data_path = 'results/inference'
        data_exists = False
        for instance_folder in os.listdir(data_path):
            folder_path = data_path + '/' + instance_folder
            f = open(folder_path + '/hyperparams.json')
            instance_hyperparams = json.load(f)
            f.close()
            if self.hyperparams == instance_hyperparams:
                data_exists = True
                self.instance = int(instance_folder.split('_')[1])
            return data_exists
            
