import numpy as np
import pandas as pd
import os
import json
from numpyro.infer import log_likelihood
import numpyro
import jax.numpy as jnp
from jax import random

# Sampler class - samples inference parameters
class Sampler:
    # Initialises the Sampler class saving all relavant variables and generating a hyperparameters object
    def __init__(self, params, model, likelihood, training_data, testing_data, n_samples, n_warmup = -1, n_chains = 1, thinning_rate = 1, data_path = 'results/inference/general_instances'):
        self.params = params
        self.model = model
        self.model_func = model.get_model()
        self.likelihood = likelihood
        self.likelihood_func = likelihood.get_likelihood_function()
        self.data = training_data
        self.test_data = testing_data
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.n_warmup = n_warmup
        self.thinning_rate = thinning_rate

        self.data_path = data_path
        
        if self.n_warmup == -1:
            self.n_warmup = int(0.25*n_samples)

        self.instance = -1

        self.get_hyperparams()

    # Generates the allotted number of samples
    def sample_all(self, rng_key = random.PRNGKey(2120)):
        data_exists = self.check_data_exists()

        # If data does not already exist, then generate samples
        if not data_exists:
            kernel = numpyro.infer.NUTS(self.sample_one)
            self.mcmc_obj = numpyro.infer.MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples, num_chains=self.n_chains, thinning=self.thinning_rate, chain_method = 'sequential')
            self.mcmc_obj.run(rng_key=rng_key)
            samples = self.mcmc_obj.get_samples(group_by_chain=True)
            self.fields = self.mcmc_obj.get_extra_fields(group_by_chain=True)
            
            self.samples, self.chain_samples = self.format_samples(samples)

            return self.samples, self.chain_samples, self.fields
        else:
            self.samples = []
            self.chain_samples = []
            self.fields = {}
            return [], [], {}
        
    # Puts the samples into the correct format - splitting samples into "all samples" and "chain samples"
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

        return chain_new_samples.sort_values(['sample_index']).reset_index().drop(columns=['chain', 'sample_index','index']), chain_new_samples
    
    # Generates one sample of the infered parameters
    def sample_one(self):
        current_params_sample ={}
        for param_ind in self.params.index:
            s = self.params[param_ind].sample_param()
            current_params_sample[param_ind] = s

        mu = self.model_func(current_params_sample, self.data.x, self.data.y, self.data.z)
        numpyro.deterministic('mu', mu)
        observations_modelled = numpyro.sample('obs', self.likelihood_func(mu, current_params_sample), obs=jnp.array(self.data.Concentration))

    # Generates the hyperparamaters object and saves it to the Sampler class
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
        
        # Adding Sampler related hyperparameters
        self.hyperparams['sampler'] = {}
        self.hyperparams['sampler']['n_samples'] = self.n_samples
        self.hyperparams['sampler']['n_warmup'] = self.n_warmup
        self.hyperparams['sampler']['n_chains'] = self.n_chains
        self.hyperparams['sampler']['thinning_rate'] = self.thinning_rate

        return self.hyperparams
        
    # Creates a copy of all of the inputted parameters
    def copy_params(self,params):
        new_params = params.copy()
        for ind in params.index:
            new_params[ind] = new_params[ind].copy()
        return new_params

    # Function for checking whether a sampler of this configuration has altready been run by 
    # looking through the list of instances and comparing the hyperparameter objects   
    def check_data_exists(self):
        data_exists = False
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        x=os.listdir(self.data_path)
        for instance_folder in os.listdir(self.data_path):
            folder_path = self.data_path + '/' + instance_folder
            f = open(folder_path + '/hyperparams.json')
            instance_hyperparams = json.load(f)
            f.close()
            if self.hyperparams == instance_hyperparams:
                data_exists = True
                self.instance = int(instance_folder.split('_')[1])
        return data_exists
            
