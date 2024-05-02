import numpy as np
import pandas as pd
import os
import json
import numpyro
import jax.numpy as jnp
from jax import random

from toolboxes.inference_toolbox.model import Model
from toolboxes.inference_toolbox.likelihood import Likelihood

import os
# cpu cores available for sampling (we want this to equal num_chains)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

class Sampler:
    """
    The Sampler class is used to sample inference parameters using MCMC (Markov Chain Monte Carlo) methods.

    Attributes:
    - params (pandas.Index): Index of parameters.
    - model (Model): The model object.
    - likelihood (Likelihood): The likelihood object.
    - training_data (numpy.ndarray): The training data.
    - testing_data (numpy.ndarray): The testing data.
    - n_samples (int): Number of samples to generate.
    - p_warmup (int, optional): Percentage of warmup samples. Defaults to 0.5.
    - n_chains (int, optional): Number of chains. Defaults to 1.
    - thinning_rate (int, optional): Thinning rate for samples. Defaults to 1.
    - data_path (str, optional): Path to save the results. Defaults to 'results/inference/general_instances'.
    - actual_parameter_names (list): A list of the actual parameter names.
    - instance (int): The instance number.
    - hyperparams (dict): A dictionary containing the hyperparameters for the inference.

    Methods:
    - sample_all(rng_key): Generates the allotted number of samples.
    - sample_one(): Generates one sample of the inferred parameters.
    - get_hyperparams(): Generates the hyperparameters object and saves it to the Sampler class.
    - copy_params(params): Creates a copy of all of the inputted parameters.
    - check_data_exists(): Function for checking whether a sampler of this configuration has already been run.
    """

    class Sampler:
        def __init__(self, 
                     params: pd.Series, 
                     model: Model, 
                     likelihood: Likelihood, 
                     training_data: pd.Series, 
                     testing_data: pd.Series, 
                     n_samples: int, 
                     p_warmup:float=0.5,
                     n_chains:int=1, 
                     thinning_rate:float=1, 
                     data_path:str='results/inference/general_instances'):
            """
            Initialize the Sampler object.

            Args:
            - params (pandas.Index): Index of parameters.
            - model (Model): The model object.
            - likelihood (Likelihood): The likelihood object.
            - training_data (numpy.ndarray): The training data.
            - testing_data (numpy.ndarray): The testing data.
            - n_samples (int): Number of samples to generate.
            - p_warmup (int, optional): Percentage of warmup samples. Defaults to 0.5.
            - n_chains (int, optional): Number of chains. Defaults to 1.
            - thinning_rate (int, optional): Thinning rate for samples. Defaults to 1.
            - data_path (str, optional): Path to save the results. Defaults to 'results/inference/general_instances'.
            """
            self.params = params
            self.model = model
            self.model_func = model.get_model()
            self.likelihood = likelihood
            self.likelihood_func = likelihood.get_likelihood_function()
            self.data = training_data
            self.test_data = testing_data
            self.n_samples = n_samples
            self.n_chains = n_chains
            self.pwarmup = p_warmup
            self.n_warmup = p_warmup*n_samples
            self.thinning_rate = thinning_rate

            self.actual_parameter_names = []
            for param in self.params.index:
                if '_and_' in param:
                    names = param.split('_and_')
                    for i in range(len(names)):
                        self.actual_parameter_names.append(names[i])
                else:
                    self.actual_parameter_names.append(param)

            self.data_path = data_path


            self.instance = -1

            self.get_hyperparams()

    def check_data(self, data):
        """
        Checks the data for any missing values.

        Args:
        - data (pd.DataFrame): The data to be checked.
        """

        if data.isnull().values.any():
            raise Exception('Data contains missing values!')
        
        if self.model.model_type == 'scalar_spatial_3D':
            if not 'x' in data.columns or not 'y' in data.columns or not 'z' in data.columns:
                raise Exception('Data does not contain x, y, z columns!')
            if not self.model.output_name in data.columns:
                raise Exception('Data does not contain '+  self.model.output_name +' column!')    


    def sample_all(self, rng_key=random.PRNGKey(2120)):
        """
        Generates the allotted number of samples.

        Parameters:
        - rng_key (jax.random.PRNGKey, optional): The random number generator key. Defaults to random.PRNGKey(2120).

        Returns:
        - samples (pd.DataFrame): A DataFrame containing all the samples.
        - chain_samples (pd.DataFrame): A DataFrame containing the samples for each chain.
        - fields (dict): A dictionary containing the extra fields.
        """
        data_exists = self.check_data_exists()

        if not data_exists:
            kernel = numpyro.infer.NUTS(self.sample_one)
            self.mcmc_obj = numpyro.infer.MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples, num_chains=self.n_chains, thinning=self.thinning_rate, chain_method='sequential')
            self.mcmc_obj.run(rng_key=rng_key)
            samples = self.mcmc_obj.get_samples(group_by_chain=True)
            self.fields = self.mcmc_obj.get_extra_fields(group_by_chain=True)

            self.samples, self.chain_samples = self._format_samples(samples)

            return self.samples, self.chain_samples, self.fields
        else:
            self.samples = []
            self.chain_samples = []
            self.fields = {}
            return [], [], {}

    def _format_samples(self, samples):
        """
        Puts the samples into the correct format - splitting samples into "all samples" and "chain samples".

        Parameters:
        - samples (dict): A dictionary containing the samples.

        Returns:
        - pd.DataFrame: A DataFrame containing the samples for each chain as well as the total samples.
        """
        chain_new_samples = pd.DataFrame({}, dtype='float64')
        for param in self.params.index:
            if '_and_' in param:
                sub_params = param.split('_and_')
                for i in range(len(sub_params)):
                    chain_array = np.array([])
                    chain_new_samples_array = np.array([])
                    sample_index_array = np.array([])

                    sub_param = sub_params[i]
                    sample_chains = samples[param][:, :, i]
                    chains = range(sample_chains.shape[0])

                    for chain in chains:
                        chain_new_samples_array = np.concatenate((chain_new_samples_array, sample_chains[chain]))
                        chain_array = np.concatenate((chain_array, (chain + 1) * np.ones(sample_chains[chain].shape)))
                        sample_index_array = np.concatenate((sample_index_array, np.array(range(sample_chains.shape[1])) + 1))

                    chain_new_samples[sub_param] = chain_new_samples_array
                    chain_new_samples[sub_param] *= self.params[param].order

            else:
                chain_array = np.array([])
                chain_new_samples_array = np.array([])
                sample_index_array = np.array([])
                sample_chains = np.array(samples[param])
                chains = range(sample_chains.shape[0])

                for chain in chains:
                    chain_new_samples_array = np.concatenate((chain_new_samples_array, sample_chains[chain]))
                    chain_array = np.concatenate((chain_array, (chain + 1) * np.ones(sample_chains[chain].shape)))
                    sample_index_array = np.concatenate((sample_index_array, np.array(range(sample_chains.shape[1])) + 1))

                chain_new_samples[param] = chain_new_samples_array
                chain_new_samples[param] *= self.params[param].order

        chain_new_samples['chain'] = chain_array
        chain_new_samples['sample_index'] = sample_index_array

        return chain_new_samples.sort_values(['sample_index']).reset_index().drop(columns=['chain', 'sample_index', 'index']), chain_new_samples

    def _format_params(self, current_params_sample):
        """
        Formats the current parameter sample.

        Parameters:
        - current_params_sample (dict): A dictionary containing the current parameter sample.

        Returns:
        - dict: A dictionary containing the formatted current parameter sample.
        """
        formatted_current_params_sample = {}

        for param_ind in self.params.index:
            if 'and' in param_ind:
                names = param_ind.split('_and_')
                for i in range(len(names)):
                    temp = current_params_sample[param_ind]
                    formatted_current_params_sample[names[i]] = temp[i]
            else:
                formatted_current_params_sample[param_ind] = current_params_sample[param_ind]
        return formatted_current_params_sample

    def sample_one(self):
        """
        Generates one sample of the inferred parameters.

        Returns:
        - numpyro.distributions.Distribution: One sample of the parameters.
        """
        current_params_sample = {}
        for param_ind in self.params.index:
            s, order = self.params[param_ind].sample_param()
            current_params_sample[param_ind] = s * order

        current_params_sample = self._format_params(current_params_sample)
        if self.model.model_type == 'scalar_spatial_3D':
            mu = self.model_func(current_params_sample, self.data.x, self.data.y, self.data.z)

        mu = numpyro.deterministic('mu', mu)
        return numpyro.sample('obs', self.likelihood_func(mu, current_params_sample), obs=jnp.array(self.data.Concentration))

    def get_hyperparams(self):
        """
        Generates the hyperparameters object and saves it to the Sampler class.

        Returns:
        - dict: A dictionary containing the hyperparameters for the inference.
        """
        self.hyperparams = {}
        self.hyperparams['params'] = {}
        for param_ind in self.params.index:
            self.hyperparams['params'][param_ind] = {}
            self.hyperparams['params'][param_ind]['prior_func'] = self.params[param_ind].prior_select
            self.hyperparams['params'][param_ind]['prior_params'] = {}
            for prior_param_ind in self.params[param_ind].prior_params.index:
                self.hyperparams['params'][param_ind]['prior_params'][prior_param_ind] = self.params[param_ind].prior_params[prior_param_ind] * self.params[param_ind].order

        self.hyperparams['model'] = {}
        self.hyperparams['model']['model_func'] = self.model.model_select
        self.hyperparams['model']['model_params'] = {}
        for model_param_ind in self.model.model_params.index:
            self.hyperparams['model']['model_params'][model_param_ind] = self.model.fixed_model_params[model_param_ind]

        self.hyperparams['likelihood'] = {}
        self.hyperparams['likelihood']['likelihood_func'] = self.likelihood.likelihood_select
        self.hyperparams['likelihood']['likelihood_params'] = {}
        for likelihood_param_ind in self.likelihood.likelihood_params.index:
            self.hyperparams['likelihood']['likelihood_params'][likelihood_param_ind] = self.likelihood.likelihood_params[likelihood_param_ind]

        self.hyperparams['sampler'] = {}
        self.hyperparams['sampler']['n_samples'] = self.n_samples
        self.hyperparams['sampler']['n_warmup'] = self.n_warmup
        self.hyperparams['sampler']['n_chains'] = self.n_chains
        self.hyperparams['sampler']['thinning_rate'] = self.thinning_rate

        return self.hyperparams

    def copy_params(self, params):
        """
        Creates a copy of all of the inputted parameters.

        Args:
        - params: The parameters to be copied.

        Returns:
        - pd.Series: A copy of the inputted parameters.
        """
        pass
        new_params = params.copy()
        for ind in params.index:
            new_params[ind] = new_params[ind].copy()
        return new_params

    def check_data_exists(self):
        """
        Function for checking whether a sampler of this configuration has altready been run by 
        looking through the list of instances and comparing the hyperparameter objects 

        Returns:
        - bool: Whether the data exists or not.  
        """
        data_exists = False
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        for instance_folder in os.listdir(self.data_path):
            folder_path = self.data_path + '/' + instance_folder
            f = open(folder_path + '/hyperparams.json')
            instance_hyperparams = json.load(f)
            f.close()
            if self.hyperparams == instance_hyperparams:
                data_exists = True
                self.instance = int(instance_folder.split('_')[1])
        return data_exists
            
