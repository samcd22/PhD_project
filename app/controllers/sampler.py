import os
import json
import numpyro
import jax.numpy as jnp
from numpyencoder import NumpyEncoder
import pandas as pd
import numpy as np
from jaxlib.xla_extension import ArrayImpl
from jax import random
import jax

from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from data_processing.data_processor import DataProcessor


class Sampler:
    """
    A class for performing sampling using the NUTS algorithm. The Sampler class is used to generate samples from the posterior distribution of the inference parameters.

    Args:
        - inference_params (pd.Series): A pandas Series containing the inference parameters. Each element is a parameter object
        - model (Model): The model object.
        - likelihood (Likelihood): The likelihood object.
        - data_processor (RawDataProcessor, SimDataProcessor): The data processor object.
        - n_samples (int, optional): The number of samples to draw. Defaults to 10000.
        - p_warmup (float, optional): The proportion of warmup samples. Defaults to 0.5. So if p_warmup is 0.5 then 50% more samples will be drawn and thrown out before the actual sampling process begins.
        - n_chains (int, optional): The number of chains to run in parallel. Defaults to 1. If set to greater than 1, the sampler will be run multiple times beginning in different locations in the parameter space.
        - thinning_rate (int, optional): The thinning rate for the samples. Defaults to 1. If thinning_rate is 1, all samples are saved. If thinning_rate is 2, every second sample is saved, and so on.
        - root_results_path (str, optional): The root path to save the results. Defaults to '/results/inference_results'.
        - controller (str, optional): The controller type. Must be one of 'sandbox', 'generator', or 'optimisor'. Defaults to 'sandbox'.
        - generator_name (str, optional): The name of the generator. Required if controller is 'generator'. Defaults to None.
        - optimiser_name (str, optional): The name of the optimiser. Required if controller is 'optimisor'. Defaults to None.


    Attributes:
        - inference_params (pd.Series): A pandas Series containing the inference parameters.
        - n_samples (int): The number of samples to draw.
        - n_chains (int): The number of chains to run in parallel.
        - p_warmup (float): The proportion of warmup samples.
        - n_warmup (int): The number of warmup samples.
        - thinning_rate (int): The thinning rate for the samples.
        - data_processor (RawDataProcessor, SimDataProcessor): The data processor object.
        - model (Model): The model object.
        - likelihood (Likelihood): The likelihood object.
        - training_data (pd.DataFrame): The training data.
        - testing_data (pd.DataFrame): The testing data.
        - data_construction (dict): The construction of the data processor.
        - model_func (callable): The model function.
        - independent_variables (list): The independent variable names of the model.
        - dependent_variables (list): The dependent variable names of the model.
        - all_model_param_names (list): All parameter names of the model.
        - fixed_model_params (pd.Series): The fixed model parameters.
        - likelihood_func (callable): The likelihood function.
        - fixed_likelihood_params (pd.Series): The fixed likelihood parameters.
        - results_path (str): The path to save the results.
        - root_results_path (str): The root path to save the results.
        - sampler_construction (dict): The construction of the sampler.
        - data_exists (bool): Flag indicating if the data exists.
        - instance (int): The instance number.
        - sampled (bool): Flag indicating if the sampler has been sampled.
        - samples (pd.DataFrame): The samples.
        - chain_samples (pd.DataFrame): The chain samples.
        - fields (dict): The fields dictionary.

    """

    def __init__(self, inference_params: pd.Series, 
                 model: Model, 
                 likelihood:Likelihood, 
                 data_processor:DataProcessor, 
                 n_samples:int=10000, 
                 p_warmup:float=0.5, 
                 n_chains:int=1, 
                 thinning_rate:int=1, 
                 root_results_path:str='/results/inference_results', 
                 controller:str='sandbox', 
                 generator_name:str=None, 
                 optimiser_name:str=None):
        """
        Initializes the Sampler class.

        Args:
            - inference_params (pd.Series): A pandas Series containing the inference parameters. Each element is a parameter object
            - model: The model object.
            - likelihood: The likelihood object.
            - data_processor: The data processor object.
            - n_samples (int, optional): The number of samples to draw. Defaults to 10000.
            - p_warmup (float, optional): The proportion of warmup samples. Defaults to 0.5.
            - n_chains (int, optional): The number of chains to run in parallel. Defaults to 1.
            - thinning_rate (int, optional): The thinning rate for the samples. Defaults to 1.
            - root_results_path (str, optional): The root path to save the results. Defaults to '/results/inference_results'.
            - controller (str, optional): The controller type. Must be one of 'sandbox', 'generator', or 'optimisor'. Defaults to 'sandbox'.
            - generator_name (str, optional): The name of the generator. Required if controller is 'generator'. Defaults to None.
            - optimiser_name (str, optional): The name of the optimiser. Required if controller is 'optimisor'. Defaults to None.
        """
        
        root_results_path = os.getcwd() + root_results_path

        if not isinstance(inference_params, pd.Series):
            raise TypeError("Sampler - inference_params must be a pandas Series")
        if not isinstance(model, Model):
            raise TypeError("Sampler - model must be an instance of the Model class")
        if not isinstance(likelihood, Likelihood):
            raise TypeError("Sampler - likelihood must be an instance of the Likelihood class")
        if not isinstance(data_processor, DataProcessor):
            raise TypeError("Sampler - data_processor must be an instance of the DataProcessor class")
        if not isinstance(n_samples, int) and n_samples > 0:
            raise TypeError("Sampler - n_samples must be a positive integer")
        if not (isinstance(p_warmup, float) and 0 <= p_warmup <= 1) and not (isinstance(p_warmup, int) and 0 <= p_warmup <= 1):
            raise TypeError("Sampler - p_warmup must be a float or an int between 0 and 1")
        if not isinstance(n_chains, int) and n_chains > 0:
            raise TypeError("Sampler - n_chains must be a positive integer")
        if not isinstance(thinning_rate, int) and thinning_rate > 0:
            raise TypeError("Sampler - thinning_rate must be a positive integer")
        if not isinstance(root_results_path, str):
            raise TypeError("Sampler - root_results_path must be a string")
        if not isinstance(controller, str):
            raise TypeError("Sampler - controller must be a string")
        if not isinstance(generator_name, str) and generator_name is not None:
            raise TypeError("Sampler - generator_name must be a string or None")
        if not isinstance(optimiser_name, str) and optimiser_name is not None:
            raise TypeError("Sampler - optimiser_name must be a string or None")

        self.inference_params = inference_params
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.p_warmup = p_warmup
        self.n_warmup = int(p_warmup * n_samples)
        self.thinning_rate = thinning_rate
        
        self.data_processor = data_processor
        self.model = model
        self.likelihood = likelihood

        self.training_data, self.testing_data = data_processor.process_data()
        self.data_construction = data_processor.get_construction()
        
        self.model_func = model.get_model()
        self.independent_variables = model.independent_variables
        self.dependent_variables = model.dependent_variables
        self.all_model_param_names = model.all_param_names
        self.fixed_model_params = model.fixed_model_params

        self.likelihood_func = likelihood.get_likelihood_function()
        self.fixed_likelihood_params = likelihood.fixed_likelihood_params
       
        self.samples = None
        self.chain_samples = None
        self.fields = None

        self.sampled = False

        if controller == 'sandbox':
            self.results_path = os.path.join(root_results_path, data_processor.processed_data_name, 'general_instances')
        elif controller == 'generator':
            self.results_path = os.path.join(root_results_path, data_processor.processed_data_name, 'auto_gen_instances', generator_name)
        elif controller == 'optimisor':
            self.results_path = os.path.join(root_results_path, data_processor.processed_data_name, 'auto_gen_instances', optimiser_name)
        else:
            raise ValueError('Controller must be one of "sandbox", "generator" or "optimisor"')

        os.makedirs(self.results_path, exist_ok=True)

        self.root_results_path = root_results_path
        self.sampler_construction = self.get_construction()

        self.data_exists, self.instance = self._check_data_exists()
        self._check_data_validity(self.training_data)
        self._check_data_validity(self.testing_data)

    def get_construction(self):
        """
        Get the construction of the sampler. The conctruction parameters includes all of the config information used to construct the sampler object. It includes:
            - inference_params: The inference parameters.
            - model: The construction of the model object.
            - likelihood: The construction of the likelihood object.
            - data_processor: The construction of the data processor object.
            - n_samples: The number of samples to draw.
            - n_chains: The number of chains to run in parallel.
            - thinning_rate: The thinning rate for the samples.
            - p_warmup: The proportion of warmup samples.
        """
        construction = {
            'inference_params': [param.get_construction() for param in self.inference_params],
            'model': self.model.get_construction(),
            'likelihood': self.likelihood.get_construction(),
            'data_processor': self.data_processor.get_construction(),
            'n_samples': self.n_samples,
            'n_chains': self.n_chains,
            'thinning_rate': self.thinning_rate,
            'p_warmup': self.p_warmup,
        }
        return construction

    def _save_samples(self, samples, chain_samples, fields):
        """
        Save the samples, chain samples, and fields to disk.

        Args:
            - samples (pd.DataFrame): The samples.
            - chain_samples (pd.DataFrame): The chain samples.
            - fields (dict): The fields dictionary.
        """
        instance_folder = os.path.join(self.results_path, f'instance_{self.instance}')
        os.makedirs(instance_folder, exist_ok=True)
        samples.to_csv(os.path.join(instance_folder, 'samples.csv'))
        chain_samples.to_csv(os.path.join(instance_folder, 'chain_samples.csv'))

        
        with open(os.path.join(instance_folder, 'fields.json'), 'w') as fp:
            json.dump(fields, fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)
        
        with open(os.path.join(instance_folder, 'sampler_construction.json'), "w") as fp:
            json.dump(self.sampler_construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

        print('Samples saved to ' + instance_folder)

    def _load_samples(self):
        """
        Load the samples, chain samples, and fields from disk.

        """
        instance_folder = os.path.join(self.results_path, f'instance_{self.instance}')
        try:
            samples = pd.read_csv(os.path.join(instance_folder, 'samples.csv'))
        except:
            raise FileNotFoundError('Sampler - samples.csv not found')
        try:
            chain_samples = pd.read_csv(os.path.join(instance_folder, 'chain_samples.csv'))
        except:
            raise FileNotFoundError('Sampler - chain_samples.csv not found')
        try:
            with open(os.path.join(instance_folder, 'fields.json'), 'r') as f:
                fields = json.load(f)
        except:
            raise FileNotFoundError('Sampler - fields.json not found')
        
        if 'Unnamed: 0' in samples.columns:
            samples.drop(columns=['Unnamed: 0'], inplace=True)
        if 'Unnamed: 0' in chain_samples.columns:
            chain_samples.drop(columns=['Unnamed: 0'], inplace=True)

        print('Samples loaded from ' + instance_folder)
        
        return samples, chain_samples, fields

    def _check_data_validity(self, data):
        """
        Check the validity of the data.

        Args:
            - data: The data to check.

        """
        if data.isnull().values.any():
            raise Exception('Sampler - Data contains missing values!')

        if not set(self.independent_variables).issubset(data.columns):
            raise Exception('Sampler - Data does not contain all independent variables of the model!')
        if not set(self.dependent_variables).issubset(data.columns):
            raise Exception('Sampler - Data does not contain all dependent variables of the model!')

    def sample_one(self):
        """
        Sample one set of parameters. This function is used by the MCMC sampler to generate samples from the posterior distribution. The sampler itterated through this function to generate the samples.

        """
        current_params_sample = {}
        for param_ind in self.inference_params.index:
            sample = self.inference_params[param_ind].sample_param()
            if jnp.isscalar(sample):
                current_params_sample[param_ind] = jax.lax.cond(
                    sample > 0,
                    lambda sample: self._safe_exponentiation(sample.astype(jnp.float32), self.inference_params[param_ind].order),
                    lambda sample: sample.astype(jnp.float32),  # Ensure the scalar stays as float32 in both branches
                    sample
                )            
            else:
                current_params_sample[param_ind] = jax.lax.cond(
                    jnp.all(sample > jnp.zeros(sample.shape, dtype=jnp.float32)),  # Ensure `sample` is float32
                    lambda sample: self._safe_exponentiation(sample.astype(jnp.float32), self.inference_params[param_ind].order),  # Ensure this returns float32
                    lambda sample: sample.astype(jnp.float32),  # Ensure the false branch returns float32
                    sample
                )        
        current_params_sample = self._format_params(current_params_sample)

        mu = self.model_func(current_params_sample, self.training_data)
        mu = jnp.where(jnp.isinf(-1*mu), -1e20, mu)
        
        try:
            self.likelihood_func(mu, current_params_sample)
        except:
            print('Sampler - Error in likelihood function')

        mu = numpyro.deterministic('mu', mu)
        with numpyro.plate('data', len(self.training_data[self.dependent_variables[0]])):
            numpyro.sample('obs', self.likelihood_func(mu, current_params_sample), obs=jnp.array(self.training_data[self.dependent_variables[0]].values))

    def _safe_exponentiation(self, base, exponent):
        """
        Use logarithms to handle large exponents safely

        Returns:
            - float: The result of the exponentiation.

        """
        res = jnp.exp(jnp.log(base) + exponent * jnp.log(10))
        return res

    def sample_all(self, rng_key=random.PRNGKey(0)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Sample all sets of parameters. This function is used to generate all samples from the posterior distribution.

        Args:
            - rng_key: The random number generator key.

        Returns:
            - pd.DataFrame: The samples.
            - pd.DataFrame: The chain samples.
            - dict: The fields dictionary.
        """
        if not self.data_exists:
            kernel = numpyro.infer.NUTS(self.sample_one)
            mcmc = numpyro.infer.MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples, num_chains=self.n_chains, thinning=self.thinning_rate, chain_method='parallel')
            with numpyro.validation_enabled():
                mcmc.run(rng_key=rng_key)
            samples = mcmc.get_samples(group_by_chain=True)
            fields = mcmc.get_extra_fields(group_by_chain=True)
            fields = self._convert_arrays_to_lists(fields)
            samples, chain_samples = self._format_samples(samples)
            self._save_samples(samples, chain_samples, fields)
            self.samples, self.chain_samples, self.fields = samples, chain_samples, fields, 
        else:
            self.samples, self.chain_samples, self.fields = self._load_samples()
        self.sampled = True
        return self.samples, self.chain_samples, self.fields
  
    def _format_samples(self, samples):
        """
        Format the samples.

        Args:
            - samples (dict): The unformatted samples.
        
        """

        chain_new_samples = pd.DataFrame({}, dtype='float64')
        for param in self.inference_params.index:

            if '_and_' in param:

                sub_params = param.split('_and_')
                for i in range(len(sub_params)):
                    chain_array = np.array([])
                    chain_new_samples_array = np.array([])
                    sample_index_array = np.array([])

                    sub_param = sub_params[i]
                    sample_chains = samples[param][:,:,i]
                    chains = range(sample_chains.shape[0])
                
                    for chain in chains:
                        chain_new_samples_array = np.concatenate((chain_new_samples_array, sample_chains[chain]))
                        chain_array = np.concatenate((chain_array, (chain+1)*np.ones(sample_chains[chain].shape)))
                        sample_index_array = np.concatenate((sample_index_array, np.array(range(sample_chains.shape[1]))+1))
                    
                    chain_new_samples[sub_param] = chain_new_samples_array
                    chain_new_samples[sub_param]=chain_new_samples[sub_param]*10**self.inference_params[param].order
            else:
                chain_array = np.array([])
                chain_new_samples_array = np.array([])
                sample_index_array = np.array([])
                sample_chains = np.array(samples[param])
                chains = range(sample_chains.shape[0])
                
                for chain in chains:
                    a = sample_chains[chain]
                    chain_new_samples_array = np.concatenate((chain_new_samples_array, sample_chains[chain]))
                    chain_array = np.concatenate((chain_array, (chain+1)*np.ones(sample_chains[chain].shape)))
                    sample_index_array = np.concatenate((sample_index_array, np.array(range(sample_chains.shape[1]))+1))
                
                chain_new_samples[param] = chain_new_samples_array
                chain_new_samples[param]=chain_new_samples[param]*10**self.inference_params[param].order
        
        chain_new_samples['chain'] = chain_array
        chain_new_samples['sample_index'] = sample_index_array

        formatted_samples = chain_new_samples.sort_values(['sample_index']).reset_index().drop(columns=['chain', 'sample_index','index'])

        return formatted_samples, chain_new_samples

    def _format_params(self, current_params_sample):
        """
        Format the current parameter samples.

        Args:
            - current_params_sample: The current parameter samples.

        """
        formatted_current_params_sample = {}
        for param_ind in self.inference_params.index:
            if 'and' in param_ind:
                names = param_ind.split('_and_')
                for i in range(len(names)):
                    temp = current_params_sample[param_ind]
                    formatted_current_params_sample[names[i]] = temp[i]
            else:
                formatted_current_params_sample[param_ind] = current_params_sample[param_ind]
        return formatted_current_params_sample

    def _check_data_exists(self):
        """
        Check if the data exists.

        Returns:
            bool: Flag indicating if the data exists.
            int: The instance number where the data is to be saved if the data does not exist. Or the instance number where the data is saved if the data exists.
        """

        data_exists = False
        if not os.path.exists(self.results_path):
            return False, 1
        instance_folders = os.listdir(self.results_path)
        for instance_folder in os.listdir(self.results_path):
            folder_path = self.results_path + '/' + instance_folder
            with open(folder_path + '/sampler_construction.json') as f:
                instance_hyperparams = json.load(f)
            if self._convert_arrays_to_lists(self.get_construction()) == instance_hyperparams:
                data_exists = True
                instance = int(instance_folder.split('_')[1])
        if not data_exists:
            instances = [int(folder.split('_')[1]) for folder in instance_folders]
            instance = 1 if len(instances) == 0 else min(i for i in range(1, max(instances) + 2) if i not in instances)
        return data_exists, instance
    
    def _convert_arrays_to_lists(self, obj):
        """
        Utility function which recursively converts NumPy arrays to lists within a dictionary.

        Args:
            - obj (dict or list or np.ndarray): The input object to convert.

        Returns:
            - dict or list or list: The converted object with NumPy arrays replaced by lists.
        """
        if isinstance(obj, dict):
            # If the object is a dictionary, convert arrays within its values
            return {k: self._convert_arrays_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # If the object is a list, convert arrays within its elements
            return [self._convert_arrays_to_lists(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            # If the object is a NumPy array, convert it to a list
            return obj.tolist()
        elif isinstance(obj, ArrayImpl):
            return np.asarray(obj).tolist()
        else:
            # If the object is neither a dictionary, list, nor NumPy array, return it unchanged
            return obj
