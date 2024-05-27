import os
import json
import numpyro
import jax.numpy as jnp
from jax import random, jit
from numpyencoder import NumpyEncoder
import pandas as pd
import numpy as np

# CPU cores available for sampling (we want this to equal num_chains)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

class Sampler:
    def __init__(self, inference_params: pd.Series, model, likelihood, data_processor, n_samples=10000, p_warmup=0.5, n_chains=1, thinning_rate=1, root_results_path='/PhD_project/results/inference_results', controller='sandbox', generator_name=None, optimiser_name=None):
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
        construction = {
            'inference_params': [param.get_construction() for param in self.inference_params],
            'model': self.model.get_construction(),
            'likelihood': self.likelihood.get_construction(),
            'data_processor': self.data_processor.get_construction(),
            'n_samples': self.n_samples,
            'n_chains': self.n_chains,
            'thinning_rate': self.thinning_rate,
        }
        return construction

    def _save_samples(self, samples, chain_samples, fields):
        instance_folder = os.path.join(self.results_path, f'instance_{self.instance}')
        os.makedirs(instance_folder, exist_ok=True)
        samples.to_csv(os.path.join(instance_folder, 'samples.csv'))
        chain_samples.to_csv(os.path.join(instance_folder, 'chain_samples.csv'))
        fields = json.dumps(fields, cls=NumpyEncoder, separators=(', ',': '), indent=4)
        
        with open(os.path.join(instance_folder, 'fields.json'), 'w') as fp:
            json.dump(fields, fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)
        
        with open(os.path.join(instance_folder, 'sampler_construction.json'), "w") as fp:
            json.dump(self.sampler_construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

    def _load_samples(self):
        instance_folder = os.path.join(self.results_path, f'instance_{self.instance}')
        samples = pd.read_csv(os.path.join(instance_folder, 'samples.csv'))
        chain_samples = pd.read_csv(os.path.join(instance_folder, 'chain_samples.csv'))
        with open(os.path.join(instance_folder, 'fields.json'), 'r') as f:
            fields = json.load(f)
        return samples, chain_samples, fields

    def _check_data_validity(self, data):
        if data.isnull().values.any():
            raise Exception('Sampler - Data contains missing values!')
        if not set(self.independent_variables).issubset(data.columns):
            raise Exception('Sampler - Data does not contain all independent variables of the model!')
        if not set(self.dependent_variables).issubset(data.columns):
            raise Exception('Sampler - Data does not contain all dependent variables of the model!')

    def sample_one(self):
        current_params_sample = {}
        for param_ind in self.inference_params.index:
            sample = self.inference_params[param_ind].sample_param()
            current_params_sample[param_ind] = sample * self.inference_params[param_ind].order

        current_params_sample = self._format_params(current_params_sample)

        mu = self.model_func(current_params_sample, self.training_data)
        mu = numpyro.deterministic('mu', mu)
        return numpyro.sample('obs', self.likelihood_func(mu, current_params_sample), obs=jnp.array(self.training_data[self.dependent_variables[0]].values))

    def sample_all(self, rng_key=random.PRNGKey(2120)):
        if not self.data_exists:
            kernel = numpyro.infer.NUTS(self.sample_one)
            mcmc = numpyro.infer.MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples, num_chains=self.n_chains, thinning=self.thinning_rate, chain_method='parallel')
            mcmc.run(rng_key=rng_key)
            samples = mcmc.get_samples(group_by_chain=True)
            fields = mcmc.get_extra_fields(group_by_chain=True)
            fields = self._format_fields(fields)
            samples, chain_samples = self._format_samples(samples)
            self._save_samples(samples, chain_samples, fields)
            return samples, chain_samples, fields
        else:
            return self._load_samples()

    def _format_fields(self, fields):
        # Assuming fields is your dictionary containing serializable and non-serializable objects
        fields_serializable = {}

        for key, value in fields.items():
            if isinstance(value, np.ndarray):
                # Convert NumPy arrays to lists
                fields_serializable[key] = value.tolist()
            elif hasattr(value, 'to_dict'):
                # Check if the object has a to_dict method (e.g., pandas DataFrame)
                fields_serializable[key] = value.to_dict()
            elif isinstance(value, (int, float, str, list, dict)):
                # Keep other serializable objects as is
                fields_serializable[key] = value
            else:
                # For unsupported types, convert to string representation
                fields_serializable[key] = str(value)
                
        return fields_serializable

    def _format_samples(self, samples):
        chain_new_samples = pd.DataFrame({}, dtype='float64')
        for param in self.inference_params.index:
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
                    chain_new_samples[sub_param] *= self.inference_params[param].order
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
        return pd.DataFrame(chain_new_samples), pd.DataFrame(chain_new_samples)

    def _format_params(self, current_params_sample):
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

    def copy_params(self, params):
        new_params = params.copy()
        for ind in params.index:
            new_params[ind] = new_params[ind].copy()
        return new_params

    def _check_data_exists(self):
        data_exists = False
        if not os.path.exists(self.results_path):
            return False, 1
        instance_folders = os.listdir(self.results_path)
        for instance_folder in os.listdir(self.results_path):
            folder_path = self.results_path + '/' + instance_folder
            with open(folder_path + '/sampler_construction.json') as f:
                instance_hyperparams = json.load(f)
            if self.convert_arrays_to_lists(self.get_construction()) == instance_hyperparams:
                data_exists = True
                instance = int(instance_folder.split('_')[1])
        if not data_exists:
            instances = [int(folder.split('_')[1]) for folder in instance_folders]
            instance = 1 if len(instances) == 0 else min(i for i in range(1, max(instances) + 2) if i not in instances)
        return data_exists, instance
    
    def convert_arrays_to_lists(self, obj):
        """
        Recursively convert NumPy arrays to lists within a dictionary.

        Args:
        - obj (dict or list or np.ndarray): The input object to convert.

        Returns:
        - dict or list or list: The converted object with NumPy arrays replaced by lists.
        """
        if isinstance(obj, dict):
            # If the object is a dictionary, convert arrays within its values
            return {k: self.convert_arrays_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # If the object is a list, convert arrays within its elements
            return [self.convert_arrays_to_lists(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            # If the object is a NumPy array, convert it to a list
            return obj.tolist()
        else:
            # If the object is neither a dictionary, list, nor NumPy array, return it unchanged
            return obj
