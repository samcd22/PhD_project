import numpy as np
import pandas as pd
import os
import json
from parameter import Parameter

class Sampler:
    def __init__(self, params, model, likelihood, data, joint_pdf = True, show_sample_info = False):
        self.current_params = params
        self.proposed_params = self.copy_params(params)
        self.model = model
        self.model_func = model.get_model()
        self.likelihood = likelihood
        self.likelihood_func = likelihood.get_log_likelihood_func()
        self.data = data
        self.joint_pdf = joint_pdf
        self.show_sample_info = show_sample_info
        self.sample_info_rows = []

    def get_hyperparams(self):
        self.hyperparams = {}
        # Adding Parameter related hyperparameters
        self.hyperparams['params'] = {}
        for param_ind in self.current_params.index:
            self.hyperparams['params'][param_ind] = {}
            self.hyperparams['params'][param_ind]['init_val'] = self.current_params[param_ind].init_val
            self.hyperparams['params'][param_ind]['step_func'] = self.current_params[param_ind].step_select
            self.hyperparams['params'][param_ind]['step_size'] = self.current_params[param_ind].step_size
            self.hyperparams['params'][param_ind]['prior_func'] = self.current_params[param_ind].prior_select
            self.hyperparams['params'][param_ind]['prior_params'] = {}
            for prior_param_ind in self.current_params[param_ind].prior_params.index:
                self.hyperparams['params'][param_ind]['prior_params'][prior_param_ind] = self.current_params[param_ind].prior_params[prior_param_ind]

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
    
    def accept_params(self, current_log_priors, proposed_log_priors, step_forward_log_prob, step_backward_log_prob):
        # Calculate the log posterior of the current parameters
        curr_modelled_vals = self.model_func(self.current_params,self.data['x'],self.data['y'],self.data['z'])
        curr_log_lhood = self.likelihood_func(curr_modelled_vals, self.data['Concentration'])
        curr_log_posterior = curr_log_lhood + current_log_priors
        
         # Calculate the log posterior of the proposed parameters
        prop_modelled_vals = self.model_func(self.proposed_params,self.data['x'],self.data['y'],self.data['z'])
        prop_log_lhood = self.likelihood_func(prop_modelled_vals, self.data['Concentration'])
        prop_log_posterior = prop_log_lhood + proposed_log_priors

        alpha = np.exp(prop_log_posterior - curr_log_posterior + step_backward_log_prob - step_forward_log_prob)
        rand_num = np.random.uniform(low = 0, high = 1)
        accepted = rand_num < np.min([1,alpha])

        if self.show_sample_info:
            sample_info_row = {}
            sample_info_row['current_params'] = [x.val for x in self.current_params]
            sample_info_row['proposed_params'] = [x.val for x in self.proposed_params]
            sample_info_row['current_log_likelihood'] = curr_log_lhood
            sample_info_row['proposed_log_likelihood'] = prop_log_lhood
            sample_info_row['step_forward_log_probs'] = step_forward_log_prob
            sample_info_row['step_backward_log_probs'] = step_backward_log_prob
            sample_info_row['current_log_posterior'] = curr_log_posterior
            sample_info_row['proposed_log_posterior'] = prop_log_posterior
            sample_info_row['alpha'] = alpha
            sample_info_row['accepted'] = accepted

            self.sample_info_rows.append(pd.Series(sample_info_row))

        # Acceptance criteria

        # Acceptance criteria.
        if accepted:
            self.current_params = self.copy_params(self.proposed_params)
            return self.copy_params(self.proposed_params), 1
        else:
            # print(prop_log_lhood, prop_log_posterior, step_backward_log_prob, step_forward_log_prob)
            # print([x.val for x in self.proposed_params])
            # print([x.val for x in self.current_params])
            # print('\n')

            return self.copy_params(self.current_params), 0
    
    def sample_one(self):
        current_log_priors = []
        proposed_log_priors = []
        step_forward_log_probs = []
        step_backward_log_probs = []

        for i in range(self.current_params.size):
            # Define current parameter
            current_param = self.current_params[i]
            proposed_param = current_param.copy()
            
            # Get functions
            step_log_prob, step_function = current_param.get_step_function()
            log_prior_func = current_param.get_log_prior()
            
            # Step to proposed parameter
            proposed_param.val = step_function(current_param.val)
            step_forward_log_probs.append(step_log_prob(proposed_param.val, current_param.val))
            step_backward_log_probs.append(step_log_prob(current_param.val, proposed_param.val))

            # Add to series of proposed parameters
            self.proposed_params[i] = proposed_param
                        
            # Create a list of log prior probabilities from each current and proposed parameter
            current_log_priors.append(log_prior_func(current_param.val))
            proposed_log_priors.append(log_prior_func(proposed_param.val))

        # print(step_forward_log_probs)
        # print(step_backward_log_probs)

            # if proposed_param.val <0:
            #     print(current_param.val)
            #     print(proposed_param.val)
            #     raise Exception("Number below 0")
            
            # Can include non joint PDF here
            
        if self.joint_pdf:
            return self.accept_params(sum(current_log_priors), sum(proposed_log_priors), sum(step_forward_log_probs), sum(step_backward_log_probs))
            
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
                print(instance_folder)
            return data_exists
            
    def sample_all(self, n_samples):
        data_exists = self.check_data_exists()
        acceptance_rate = 0
        samples = []
        accepted = []
        sample_info_file_name = 'most_recent_sample_info.csv'
        if os.path.exists(sample_info_file_name):
            os.remove(sample_info_file_name)
        if not data_exists:
            for i in range(1,n_samples+1):
                if (i % 1000 == 0):
                    print('Running sample ' + str(i) + '...')    # Print progress every 1000th sample.
                sample, accept = self.sample_one()
                accepted.append(accept)
                samples.append(sample)
            if self.show_sample_info:
                self.sample_info = pd.concat(self.sample_info_rows, axis=1, ignore_index=True).T
                self.sample_info.to_csv(sample_info_file_name)
            acceptance_rate = sum(accepted)/len(accepted)*100
            samples = pd.DataFrame(samples)

        return samples, acceptance_rate
    
    def get_mean_samples(self,samples):
        if type(samples) == list and samples == []:
            means = []
        else: 
            means = pd.Series({},dtype='float64')
            for col in samples.columns:
                mean = samples[col].apply(lambda x: x.val).mean()
                means[col] = Parameter(mean)
        return means
    
    def get_lower_samples(self,samples):
        if type(samples) == list and samples == []:
            lowers = []
        else: 
            lowers = pd.Series({},dtype='float64')
            for col in samples.columns:
                formatted_samples = samples[col].apply(lambda x: x.val)
                lower = np.quantile(formatted_samples, 0.05)
                lowers[col] = Parameter(lower)
        return lowers
        
    def get_upper_samples(self,samples):
        if type(samples) == list and samples == []:
            uppers = []
        else: 
            uppers = pd.Series({},dtype='float64')
            for col in samples.columns:
                formatted_samples = samples[col].apply(lambda x: x.val)
                upper = np.quantile(formatted_samples, 0.95)
                uppers[col] = Parameter(upper)
        return uppers