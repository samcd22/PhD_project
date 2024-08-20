import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import json
from numpyencoder import NumpyEncoder
from matplotlib import pyplot as plt
import shutil

from controllers.controller import Controller
from toolboxes.inference_toolbox.parameter import Parameter
from app.toolboxes.inference_toolbox.model import Model
from toolboxes.inference_toolbox.likelihood import Likelihood
from toolboxes.inference_toolbox.sampler import Sampler
from toolboxes.inference_toolbox.visualiser import Visualiser
from toolboxes.plotting_toolbox.domain import Domain

# Optimiser class - generates multiple instances of the sampler, varying their hyperparameters to optimise the results
class Optimiser(Controller):
    def __init__(self,
                results_name = 'name_placeholder',
                data_params = None,
                default_params = None,
                results_path = 'results/inference_results'):
        
        # Inherits methods and attributes from parent Controller class
        super().__init__(results_name, data_params, default_params, results_path)
        
        # Generates results folder
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Initialises the default parameters
        self.default_values = pd.Series({
            'RMSE': 'NaN',
            'AIC': 'NaN',
            'BIC': 'NaN',
            'av_divergence': 'NaN'
        })
    
        # Initialises the parameter to data frame column dictionary
        self.par_to_col = {
            'AIC': ('results','misc', 'AIC'),
            'BIC': ('results','misc', 'BIC'),
            'RMSE': ('results','misc', 'RMSE'),
            'av_divergence': ('results','misc', 'average_diverging'),
        }

        self.data_params = data_params

        # Adds the sampler info to the default parameters and the parameter to data frame column dictionary
        sampler_info = default_params['sampler']
        self.default_values['n_samples'] = sampler_info['n_samples']
        self.par_to_col['n_samples'] = ('parameters','sampler','n_samples')
        self.default_values['n_chains'] = sampler_info['n_chains']
        self.par_to_col['n_chains'] = ('parameters','sampler','n_chains')
        self.default_values['thinning_rate'] = sampler_info['thinning_rate']
        self.par_to_col['thinning_rate'] = ('parameters','sampler','thinning_rate')

        def set_prior_param_default_values(param_name, params):
            multi_mode = False
            if 'multi_mode' in params[param_name].prior_select:
                multi_mode = True
            
            for prior_param in params[param_name].prior_params.index:
                n_dims = len(np.array(params[param_name].prior_params[prior_param]).shape)
                if multi_mode:
                    n_modes = np.array(params[param_name].prior_params[prior_param]).shape[0]
                    dim1 = 1
                    dim2 = 1
                    if n_dims > 1:
                        dim1 = np.array(params[param_name].prior_params[prior_param]).shape[1]
                    if n_dims > 2:
                        dim2 = np.array(params[param_name].prior_params[prior_param]).shape[2]
                    for n in range(n_modes):
                        if dim1 == 1:
                            full_prior_param_name = param_name + '_' + prior_param + '_mode_' + str(n)
                            self.default_values[full_prior_param_name] = np.array(params[param_name].prior_params[prior_param])[n]
                            self.par_to_col[full_prior_param_name] = ('parameters', param_name, prior_param + '_mode_' + str(n))                    
                        elif dim1 > 1:
                            if dim2 == 1:
                                for i in range(dim1):
                                    full_prior_param_name = param_name + '_' + prior_param + '_' + str(i) + '_mode_' + str(n)
                                    self.default_values[full_prior_param_name] = np.array(params[param_name].prior_params[prior_param])[n,i]
                                    self.par_to_col[full_prior_param_name] = ('parameters', param_name, prior_param + '_' + str(i) + '_mode_' + str(n))

                            elif dim2 > 1:
                                for i in range(dim1):
                                    for j in range(dim2):
                                        full_prior_param_name = param_name + '_' + prior_param + '_' + str(i) + '_' + str(j) + '_mode_' + str(n)
                                        self.default_values[full_prior_param_name] = np.array(params[param_name].prior_params[prior_param])[n,i,j]
                                        self.par_to_col[full_prior_param_name] = ('parameters', param_name, prior_param + '_' + str(i) + '_' + str(j) + '_mode_' + str(n))
                            else:
                                raise Exception('Invalid shape of prior parameter')
                        else:
                            raise Exception('Invalid shape of prior parameter')


                else:
                    dim1 = 1
                    dim2 = 1 
                    if n_dims > 0:
                        dim1 = np.array(params[param_name].prior_params[prior_param]).shape[0]
                    if n_dims > 1:
                        dim2 = np.array(params[param_name].prior_params[prior_param]).shape[1]

                    if dim1 == 1:
                        full_prior_param_name = param_name + '_' + prior_param
                        self.default_values[full_prior_param_name] = params[param_name].prior_params[prior_param]
                        self.par_to_col[full_prior_param_name] = ('parameters', param_name, prior_param)                    
                    elif dim1 > 1:
                        if dim2 == 1:
                            for i in range(dim1):
                                full_prior_param_name = param_name + '_' + prior_param + '_' + str(i)
                                self.default_values[full_prior_param_name] = params[param_name].prior_params[prior_param][i]
                                self.par_to_col[full_prior_param_name] = ('parameters', param_name, prior_param + '_' + str(i))

                        elif dim2 > 1:
                            for i in range(dim1):
                                for j in range(dim2):
                                    full_prior_param_name = param_name + '_' + prior_param + '_' + str(i) + '_' + str(j)
                                    self.default_values[full_prior_param_name] = params[param_name].prior_params[prior_param][i][j]
                                    self.par_to_col[full_prior_param_name] = ('parameters', param_name, prior_param + '_' + str(i) + '_' + str(j))
                        else:
                            raise Exception('Invalid shape of prior parameter')
                        
                    else:
                        raise Exception('Invalid shape of prior parameter')

        # Adds all parameter information to the default parameters object and the parameter to data frame dictionary
        def set_param_default_values(params):
            new_param_list = []
            
            for param in params.index:
                self.default_values[param+'_prior'] = params[param].prior_select
                self.par_to_col[param+'_prior'] = ('parameters', param, 'prior')

                self.default_values[param+'_order'] = params[param].order
                self.par_to_col[param+'_order'] = ('parameters', param, 'order')

                set_prior_param_default_values(param, params)

                if '_and_' in param:
                    new_param_list.append(param.split('_and_')[0])
                    new_param_list.append(param.split('_and_')[1])
                else:
                    new_param_list.append(param)

            for param in new_param_list:

                self.default_values[param+'_mean'] = 'NaN'
                self.par_to_col[param+'_mean'] = ('results', param, 'mean')
                
                self.default_values[param+'_lower'] = 'NaN'
                self.par_to_col[param+'_lower'] = ('results', param, 'lower')

                self.default_values[param+'_upper'] = 'NaN'
                self.par_to_col[param+'_upper'] = ('results', param, 'upper')

                self.default_values[param+'_tau'] = 'NaN'
                self.par_to_col[param+'_tau'] = ('results', param, 'average_tau')

                self.default_values[param+'_param_accuracy'] = 'NaN'
                self.par_to_col[param+'_param_accuracy'] = ('results', param, 'param_accuracy')
        
        # Uses previously defined function for the model and likelihood infered parameters
        infered_model_params = default_params['infered_params']['model_params']
        infered_likelihood_params = default_params['infered_params']['likelihood_params']
        set_param_default_values(infered_model_params)
        set_param_default_values(infered_likelihood_params)

        # Adds all model information to the default parameters object and the parameter to data frame dictionary
        model = default_params['model']
        self.default_values['model_type'] = model.model_select
        self.par_to_col['model_type'] = ('parameters', 'model', 'type')
        for model_param in model.model_params.index:
            self.default_values['model_' + model_param] = model.model_params[model_param]
            self.par_to_col['model_' + model_param] = ('parameters', 'model', model_param)

        # Adds all likelihood information to the default parameters object and the parameter to data frame dictionary
        likelihood = default_params['likelihood']
        self.default_values['likelihood_type'] = likelihood.likelihood_select
        self.par_to_col['likelihood_type'] = ('parameters', 'likelihood', 'type')
        for likelihood_param in likelihood.likelihood_params.index:
            self.default_values['likelihood_' + likelihood_param] = likelihood.likelihood_params[likelihood_param]
            self.par_to_col['likelihood_' + likelihood_param] = ('parameters', 'likelihood', likelihood_param)

        # Actual parameter values are saved if they are available
        self.actual_values = []
        if self.data_params['data_type'] == 'dummy':
            for inference_param in self.data_params['model']['inference_params'].keys():
                self.actual_values.append(self.data_params['model']['inference_params'][inference_param])

        # Generates the data construction object
        data_construction = self.get_data_construction()

        # Generates the default_params construction object
        default_params_construction = self.get_default_params_construction()

        # Initialises the construction
        self.init_data_construction(data_construction)
        self.init_default_params_construction(default_params_construction)

        # Sort out data
        data = get_data(self.data_params)
        self.training_data, self.testing_data = train_test_split(data, test_size=0.2, random_state = 1)

    # Initialises the construction using the construction object, checking and creating all relevant files and folders
    def init_data_construction(self, construction):
        self.data_construction_results_path = self.results_path + '/' + self.results_name

        if not os.path.exists(self.data_construction_results_path):
            os.makedirs(self.data_construction_results_path)

        if os.path.exists(self.data_construction_results_path + '/data_construction.json'):
            f = open(self.data_construction_results_path + '/data_construction.json')
            saved_construction = json.load(f)
            f.close()

            if saved_construction != construction:
                raise Exception('Default generator parameters do not match for this folder name!')
        else:
            with open(self.full_results_path + '/data_construction.json', "w") as fp:
                json.dump(construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

    def init_default_params_construction(self, construction):
        self.generators_path = self.data_construction_results_path + '/auto_gen_instances'

        if not os.path.exists(self.generators_path):
            os.makedirs(self.generators_path)
        
        generator_name = self.get_generator_name(self.generators_path, construction)

        self.unique_generator_path = self.generators_path + '/' + generator_name
        self.full_results_path = self.unique_generator_path + '/' + 'optimisers'

        if not os.path.exists(self.full_results_path):
            os.makedirs(self.full_results_path)

        if os.path.exists(self.unique_generator_path + '/default_params_construction.json'):
            f = open(self.unique_generator_path + '/default_params_construction.json')
            saved_construction = json.load(f)
            f.close()

            if saved_construction != construction:
                raise Exception('Default generator parameters do not match for this folder name!')
        else:
            with open(self.unique_generator_path + '/default_params_construction.json', "w") as fp:
                json.dump(construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

    def get_generator_name(self, generators_path, default_params_construction):
        data_exists = False
        if not os.path.exists(generators_path):
            os.makedirs(generators_path)
        generator_folders = os.listdir(generators_path)
        for generator_folder in generator_folders:
            folder_path = self.generators_path + '/' + generator_folder
            f = open(folder_path + '/default_params_construction.json')
            generator_construction = json.load(f)
            f.close()
            if default_params_construction == generator_construction:
                data_exists = True
                output = generator_folder
        
        if not data_exists:
            generator_nums = [int(x.split('_')[1]) for x in generator_folders]
            missing_elements = []
            if len(generator_nums) == 0:
                output = 'generator_1'
            else:
                for el in range(1,np.max(generator_nums) + 2):
                    if el not in generator_nums:
                        missing_elements.append(el)
                generator_num = np.min(missing_elements)
                output = 'generator_' + str(generator_num)

        return output

    # Generates a optimiser conctruction object which includes all info on how the optimiser was costructed
    def get_optimiser_construction(self):
        return {
            'parameter': self.optimising_parameters, 
            'index_name': self.index_name
            }
    
    # Initialises the optimiser construction using the optimiser construction object, checking and creating all relevant files and folders
    def init_optimiser(self, optimiser_construction):
        self.optimiser_path = self.full_results_path + '/' + self.optimiser_name
        if not os.path.exists(self.optimiser_path):
            os.makedirs(self.optimiser_path)

        self.instances_path = self.optimiser_path + '/instances' 
        if os.path.exists(self.optimiser_path + '/varying_parameters.json'):
            f = open(self.optimiser_path + '/varying_parameters.json')
            saved_construction = json.load(f)
            f.close()
            
            if saved_construction != optimiser_construction:
                raise Exception('These optimiser parameters do not match for this folder name!')
        else:
            with open(self.optimiser_path + '/varying_parameters.json', "w") as fp:
                json.dump(optimiser_construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)
    
    # Runs multiple instances of the sampler optimising the inputted hyperparameters based on the inputted success metric
    def run(self, n_trials = 100, optimising_parameters = None,
                index_name = 'rmse',
                optimiser_name = 'optimiser_1',
                direction = 'minimize'):
        
        # Assigns variables
        self.optimising_parameters = optimising_parameters
        self.optimiser_name = optimiser_name
        self.index_name = index_name
        self.n_trials = n_trials
        self.direction = direction

        # Generates the optimiser construction object
        construction = self.get_optimiser_construction()

        # Initialises the construction
        self.init_optimiser(construction)

        # Initialises the inputs object using the default values as references for the column names
        self.all_inputs = pd.DataFrame(columns=[self.par_to_col[x] for x in self.default_values.index])

        if os.path.exists(self.optimiser_path + '/results.csv'):
            previous_inputs = pd.read_csv(self.optimiser_path + '/results.csv', header=[0,1,2], index_col=0)
            self.all_inputs = previous_inputs

        study_name = self.optimiser_name  # Unique identifier of the study.
        storage_name = self.full_results_path + '/' + study_name +'.db'

        # Initialises the optimiser instance
        study = optuna.create_study(study_name=study_name, storage='sqlite:///' + storage_name, direction = direction, load_if_exists=True)

        num_previous_trials = len(study.trials)

        if n_trials >= num_previous_trials:
            if n_trials > num_previous_trials:
                if os.path.exists(self.optimiser_path + '/best_instance'):
                    shutil.rmtree(self.optimiser_path + '/best_instance')
            n_trials -= num_previous_trials
        else:
            raise Exception('The previously run optimiser has run more trials!')

        # Defines the success metric for optimisation
        if self.index_name == 'rmse':
            optimizing_function = self.calculate_rmse
        elif self.index_name == 'aic':
            optimizing_function = self.calculate_aic
        elif self.index_name == 'bic':
            optimizing_function = self.calculate_bic

        # Runs the optimiser using the optimisation function
        study.optimize(optimizing_function, n_trials=n_trials, timeout=6000)

        return study

    # Generates one instance of the sampler based on the trials hyperparameters
    def run_optimisation_inference(self, trial):
        
        # Assigns variables for inference from the optimisation trial
        params, model, likelihood = self.prepare_inference(trial = trial)
        
        # Sampler parameters
        num_samples = self.default_params['sampler']['n_samples']
        num_chains = self.default_params['sampler']['n_chains']
        thinning_rate = self.default_params['sampler']['thinning_rate']

        # Initialises and runs the sampler
        sampler = Sampler(params, model, likelihood, self.training_data, self.testing_data, num_samples, n_chains=num_chains, thinning_rate=thinning_rate, data_path = self.instances_path)
        sampler.sample_all()

        # Initialises the visualiser
        visualiser = Visualiser(self.testing_data, 
                        sampler, 
                        model, 
                        previous_instance = sampler.instance, 
                        data_path = self.instances_path, 
                        suppress_prints = True,
                        actual_values = self.actual_values)
        
        # Generates the summary object of the results of the instance
        summary = visualiser.get_summary()

        param_list_1 = self.default_params['infered_params']['model_params'].index
        new_param_list_1 = []
        for param in param_list_1:
            if '_and_' in param:
                new_param_list_1.append(param.split('_and_')[0])
                new_param_list_1.append(param.split('_and_')[1])
            else:
                new_param_list_1.append(param)

        # Adds the infered model parameter results to the results data frame
        for param in new_param_list_1:
            self.one_row[self.par_to_col[param + '_lower']] = summary['overall'][param]['lower']
            self.one_row[self.par_to_col[param + '_mean']] = summary['overall'][param]['mean']
            self.one_row[self.par_to_col[param + '_upper']] = summary['overall'][param]['upper']
            self.one_row[self.par_to_col[param + '_tau']] = summary['overall'][param]['tau']
            if 'param_accuracy' in summary['overall'][param]:
                self.one_row[self.par_to_col[param + '_param_accuracy']] = summary['overall'][param]['param_accuracy']

        param_list_2 = self.default_params['infered_params']['likelihood_params'].index
        new_param_list_2 = []
        for param in param_list_2:
            if '_and_' in param:
                new_param_list_2.append(param.split('_and_')[0])
                new_param_list_2.append(param.split('_and_')[1])
            else:
                new_param_list_2.append(param)

        # Adds the infered likelihood parameter results to the results data frame
        for param in new_param_list_2:
            self.one_row[self.par_to_col[param + '_lower']] = summary['overall'][param]['lower']
            self.one_row[self.par_to_col[param + '_mean']] = summary['overall'][param]['mean']
            self.one_row[self.par_to_col[param + '_upper']] = summary['overall'][param]['upper']
            self.one_row[self.par_to_col[param + '_tau']] = summary['overall'][param]['tau']
            if 'param_accuracy' in summary['overall'][param]:
                self.one_row[self.par_to_col[param + '_param_accuracy']] = summary['overall'][param]['param_accuracy']

        # Adds the success metrics to the results dara frame
        self.one_row[self.par_to_col['RMSE']] = summary['RMSE']
        self.one_row[self.par_to_col['AIC']] = summary['AIC']
        self.one_row[self.par_to_col['BIC']] = summary['BIC']

        # Adds the number of divergences to the results data frame 
        divergences = []
        for i in range(visualiser.num_chains):
            divergences.append(summary['chain_' + str(i+1)]['fields']['diverging'])
        self.one_row[self.par_to_col['av_divergence']] = np.mean(divergences)

        # Formats and saves the updated results data frame
        self.all_inputs = pd.concat([self.all_inputs, self.one_row.to_frame().T], ignore_index=True)
        self.all_inputs.columns = pd.MultiIndex.from_tuples(self.all_inputs.columns)

        self.all_inputs.to_csv(self.optimiser_path + '/results.csv')

        return sampler, visualiser

    # Extracts information from the optimisation trial
    def prepare_inference(self, trial = '', results = {}):

        self.one_row = pd.Series(index = [self.par_to_col[x] for x in self.default_values.index], data = self.default_values.values)
        
        params = pd.Series({},dtype='float64')

        def assign_prior_params(params, param_name, trial, prior_params):
            multi_mode = False
            if 'multi_mode' in params[param_name].prior_select:
                multi_mode = True
            
            for prior_param_name in prior_params.index:
                n_dims = len(np.array(prior_params[prior_param_name]).shape)
                if multi_mode:
                    n_modes = np.array(prior_params[prior_param_name]).shape[0]
                    dim1 = 1
                    dim2 = 1
                    if n_dims > 1:
                        dim1 = np.array(prior_params[prior_param_name]).shape[1]
                    if n_dims > 2:
                        dim2 = np.array(prior_params[prior_param_name]).shape[2]
                    for n in range(n_modes):
                        prior_param_val = []
                        # Case 1
                        if dim1 == 1:
                            full_prior_param_name = param_name + '_' + prior_param_name + '_mode_' + str(n)
                            if full_prior_param_name in self.optimising_parameters:
                                if results:
                                    prior_param_mode_val = results[full_prior_param_name]
                                else:
                                    prior_param_mode_val = trial.suggest_float(full_prior_param_name, *self.optimising_parameters[full_prior_param_name])
                                    self.one_row[self.par_to_col[full_prior_param_name]] = prior_param_mode_val
                            else:
                                prior_param_mode_val = prior_params[prior_param_name][n]
                    
                        # Case 2
                        elif dim1 > 1:
                            if dim2 == 1:
                                prior_param_mode_val = np.zeros(dim1)
                                for i in range(dim1):
                                    full_prior_param_name = param_name + '_' + prior_param_name + '_' + str(i) + '_mode_' + str(n)
                                    if full_prior_param_name in self.optimising_parameters:
                                        if results:
                                            prior_param_mode_val[i] = results[full_prior_param_name]
                                        else:
                                            prior_param_mode_val[i] = trial.suggest_float(full_prior_param_name, *self.optimising_parameters[full_prior_param_name])
                                            self.one_row[self.par_to_col[full_prior_param_name]] = prior_param_mode_val[i]
                                    else:
                                        prior_param_mode_val[i] = np.array(prior_params[prior_param_name])[n,i]
                                
                                prior_param_mode_val = np.array(prior_param_mode_val).tolist()

                        # Case 3
                            elif dim2 > 1:
                                prior_param_mode_val = np.zeros((dim1,dim2))
                                for i in range(dim1):
                                    for j in range(dim2):
                                        full_prior_param_name = param_name + '_' + prior_param_name + '_' + str(i) + '_' + str(j) + '_mode_' + str(n)
                                        if full_prior_param_name in self.optimising_parameters:
                                            if results:
                                                prior_param_mode_val[i,j] = results[full_prior_param_name]
                                            else:
                                                prior_param_mode_val[i,j] = trial.suggest_float(full_prior_param_name, *self.optimising_parameters[full_prior_param_name])
                                                self.one_row[self.par_to_col[full_prior_param_name]] = prior_param_mode_val[i,j]
                                        else:
                                            prior_param_mode_val[i,j] = np.array(prior_params[prior_param_name])[n,i,j]
                                
                                prior_param_mode_val = np.array(prior_param_mode_val).tolist()
                            
                            else:
                                raise Exception('Invalid shape of prior parameter')
                        else:
                            raise Exception('Invalid shape of prior parameter')

                        prior_param_val.append(prior_param_mode_val)

                else:
                    dim1 = 1
                    dim2 = 1 
                    if n_dims > 0:
                        dim1 = np.array(prior_params[prior_param_name]).shape[0]
                    if n_dims > 1:
                        dim2 = np.array(prior_params[prior_param_name]).shape[1]

                    # Case 4
                    if dim1 == 1:
                        full_prior_param_name = param_name + '_' + prior_param_name
                        if full_prior_param_name in self.optimising_parameters:
                            if results:
                                prior_param_val = results[full_prior_param_name]
                            else:
                                prior_param_val = trial.suggest_float(full_prior_param_name, *self.optimising_parameters[full_prior_param_name])
                                self.one_row[self.par_to_col[full_prior_param_name]] = prior_param_val
                        else:
                            prior_param_val = prior_params[prior_param_name]

                    elif dim1 > 1:
                    # Case 5
                        if dim2 == 1:
                            prior_param_val = np.zeros(dim1)
                            for i in range(dim1):
                                full_prior_param_name = param_name + '_' + prior_param_name + '_' + str(i)
                                if full_prior_param_name in self.optimising_parameters:
                                    if results:
                                        prior_param_val[i] = results[full_prior_param_name]
                                    else:
                                        prior_param_val[i] = trial.suggest_float(full_prior_param_name, *self.optimising_parameters[full_prior_param_name])
                                        self.one_row[self.par_to_col[full_prior_param_name]] = prior_param_val[i]
                                else:
                                    prior_param_val[i] = prior_params[prior_param_name][i]
                                
                                prior_param_val = np.array(prior_param_val).tolist()

                    # Case 6
                        elif dim2 > 1:
                            prior_param_val = np.zeros((dim1,dim2))
                            for i in range(dim1):
                                for j in range(dim2):
                                    full_prior_param_name = param_name + '_' + prior_param_name + '_' + str(i) + '_' + str(j)
                                    if full_prior_param_name in self.optimising_parameters:
                                        if results:
                                            prior_param_val[i,j] = results[full_prior_param_name]
                                        else:
                                            prior_param_val[i,j] = trial.suggest_float(full_prior_param_name, *self.optimising_parameters[full_prior_param_name])
                                            self.one_row[self.par_to_col[full_prior_param_name]] = prior_param_val[i,j]
                                    else:
                                        prior_param_val[i,j] = np.array(prior_params[prior_param_name])[i,j]
                            
                            prior_param_val = np.array(prior_param_val).tolist()
                        else:
                            raise Exception('Invalid shape of prior parameter')
                        
                    else:
                        raise Exception('Invalid shape of prior parameter')
                    
                params[param_name].add_prior_param(prior_param_name, prior_param_val)

            return params, trial

        # Model infered parameters
        for param_name in self.default_params['infered_params']['model_params'].keys():

            if param_name + '_prior' in self.optimising_parameters:
                if results:
                    prior_select = results[param_name + '_prior']
                else:
                    prior_select = trial.suggest_categorical(param_name + '_prior', [*self.optimising_parameters[param_name + '_prior']])
                    self.one_row[self.par_to_col[param_name + '_prior']] = prior_select
            else:
                prior_select = self.default_params['infered_params']['model_params'][param_name].prior_select

            if param_name + '_order' in self.optimising_parameters:
                if results:
                    order = results[param_name + '_order']
                else:
                    order = trial.suggest_float(param_name + '_order', [*self.optimising_parameters[param_name + '_prior']])
                    self.one_row[self.par_to_col[param_name + '_order']] = order
            else:
                order = self.default_params['infered_params']['model_params'][param_name].order


            if '_and_' in param_name:
                params[param_name] = Parameter(name = param_name.split('_and_')[0], name_2 = param_name.split('_and_')[1], prior_select=prior_select, order = order)
            else:
                params[param_name] = Parameter(name = param_name, prior_select=prior_select, order = order)

            params, trial = assign_prior_params(params, param_name, trial, self.default_params['infered_params']['model_params'][param_name].prior_params)
    


        # Likelihood infered parameters
        for param_name in self.default_params['infered_params']['likelihood_params'].keys():
            
            if param_name + '_prior' in self.optimising_parameters:
                if results:
                    prior_select = results[param_name + '_prior']
                else:
                    prior_select = trial.suggest_categorical(param_name + '_prior', [*self.optimising_parameters[param_name + '_prior']])
                    self.one_row[self.par_to_col[param_name + '_prior']] = prior_select
            else:
                prior_select = self.default_params['infered_params']['likelihood_params'][param_name].prior_select

            if param_name + '_order' in self.optimising_parameters:
                if results:
                    order = results[param_name + '_order']
                else:
                    order = trial.suggest_float(param_name + '_order', [*self.optimising_parameters[param_name + '_prior']])
                    self.one_row[self.par_to_col[param_name + '_order']] = order
            else:
                order = self.default_params['infered_params']['likelihood_params'][param_name].order

            if '_and_' in param_name:
                params[param_name] = Parameter(name = param_name.split('_and_')[0], name_2 = param_name.split('_and_')[1], prior_select=prior_select, order = order)
            else:
                params[param_name] = Parameter(name = param_name, prior_select=prior_select, order = order)
                
            params, trial = assign_prior_params(params, param_name, trial, self.default_params['infered_params']['likelihood_params'][param_name].prior_params)


        # Likelihood
        if "likelihood_type" in self.optimising_parameters:
            if results:
                likelihood_type = results["likelihood_type"]
            else:
                likelihood_type = trial.suggest_catagorical('likelihood_type', [*self.optimising_parameters['likelihood_type']])
                self.one_row[self.par_to_col['likelihood_type']] = likelihood_type
        else:
            likelihood_type = self.default_params['likelihood'].likelihood_select

        likelihood = Likelihood(likelihood_type)
        
        for likelihood_param_name in self.default_params['likelihood'].likelihood_params.index:
            if 'likelihood_' + likelihood_param_name in self.optimising_parameters:
                if results:
                    likelihood_param = results['likelihood_' + likelihood_param_name]
                else:
                    likelihood_param = trial.suggest_float('likelihood_' + likelihood_param_name, *self.optimising_parameters['likelihood_' + likelihood_param_name])
                    self.one_row[self.par_to_col['likelihood_' + likelihood_param_name]] = likelihood_param
            else:
                likelihood_param = self.default_params['likelihood'].likelihood_params[likelihood_param_name]
            
            likelihood.add_likelihood_param(likelihood_param_name, likelihood_param)

        # Model
        if "model_type" in self.optimising_parameters:
            if results:
                model_type = results["model_type"]
            else:
                model_type = trial.suggest_catagorical('model_type', [*self.optimising_parameters['model_type']])
                self.one_row[self.par_to_col['model_type']] = model_type
        else:
            model_type = self.default_params['model'].model_select

        model = Model(model_type)
        
        for model_param_name in self.default_params['model'].model_params.index:
            if 'model_' + model_param_name in self.optimising_parameters:
                if results:
                    model_param = results['model_' + model_param_name]
                else:
                    model_param = trial.suggest_float('model_' + model_param_name, *self.optimising_parameters['model_' + model_param_name])
                    self.one_row[self.par_to_col['model_' + model_param_name]] = model_param
            else:
                model_param = self.default_params['model'].model_params[model_param_name]
            
            model.add_model_param(model_param_name, model_param)

        if self.data_params['data_type'] == 'simulated_data' and self.data_params['noise_level'] == 'NaN':
                raise Exception('Set the noise level!')

        return params, model, likelihood
    
    # RMSE optimisation function
    def calculate_rmse(self, trial):
        _, visualiser = self.run_optimisation_inference(trial)
        return visualiser.RMSE
    
    # AIC optimisation function
    def calculate_aic(self, trial):
        _, visualiser = self.run_optimisation_inference(trial)
        return visualiser.AIC
    
    # BIC optimisation function
    def calculate_bic(self, trial):
        _, visualiser = self.run_optimisation_inference(trial)
        return visualiser.BIC

    # Generates and saves plots of the optimiser's results
    def get_plots(self, study):
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(self.optimiser_path + '/optimisation_history.png', bbox_inches="tight")
        plt.close()

        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(self.optimiser_path + '/parameter_importances.png', bbox_inches="tight")
        plt.close()

        self.plot_parameter_history()

    def plot_parameter_history(self):
        if not os.path.exists(self.optimiser_path + '/parameter_histories'):
            os.makedirs(self.optimiser_path + '/parameter_histories')

        all_results = pd.read_csv(self.optimiser_path + '/results.csv', header=[0,1,2], index_col=0)
        optimising_index_vals = all_results[self.par_to_col[self.index_name.upper()]]

        plt.figure()
        for optimising_param in self.optimising_parameters.keys():
            best_current_index = optimising_index_vals[0]
            best_current_params= np.zeros(optimising_index_vals.size)
            param_data = all_results[self.par_to_col[optimising_param]]

            for i, index_val in enumerate(optimising_index_vals):
                if self.direction == 'minimize':
                    if index_val <= best_current_index:
                        best_current_params[i] = param_data[i]
                        best_current_index = index_val
                    else:
                        best_current_params[i] = best_current_params[i-1]
                elif self.direction == 'maximize':
                    if index_val >= best_current_index:
                        best_current_params[i] = param_data[i]
                        best_current_index = index_val
                    else:
                        best_current_params[i] = best_current_params[i-1]                        
                else:
                    raise Exception('Invalid optimising direction!')

            plt.plot(param_data, '.-b', label = 'Parameter Values', alpha = 0.7)
            plt.plot(best_current_params, '--r', label = 'Current Best Value')

            plt.title(optimising_param)
            plt.ylabel('Value')
            plt.xlabel('Trial Number')
            plt.legend()
            plt.savefig(self.optimiser_path + '/parameter_histories/' + optimising_param + '_histories.png')
            plt.close()

    # Generates an instance of the sampler based on the optimal hyperparameters concluded from the optimiser
    def run_best_params(self, study, domain, name, prior_plots = None):
        # Extracts information from the best trial of the optimiser
        self.results = study.best_params
        self.save_results()
        params, model, likelihood = self.prepare_inference(results=self.results)
        
        # Sampler parameters
        num_samples = self.default_params['sampler']['n_samples']
        num_chains = self.default_params['sampler']['n_chains']
        thinning_rate = self.default_params['sampler']['thinning_rate']

        # Generates folders
        best_instance_path = self.optimiser_path + '/best_instance'
        if not os.path.exists(best_instance_path):
            os.makedirs(best_instance_path)

        # Initialises the sampler instance and runs it for the allotted number of samples
        sampler = Sampler(params, model, likelihood, self.training_data, self.testing_data, num_samples, n_chains=num_chains, thinning_rate=thinning_rate, data_path = best_instance_path)
        sampler.sample_all()

        # Initialises the visualiser
        visualiser = Visualiser(self.testing_data, 
                        sampler, 
                        model, 
                        previous_instance = sampler.instance, 
                        data_path = best_instance_path, 
                        suppress_prints = True,
                        actual_values=self.actual_values)
        
        # Generates the summary object, traceplots and autocorrelation plots
        visualiser.get_summary()
        visualiser.get_traceplot()
        visualiser.get_autocorrelations()

        if prior_plots:
            visualiser.get_prior_plots(prior_plots)

        # Generates the plots and animation of the modeled system using the sampled parameters
        visualiser.visualise_results(domain = domain, plot_type = '2D_slice', name = name, title='Log Concentration of Droplets', log_results=False)
        visualiser.visualise_results(domain = domain, plot_type = '3D', name = name, title='Log Concentration of Droplets', log_results=False)


    def save_results(self):
        with open(self.optimiser_path + '/results.json', "w") as fp:
            json.dump(self.results,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)
