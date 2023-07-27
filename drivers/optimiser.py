import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import json
from numpyencoder import NumpyEncoder
from matplotlib import pyplot as plt

from drivers.driver import Driver
from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from inference_toolbox.sampler import Sampler
from inference_toolbox.visualiser import Visualiser
from inference_toolbox.domain import Domain
from data_processing.get_data import get_data

# Optimiser class - generates multiple instances of the sampler, varying their hyperparameters to optimise the results
class Optimiser(Driver):
    def __init__(self,
                 results_name = 'name_placeholder',
                 default_params = {
                    'infered_params':pd.Series({
                        'model_params':pd.Series({
                            'I_y': Parameter('I_y', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'I_z': Parameter('I_z', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'Q': Parameter('Q', prior_select = 'gamma', default_value=3e13).add_prior_param('mu', 3e13).add_prior_param('sigma',1e13),
                        }),
                        'likelihood_params':pd.Series({},dtype='float64')
                    }),
                    'model':Model('log_gpm_alt_norm').add_model_param('H',10),
                    'likelihood': Likelihood('gaussian_fixed_sigma').add_likelihood_param('sigma',1),
                    'sampler': {
                        'n_samples': 10000,
                        'n_chains': 1,
                        'thinning_rate': 1
                    }
                },
                data_params = {
                    'data_type': 'dummy',
                    'data_path': 'data',
                    'sigma': 'NaN',
                    'model_select': 'log_gpm_alt_norm',
                    'noise_dist': 'gaussian',
                    'model': {
                        'model_params':{
                            'H': 10
                        },
                        'inference_params':{
                            'I_y': 0.1,
                            'I_z': 0.1,
                            'Q': 3e13
                        },
                    },
                    'domain': {
                        'domain_select': 'cone_from_source_z_limited', 
                        'resolution': 20,
                        'domain_params':{
                            'r': 100,
                            'theta': np.pi/8,
                            'source': [0,0,10]}
                    },
                    'output_header': 'Concentration'
                },
                results_path = 'results'):
        
        # Inherits methods and attributes from parent Driver class
        super().__init__(results_name, default_params, data_params, results_path)
        
        # Initialises the default parameters
        self.default_values = pd.Series({
            'RMSE': 'NaN',
            'AIC': 'NaN',
            'av_divergence': 'NaN'
        })
    
        # Initialises the parameter to data frame column dictionary
        self.par_to_col = {
            'AIC': ('results','misc', 'AIC'),
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

        # Adds all parameter information to the default parameters object and the parameter to data frame dictionary
        def set_param_default_values(params):
            for param in params.index:
                self.default_values[param+'_prior'] = params[param].prior_select
                self.par_to_col[param+'_prior'] = ('parameters', param, 'prior')
                
                for prior_param in params[param].prior_params.index:
                    self.default_values[param+'_'+prior_param] = params[param].prior_params[prior_param]
                    self.par_to_col[param+'_'+prior_param] = ('parameters', param, prior_param)

                self.default_values[param+'_mean'] = params[param].default_value
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

        # Generates the construction object
        construction = self.get_constriction()

        # Initialises the construction
        self.init_construction(construction)

        # Sort out data
        data = get_data(self.data_params['data_type'], self.data_params)
        self.training_data, self.testing_data = train_test_split(data, test_size=0.2, random_state = 1)

    # Initialises the construction using the construction object, checking and creating all relevant files and folders
    def init_construction(self, construction):         
        self.construction_results_path = self.results_path + '/' + self.results_name
        self.full_results_path = self.construction_results_path + '/optimisers/'

        if not os.path.exists(self.construction_results_path):
            os.makedirs(self.construction_results_path)
        
        if os.path.exists(self.construction_results_path + '/construction.json'):
            f = open(self.construction_results_path + '/construction.json')
            saved_construction = json.load(f)
            f.close()

            if saved_construction != construction:
                raise Exception('Default generator parameters do not match for this folder name!')
        else:
            with open(self.construction_results_path + '/construction.json', "w") as fp:
                json.dump(construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

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
    def run(self, n_trials = 100, optimising_parameters = {
                    'I_y_mu': [1e-2, 10],
                    'I_y_sigma': [1e-2, 10],
                    'I_z_mu': [1e-2, 10],
                    'I_z_sigma': [1e-2, 10],
                    'Q_mu': [1e9, 1e18],
                    'Q_sigma': [1e9, 1e18],
                },
                index_name = 'rmse',
                optimiser_name = 'optimiser_1'):
        
        # Assigns variables
        self.optimising_parameters = optimising_parameters
        self.optimiser_name = optimiser_name
        self.index_name = index_name

        # Generates the optimiser construction object
        construction = self.get_optimiser_construction()

        # Initialises the construction
        self.init_optimiser(construction)

        # Initialises the inputs object using the default values as references for the column names
        self.all_inputs = pd.DataFrame(columns=[self.par_to_col[x] for x in self.default_values.index])

        # Initialises the optimiser instance
        study = optuna.create_study(direction="minimize")

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

        # Adds the infered model parameter results to the results data frame
        for param in self.default_params['infered_params']['model_params'].index:
            self.one_row[self.par_to_col[param + '_lower']] = summary['overall'][param]['lower']
            self.one_row[self.par_to_col[param + '_mean']] = summary['overall'][param]['mean']
            self.one_row[self.par_to_col[param + '_upper']] = summary['overall'][param]['upper']
            self.one_row[self.par_to_col[param + '_tau']] = summary['overall'][param]['tau']
            if 'param_accuracy' in summary['overall'][param]:
                self.one_row[self.par_to_col[param + '_param_accuracy']] = summary['overall'][param]['param_accuracy']

        # Adds the infered likelihood parameter results to the results data frame
        for param in self.default_params['infered_params']['likelihood_params'].index:
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
        self.all_inputs.to_excel(self.optimiser_path + '/results.xlsx')

        return sampler, visualiser

    # Extracts information from the optimisation trial
    def prepare_inference(self, trial = '', results = {}):
        self.one_row = pd.Series(index = [self.par_to_col[x] for x in self.default_values.index], data = self.default_values.values)
        
        params = pd.Series({},dtype='float64')

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

            params[param_name] = Parameter(name = param_name, prior_select=prior_select)
            
            for prior_param_name in self.default_params['infered_params']['model_params'][param_name].prior_params.index:
                
                if param_name + '_' + prior_param_name in self.optimising_parameters:
                    if results:
                        prior_param = results[param_name + '_' + prior_param_name]
                    else:
                        prior_param = trial.suggest_float(param_name + '_' + prior_param_name, *self.optimising_parameters[param_name + '_' + prior_param_name])
                        self.one_row[self.par_to_col[param_name + '_' + prior_param_name]] = prior_param
                else:
                    prior_param = self.default_params['infered_params']['model_params'][param_name].prior_params[prior_param_name]

                params[param_name].add_prior_param(prior_param_name, prior_param)

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

            params[param_name] = Parameter(name = param_name, prior_select=prior_select)
            for prior_param_name in self.default_params['infered_params']['likelihood_params'][param_name].prior_params.index:
                
                if param_name + '_' + prior_param_name in self.optimising_parameters:
                    if results:
                        prior_param = results[param_name + '_' + prior_param_name]
                    else:
                        prior_param = trial.suggest_float(param_name + '_' + prior_param_name, *self.optimising_parameters[param_name + '_' + prior_param_name])
                        self.one_row[self.par_to_col[param_name + '_' + prior_param_name]] = prior_param
                else:
                    prior_param = self.default_params['infered_params']['likelihood_params'][param_name].prior_params[prior_param_name]

                params[param_name].add_prior_param(prior_param_name, prior_param)

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

        if self.data_params['data_type'] == 'dummy':
            if self.data_params['sigma'] == 'NaN':
                if 'sigma' not in likelihood.likelihood_params:
                    raise Exception('Either define your noise level with a fixed sigma in the likelihood, or set the noise level!')
                self.data_params['sigma'] = likelihood.likelihood_params['sigma']

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
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(self.optimiser_path + '/parameter_importances.png', bbox_inches="tight")

    # Generates an instance of the sampler based on the optimal hyperparameters concluded from the optimiser
    def run_best_params(self, study):

        # Extracts information from the best trial of the optimiser
        results = study.best_params
        params, model, likelihood = self.prepare_inference(results=results)

        # Sort out data
        data = get_data(self.data_params['data_type'], self.data_params)
        self.training_data, self.testing_data = train_test_split(data, test_size=0.2, random_state = 1)
        
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
                        suppress_prints = True)
        
        # Generates the summary object, traceplots and autocorrelation plots
        visualiser.get_summary()
        visualiser.get_traceplot()
        visualiser.get_autocorrelations()

        # Generates the plots and animation of the modeled system using the sampled parameters
        domain = Domain('cone_from_source_z_limited', resolution=80)
        domain.add_domain_param('r', 1000)
        domain.add_domain_param('theta', np.pi/8)
        domain.add_domain_param('source', [0,0,10])
        visualiser.visualise_results(domain = domain, name = 'small_scale_3D_plots', title='Log Concentration of Droplets', log_results=False)
        visualiser.animate(name = 'small_scale_3D_plots')      