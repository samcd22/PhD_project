import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt

from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from inference_toolbox.sampler import Sampler
from inference_toolbox.visualiser import Visualiser
from data_processing.get_data import get_data

import warnings

# Filter all warnings
warnings.simplefilter("ignore")


class Generator:
    def __init__(self, 
                generator_defaults = {
                    'infered_params':pd.Series({
                        'model_params':pd.Series({
                            'I_y': Parameter('I_y', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'I_z': Parameter('I_z', prior_select = 'gamma', default_value=0.1).add_prior_param('mu', 0.1).add_prior_param('sigma',0.1),
                            'Q': Parameter('Q', prior_select = 'gamma', default_value=3e13).add_prior_param('mu', 3e13).add_prior_param('sigma',1e13),
                        }),
                        'likelihood_params':pd.Series({})
                    }),
                    'model':Model('log_gpm_alt_norm').add_model_param('H',10),
                    'likelihood': Likelihood('gaussian_fixed_sigma').add_likelihood_param('sigma',1),
                    'sampler': {
                        'n_samples': 10000,
                        'n_chains': 3,
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
                data_path = 'results',
                data_name = 'simulated_data_1'):

        full_data_path = data_path + '/' + data_name + '/inference/auto_gen_results'

        if not os.path.exists(data_path + '/' + data_name):
            os.mkdir(data_path + '/' + data_name)
            if not os.path.exists(data_path + '/' + data_name + '/inference'):
                os.mkdir(data_path + '/' + data_name + '/inference')
                if not os.path.exists(full_data_path):
                    os.mkdir(full_data_path)
        
        self.generator_defaults = generator_defaults

        self.data_path = full_data_path

        self.default_values = pd.Series({
            'RMSE': 'NaN',
            'av_divergence': 'NaN'
        })
    
        self.par_to_col = {
            'RMSE': ('results','misc', 'RMSE'),
            'av_divergence': ('results','misc', 'average_diverging'),
        }

        self.data_params = data_params

        sampler_info = generator_defaults['sampler']

        self.default_values['n_samples'] = sampler_info['n_samples']
        self.par_to_col['n_samples'] = ('parameters','sampler','n_samples')

        self.default_values['n_chains'] = sampler_info['n_chains']
        self.par_to_col['n_chains'] = ('parameters','sampler','n_chains')

        self.default_values['thinning_rate'] = sampler_info['thinning_rate']
        self.par_to_col['thinning_rate'] = ('parameters','sampler','thinning_rate')

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
        
        infered_model_params = generator_defaults['infered_params']['model_params']
        infered_likelihood_params = generator_defaults['infered_params']['likelihood_params']

        set_param_default_values(infered_model_params)
        set_param_default_values(infered_likelihood_params)

        model = generator_defaults['model']
        self.default_values['model_type'] = model.model_select
        self.par_to_col['model_type'] = ('parameters', 'model', 'type')

        for model_param in model.model_params.index:
            self.default_values['model_' + model_param] = model.model_params[model_param]
            self.par_to_col['model_' + model_param] = ('parameters', 'model', model_param)

        likelihood = generator_defaults['likelihood']
        self.default_values['likelihood_type'] = likelihood.likelihood_select
        self.par_to_col['likelihood_type'] = ('parameters', 'likelihood', 'type')

        for likelihood_param in likelihood.likelihood_params.index:
            self.default_values['likelihood_' + likelihood_param] = likelihood.likelihood_params[likelihood_param]
            self.par_to_col['likelihood_' + likelihood_param] = ('parameters', 'model', likelihood_param)

    def generate_from_inputs(self, inputs, results_path):
        instances_path = results_path + '/instances'
        if not os.path.exists(instances_path):
            os.mkdir(instances_path)

        def assign_parameters(input_params, inputs, instance):
            params = pd.Series({},dtype='float64')
            for param_name in input_params.index:
                params[param_name] = Parameter(name = param_name, 
                                               prior_select=inputs[self.par_to_col[param_name + '_prior']].values[instance-1])
                for prior_param_name in input_params[param_name].prior_params.index:
                    params[param_name].add_prior_param(prior_param_name, 
                                                       np.float64(inputs[self.par_to_col[param_name + '_' + prior_param_name]].values[instance-1]))
            return params
        
        def output_param_results(input_params, inputs, instance):
            for param in input_params.index:
                inputs.loc[instance,self.par_to_col[param + '_lower']] = summary['overall'][param]['lower']
                inputs.loc[instance,self.par_to_col[param + '_mean']] = summary['overall'][param]['mean']
                inputs.loc[instance,self.par_to_col[param + '_upper']] = summary['overall'][param]['upper']
                inputs.loc[instance,self.par_to_col[param + '_tau']] = summary['overall'][param]['tau']
                if 'param_accuracy' in summary['overall'][param]:
                    inputs.loc[instance,self.par_to_col[param + '_param_accuracy']] = summary['overall'][param]['param_accuracy']
            return inputs

        for instance in inputs.index:
            print('Generating instance ' +str(instance) + '...')
            
            infered_model_params = assign_parameters(self.generator_defaults['infered_params']['model_params'], inputs, instance)
            infered_likelihood_params = assign_parameters(self.generator_defaults['infered_params']['likelihood_params'], inputs, instance)
            params = pd.concat([infered_model_params, infered_likelihood_params], axis = 0)

            # Model Assignment
            model = Model(inputs[self.par_to_col['model_type']].values[instance-1])
            for model_param_name in self.generator_defaults['model'].model_params.index:
                model.add_model_param(model_param_name, 
                                      np.float64(inputs[self.par_to_col['model_' + model_param_name]].values[instance-1]))
                

            likelihood = Likelihood(inputs[self.par_to_col['likelihood_type']].values[instance-1])
            for likelihood_param_name in self.generator_defaults['likelihood'].likelihood_params.index:
                likelihood.add_likelihood_param(likelihood_param_name, 
                                                np.float64(inputs[self.par_to_col['likelihood_' + likelihood_param_name]].values[instance-1]))

            if self.data_params['data_type'] == 'dummy':
                if self.data_params['sigma'] == 'NaN':
                    if 'sigma' not in likelihood.likelihood_params:
                        Exception('Either define your noise level with a fixed sigma in the likelihood, or set the noise level!')
                    self.data_params['sigma'] = likelihood.likelihood_params['sigma']


            num_samples = int(inputs[self.par_to_col['n_samples']].values[instance-1])
            num_chains = int(inputs[self.par_to_col['n_chains']].values[instance-1])
            thinning_rate = int(inputs[self.par_to_col['thinning_rate']].values[instance-1])

            data = get_data(self.data_params['data_type'], self.data_params)
            
            training_data, testing_data = train_test_split(data, test_size=0.2)

            actual_values = []
            if self.data_params['data_type'] == 'dummy':
                for inference_param in self.data_params['model']['inference_params'].keys():
                    actual_values.append(self.data_params['model']['inference_params'][inference_param])

            # Run sampler and visualiser
            sampler = Sampler(params, model, likelihood, training_data, num_samples, show_sample_info = True, n_chains=num_chains, thinning_rate=thinning_rate, data_path = instances_path)
            params_samples, chain_samples, fields = sampler.sample_all()
            visualiser = Visualiser(testing_data, 
                                    params_samples, 
                                    model, sampler.hyperparams, 
                                    chain_samples=chain_samples,
                                    fields = fields, 
                                    previous_instance = sampler.instance, 
                                    data_path = instances_path, 
                                    suppress_prints = True, 
                                    actual_values = actual_values)
            summary = visualiser.get_summary()
            visualiser.get_traceplot()

            inputs = output_param_results(self.generator_defaults['infered_params']['model_params'],inputs, instance)
            inputs = output_param_results(self.generator_defaults['infered_params']['likelihood_params'],inputs, instance)

            inputs.loc[instance,self.par_to_col['RMSE']] = summary['RMSE']

            divergences = []
            for i in range(visualiser.num_chains):
                divergences.append(summary['chain_' + str(i+1)]['fields']['diverging'])

            inputs.loc[instance,self.par_to_col['av_divergence']] = np.mean(divergences)

            # inputs.columns=inputs.sort_index(axis=1,level=[0,1,2],ascending=[True,False, False]).columns

            inputs.to_excel(results_path + '/results.xlsx')

        return inputs

    def iterate_through_excel(self, file_path):
        inputs = pd.read_excel(file_path,header=[0,1,2], index_col=0)
        return self.generate_from_inputs(inputs)
    
    def vary_one_parameter(self, parameter_name, values, plot = True, xscale = 'log'):
        results_path = self.data_path + '/varying_' + parameter_name
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        if os.path.exists(results_path +'/results.xlsx'):
            results = pd.read_excel(results_path +'/results.xlsx', header=[0,1,2], index_col=0)
        else:
        
            inputs = pd.DataFrame(columns=[self.par_to_col[x] for x in self.default_values.index])
            for val in values:
                one_row = self.default_values
                one_row = pd.Series(index = [self.par_to_col[x] for x in self.default_values.index], data = self.default_values.values)
                one_row[self.par_to_col[parameter_name]] = val
                inputs = pd.concat([inputs, one_row.to_frame().T], ignore_index=True)
            inputs.columns = pd.MultiIndex.from_tuples(inputs.columns)
            inputs.index += 1
            results = self.generate_from_inputs(inputs, results_path)

        if plot:
            self.plot_varying_one(results, parameter_name, results_path, xscale = xscale)

        return results
    
    def vary_two_parameters(self, parameter_name_1, parameter_name_2, values_1, values_2, plot = True, scale_1 = 'log', scale_2 = 'log'):
        inputs = pd.DataFrame(columns=[self.par_to_col[x] for x in self.default_values.index])
        if parameter_name_1 == parameter_name_2:
            Exception('Varying parameters must be different!')

        results_path = self.data_path + '/varying_' + parameter_name_1 + '_and_' + parameter_name_2
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        if os.path.exists(results_path +'/results.xlsx'):
            results = pd.read_excel(results_path +'/results.xlsx', header=[0,1,2], index_col=0)
        else:

            V1, V2 = np.meshgrid(values_1, values_2)
            V1_flattened = V1.flatten()
            V2_flattened = V2.flatten()

            for i in range(V1_flattened.size):
                val1 = V1_flattened[i]
                val2 = V2_flattened[i]
                one_row = self.default_values
                one_row = pd.Series(index = [self.par_to_col[x] for x in self.default_values.index], data = self.default_values.values)
                one_row[self.par_to_col[parameter_name_1]] = val1
                one_row[self.par_to_col[parameter_name_2]] = val2

                inputs = pd.concat([inputs, one_row.to_frame().T], ignore_index=True)
            inputs.columns = pd.MultiIndex.from_tuples(inputs.columns)
            inputs.index += 1
    
            results = self.generate_from_inputs(inputs, results_path)

        if plot:
            self.plot_varying_two(results, parameter_name_1, parameter_name_2, results_path, scale_1 = scale_1, scale_2 = scale_2)

        return results
    
    def plot_varying_two(self, results, parameter_name_1, parameter_name_2, results_path, scale_1 = 'log', scale_2 = 'log'):
        results = results.sort_values([self.par_to_col[parameter_name_1], self.par_to_col[parameter_name_2]])

        param_taus = pd.Series({})
        param_accuracies = pd.Series({})

        model_inference_param_names = self.generator_defaults['infered_params']['model_params'].index
        likelihood_inference_param_names = self.generator_defaults['infered_params']['likelihood_params'].index

        inference_param_names = [*model_inference_param_names, *likelihood_inference_param_names]

        param_accuracy_conditional = len(self.data_params['model']['inference_params']) == len(inference_param_names)

        for param_name in inference_param_names:
            param_taus[param_name] = results[self.par_to_col[param_name + '_tau']].values.astype('float64')
            if param_accuracy_conditional:
                param_accuracies[param_name] = results[self.par_to_col[param_name + '_param_accuracy']].values.astype('float64')


        varying_parameter_1 = results[self.par_to_col[parameter_name_1]]
        varying_parameter_2 = results[self.par_to_col[parameter_name_2]]

        RMSE_results = results[self.par_to_col['RMSE']].values.astype('float64')
        diverging_results = np.around(results[self.par_to_col['av_divergence']].values.astype('float64'))
        
        tau_av = np.around(np.mean(param_taus.values, axis=0))
        param_accuracy_av = np.mean([param_accuracies.values], axis=0)

        new_shape = (np.unique(varying_parameter_1).size, np.unique(varying_parameter_2).size)

        varying_parameter_1 = np.reshape([varying_parameter_1], new_shape)
        varying_parameter_2 = np.reshape([varying_parameter_2], new_shape)
        RMSE_results = np.reshape([RMSE_results], new_shape)
        diverging_results = np.reshape([diverging_results], new_shape)
        tau_av = np.reshape([tau_av], new_shape)
        param_accuracy_av = np.reshape([param_accuracy_av], new_shape)

        if scale_1 == 'log':
            scaled_varying_parameter_1 = np.log10(varying_parameter_1.astype(np.float64))
            x_label = 'log ' + parameter_name_1
        else:
            x_label = parameter_name_1
            scaled_varying_parameter_1 = varying_parameter_1.astype(np.float64)

        if scale_2 == 'log':
            scaled_varying_parameter_2 = np.log10(varying_parameter_2.astype(np.float64))
            y_label = 'log ' + parameter_name_2
        else:
            y_label = parameter_name_2
            scaled_varying_parameter_2 = varying_parameter_2.astype(np.float64)

        fig1 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, RMSE_results, cmap='jet')
        plt.title('RMSE of the algorithm for varying ' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        fig1.savefig(results_path + '/RMSE_plot.png')
        plt.close()

        fig2 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, diverging_results, cmap='jet')
        plt.title('Number of divergences of the algorithm for varying ' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        fig2.savefig(results_path + '/divergence_plot.png')
        plt.close()

        fig3 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, tau_av, cmap='jet')
        plt.title('Tau convergence of the algorithm for varying ' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        fig3.savefig(results_path + '/convergance_variation.png')
        plt.close()

        if param_accuracy_conditional:
            fig4 = plt.figure()
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, param_accuracy_av, cmap='jet')
            plt.title('Average parameter percentage error for varying ' + parameter_name_1 + ' and ' + parameter_name_2)
            plt.colorbar()
            fig4.savefig(results_path + '/param_accuracy_plot.png')
            plt.close()


    def plot_varying_one(self, results, parameter_name, results_path, xscale = 'log'):

        results = results.sort_values(self.par_to_col[parameter_name])

        param_taus = pd.Series({})
        param_accuracies = pd.Series({})

        model_inference_param_names = self.generator_defaults['infered_params']['model_params'].index
        likelihood_inference_param_names = self.generator_defaults['infered_params']['likelihood_params'].index

        inference_param_names = [*model_inference_param_names, *likelihood_inference_param_names]

        param_accuracy_conditional = len(self.data_params['model']['inference_params']) == len(inference_param_names)

        for param_name in inference_param_names:
            param_taus[param_name] = results[self.par_to_col[param_name + '_tau']].values.astype('float64')
            if param_accuracy_conditional:
                param_accuracies[param_name] = results[self.par_to_col[param_name + '_param_accuracy']].values.astype('float64')


        RMSE_results = results[self.par_to_col['RMSE']].values.astype('float64')
        diverging_results = np.around(results[self.par_to_col['av_divergence']].values.astype('float64'))
        
        varying_parameter = results[self.par_to_col[parameter_name]]

        fig1 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('RMSE')
        plt.plot(varying_parameter, RMSE_results)
        plt.xscale(xscale)
        plt.title('RMSE of the algorithm for varying ' + parameter_name)
        fig1.savefig(results_path + '/RMSE_plot.png')
        plt.close()

        fig2 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('Divergences')
        plt.plot(varying_parameter, diverging_results)
        plt.xscale(xscale)
        plt.title('Number of divergences of the algorithm for varying ' + parameter_name)
        fig2.savefig(results_path + '/divergence_plot.png')
        plt.close()

        fig3 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('Tau')
        for param_name in param_taus.index:
            plt.plot(varying_parameter, param_taus[param_name], label = param_name)

        plt.xscale(xscale)
        plt.title('Convergence of the algorithm for varying ' + parameter_name)
        plt.legend()
        fig3.savefig(results_path + '/convergance_variation.png')
        plt.close()

        if param_accuracy_conditional:
            fig4 = plt.figure()
            plt.xlabel(parameter_name)
            plt.ylabel('Parameter percentage error (%)')
            for param_name in param_accuracies.index:
                plt.plot(varying_parameter, param_accuracies[param_name], label = param_name)

            plt.xscale(xscale)
            plt.title('Average parameter percentage error for varying ' + parameter_name)
            plt.legend()
            fig4.savefig(results_path + '/param_accuracy_plot.png')
            plt.close()


