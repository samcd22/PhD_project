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
from data_processing.generate_dummy_data import generate_dummy_data

import warnings

# Filter all warnings
warnings.simplefilter("ignore")


class Generator:
    def __init__(self, data_path = 'results/simulated_data_1/inference/auto_gen_results', log_data = False, default_values = {
            'I_y_init_val':0.1,
            'I_y_prior':'gamma',
            'I_y_mu':0.1,
            'I_y_sigma':0.1,
            'I_z_init_val':0.1,
            'I_z_prior':'gamma',
            'I_z_mu':0.1,
            'I_z_sigma':0.1,
            'Q_init_val':3e13,
            'Q_prior':'gamma',
            'Q_mu':3e13,
            'Q_sigma':1e13,
            'model_type':'log_gpm_alt_norm',
            'model_H':10,
            'likelihood_type': 'gaussian_fixed_sigma',
            'likelihood_sigma': 1,
            'n_samples':10000,
            'n_chains':3,
            'thinning_rate':1,
            'I_y_lower': 'NaN',
            'I_y_mean': 0.1,
            'I_y_upper': 'NaN',
            'I_y_tau': 'NaN',
            'I_y_param_accuracy': 'NaN',
            'I_z_lower': 'NaN',
            'I_z_mean': 0.1,
            'I_z_upper': 'NaN',
            'I_z_tau': 'NaN',
            'I_z_param_accuracy': 'NaN',
            'Q_lower': 'NaN',
            'Q_mean': 3e13,
            'Q_upper': 'NaN',
            'Q_tau': 'NaN',
            'Q_param_accuracy': 'NaN',
            'RMSE': 'NaN',
            'av_divergence': 'NaN',
        }):

        self.default_values = pd.Series(default_values)

        self.data_path = data_path
        self.log_data = log_data
        self.par_to_col = {
            'I_y_init_val':('parameters','I_y','init_val'),
            'I_y_prior':('parameters','I_y','prior'),
            'I_y_mu':('parameters','I_y','mu'),
            'I_y_sigma':('parameters','I_y','sigma'),
            'I_z_init_val':('parameters','I_z','init_val'),
            'I_z_prior':('parameters','I_z','prior'),
            'I_z_mu':('parameters','I_z','mu'),
            'I_z_sigma':('parameters','I_z','sigma'),
            'Q_init_val':('parameters','Q','init_val'),
            'Q_prior':('parameters','Q','prior'),
            'Q_mu':('parameters','Q','mu'),
            'Q_sigma':('parameters','Q','sigma'),
            'model_type':('parameters','model','type'),
            'model_H':('parameters','model','H'),
            'likelihood_type': ('parameters','likelihood','type'),
            'likelihood_sigma': ('parameters','likelihood','sigma'),
            'n_samples':('parameters','sampler','n_samples'),
            'n_chains':('parameters','sampler','n_chains'),
            'thinning_rate':('parameters','sampler','thinning_rate'),
            'I_y_lower': ('results','I_y', 'lower'),
            'I_y_mean': ('results','I_y', 'mean'),
            'I_y_upper': ('results','I_y', 'upper'),
            'I_y_tau': ('results','I_y', 'average_tau'),
            'I_y_param_accuracy': ('results','I_y', 'param_accuracy'),
            'I_z_lower': ('results','I_z', 'lower'),
            'I_z_mean': ('results','I_z', 'mean'),
            'I_z_upper': ('results','I_z', 'upper'),
            'I_z_tau': ('results','I_z', 'average_tau'),
            'I_z_param_accuracy': ('results','I_z', 'param_accuracy'),
            'Q_lower': ('results','Q', 'lower'),
            'Q_mean': ('results','Q', 'mean'),
            'Q_upper': ('results','Q', 'upper'),
            'Q_tau': ('results','Q', 'average_tau'),
            'Q_param_accuracy': ('results','Q', 'param_accuracy'),
            'RMSE': ('results','misc', 'RMSE'),
            'av_divergence': ('results','misc', 'average_diverging'),
        }

    def generate_from_inputs(self, inputs, results_path, generate_excel=True):
        instances_path = results_path + '/instances'
        if not os.path.exists(instances_path):
            os.mkdir(instances_path)

        for instance in inputs.index:

            print('Generating instance ' +str(instance) + '...')
            I_y_init_val = np.float64(inputs[('parameters','I_y','init_val')].values[instance-1])
            I_y_prior = inputs[('parameters','I_y','prior')].values[instance-1]
            I_y_mu = np.float64(inputs[('parameters','I_y','mu')].values[instance-1])
            I_y_sigma = np.float64(inputs[('parameters','I_y','sigma')].values[instance-1])
            I_z_init_val = np.float64(inputs[('parameters','I_z','init_val')].values[instance-1])
            I_z_prior = inputs[('parameters','I_z','prior')].values[instance-1]
            I_z_mu = np.float64(inputs[('parameters','I_z','mu')].values[instance-1])
            I_z_sigma = np.float64(inputs[('parameters','I_z','sigma')].values[instance-1])
            Q_init_val = np.float64(inputs[('parameters','Q','init_val')].values[instance-1])
            Q_prior = inputs[('parameters','Q','prior')].values[instance-1]
            Q_mu = np.float64(inputs[('parameters','Q','mu')].values[instance-1])
            Q_sigma = np.float64(inputs[('parameters','Q','sigma')].values[instance-1])
            model_type = inputs[('parameters','model','type')].values[instance-1]
            model_H = np.float64(inputs[('parameters','model','H')].values[instance-1])
            likelihood_type = inputs[('parameters','likelihood','type')].values[instance-1]
            likelihood_sigma = np.float64(inputs[('parameters','likelihood','sigma')].values[instance-1])
            num_samples = int(inputs[('parameters','sampler','n_samples')].values[instance-1])
            num_chains = int(inputs[('parameters','sampler','n_chains')].values[instance-1])
            thinning_rate = int(inputs[('parameters','sampler','thinning_rate')].values[instance-1])


            dummy_data_model_params = {
                'model_params':{
                    'H': 10
                },
                'inference_params':{
                    'I_y': self.default_values['I_y_mean'],
                    'I_z': self.default_values['I_z_mean'],
                    'Q': self.default_values['Q_mean']
                },
            }
            dummy_data_domain_params = {
                'domain_select': 'cone_from_source_z_limited', 
                'resolution': 20,
                'domain_params':{
                    'r': 100,
                    'theta': np.pi/8,
                    'source': [0,0,10]}
            }

            dummy_data = generate_dummy_data(likelihood_sigma, 
                                             self.default_values['model_type'], 
                                             model_params = dummy_data_model_params, 
                                             domain_params=dummy_data_domain_params)
            
            training_data, testing_data = train_test_split(dummy_data, test_size=0.2)

            # Parameter Assignment
            params = pd.Series({},dtype='float64')

            I_y = Parameter(name = 'I_y', init_val=I_y_init_val, prior_select=I_y_prior)
            I_y.add_prior_param("mu",I_y_mu)
            I_y.add_prior_param("sigma",I_y_sigma)
            params['I_y'] = I_y

            I_z = Parameter(name = 'I_z', init_val=I_z_init_val, prior_select=I_z_prior)
            I_z.add_prior_param("mu",I_z_mu)
            I_z.add_prior_param("sigma",I_z_sigma)
            params['I_z'] = I_z

            Q = Parameter(name = 'Q', init_val=Q_init_val, prior_select=Q_prior)
            Q.add_prior_param("mu",Q_mu)
            Q.add_prior_param("sigma",Q_sigma)
            params['Q'] = Q

            # Model Assignment
            model = Model(model_type)
            model.add_model_param("H",model_H)

            # Likelihood function assigmnent
            likelihood = Likelihood(likelihood_type)
            likelihood.add_likelihood_param("sigma",likelihood_sigma)

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
                                    actual_values = [self.default_values['I_y_mean'],self.default_values['I_z_mean'],self.default_values['Q_mean']])
            summary = visualiser.get_summary()
            visualiser.get_traceplot()

            inputs.loc[instance,('results','I_y', 'lower')] = summary['overall']['I_y']['lower']
            inputs.loc[instance,('results','I_y', 'mean')] = summary['overall']['I_y']['mean']
            inputs.loc[instance,('results','I_y', 'upper')] = summary['overall']['I_y']['upper']
            inputs.loc[instance,('results','I_y', 'average_tau')] = summary['overall']['I_y']['tau']
            if 'param_accuracy' in summary['overall']['I_y']:
                inputs.loc[instance,('results','I_y', 'param_accuracy')] = summary['overall']['I_y']['param_accuracy']

            inputs.loc[instance,('results','I_z', 'lower')] = summary['overall']['I_z']['lower']
            inputs.loc[instance,('results','I_z', 'mean')] = summary['overall']['I_z']['mean']
            inputs.loc[instance,('results','I_z', 'upper')] = summary['overall']['I_z']['upper']
            inputs.loc[instance,('results','I_z', 'average_tau')] = summary['overall']['I_z']['tau']
            if 'param_accuracy' in summary['overall']['I_z']:
                inputs.loc[instance,('results','I_z', 'param_accuracy')] = summary['overall']['I_z']['param_accuracy']

            inputs.loc[instance,('results','Q', 'lower')] = summary['overall']['Q']['lower']
            inputs.loc[instance,('results','Q', 'mean')] = summary['overall']['Q']['mean']
            inputs.loc[instance,('results','Q', 'upper')] = summary['overall']['Q']['upper']
            inputs.loc[instance,('results','Q', 'average_tau')] = summary['overall']['Q']['tau']
            if 'param_accuracy' in summary['overall']['Q']:
                inputs.loc[instance,('results','Q', 'param_accuracy')] = summary['overall']['Q']['param_accuracy']

            inputs.loc[instance,('results','misc', 'RMSE')] = summary['RMSE']

            divergences = []
            for i in range(visualiser.num_chains):
                divergences.append(summary['chain_' + str(i+1)]['fields']['diverging'])

            inputs.loc[instance,('results','misc', 'average_diverging')] = np.mean(divergences)

        if generate_excel:
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

        I_y_tau = ('results', 'I_y', 'average_tau')
        I_z_tau = ('results', 'I_z', 'average_tau')
        Q_tau = ('results', 'Q', 'average_tau')
        I_y_param_accuracy = ('results', 'I_y', 'param_accuracy')
        I_z_param_accuracy = ('results', 'I_z', 'param_accuracy')
        Q_param_accuracy = ('results', 'Q', 'param_accuracy')
        RMSE = ('results', 'misc', 'RMSE')
        diverging = ('results', 'misc', 'average_diverging')

        varying_parameter_1 = results[self.par_to_col[parameter_name_1]]
        varying_parameter_2 = results[self.par_to_col[parameter_name_2]]
        RMSE_results = results[RMSE].values.astype('float64')
        diverging_results = np.around(results[diverging].values.astype('float64'))
        I_y_tau_results = results[I_y_tau].values.astype('float64')
        I_z_tau_results = results[I_z_tau].values.astype('float64')
        Q_tau_results = results[Q_tau].values.astype('float64')
        I_y_param_accuracy_results = results[I_y_param_accuracy].values.astype('float64')
        I_z_param_accuracy_results = results[I_z_param_accuracy].values.astype('float64')
        Q_param_accuracy_results = results[Q_param_accuracy].values.astype('float64')
        
        tau_av = np.around(np.mean([I_y_tau_results, I_z_tau_results, Q_tau_results], axis=0))
        param_accuracy_av = np.mean([I_y_param_accuracy_results, I_z_param_accuracy_results, Q_param_accuracy_results], axis=0)

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

        I_y_tau = ('results', 'I_y', 'average_tau')
        I_z_tau = ('results', 'I_z', 'average_tau')
        Q_tau = ('results', 'Q', 'average_tau')
        I_y_param_accuracy = ('results', 'I_y', 'param_accuracy')
        I_z_param_accuracy = ('results', 'I_z', 'param_accuracy')
        Q_param_accuracy = ('results', 'Q', 'param_accuracy')
        RMSE = ('results', 'misc', 'RMSE')
        diverging = ('results', 'misc', 'average_diverging')

        varying_parameter = results[self.par_to_col[parameter_name]]

        fig1 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('RMSE')
        plt.plot(varying_parameter, results[RMSE])
        plt.xscale(xscale)
        plt.title('RMSE of the algorithm for varying ' + parameter_name)
        fig1.savefig(results_path + '/RMSE_plot.png')
        plt.close()

        fig2 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('Divergences')
        plt.plot(varying_parameter, results[diverging])
        plt.xscale(xscale)
        plt.title('Number of divergences of the algorithm for varying ' + parameter_name)
        fig2.savefig(results_path + '/divergence_plot.png')
        plt.close()

        fig3 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('Parameter percentage error (%)')
        plt.plot(varying_parameter, results[I_y_param_accuracy], label = I_y_param_accuracy[1])
        plt.plot(varying_parameter, results[I_z_param_accuracy], label = I_z_param_accuracy[1])
        plt.plot(varying_parameter, results[Q_param_accuracy], label = Q_param_accuracy[1])
        plt.xscale(xscale)
        plt.title('Average parameter percentage error for varying ' + parameter_name)
        plt.legend()
        fig3.savefig(results_path + '/param_accuracy_plot.png')
        plt.close()

        fig4 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('Tau')
        plt.plot(varying_parameter, results[I_y_tau], label = I_y_tau[1])
        plt.plot(varying_parameter, results[I_z_tau], label = I_z_tau[1])
        plt.plot(varying_parameter, results[Q_tau], label = Q_tau[1])
        plt.xscale(xscale)
        plt.title('Convergence of the algorithm for varying ' + parameter_name)
        plt.legend()
        fig4.savefig(results_path + '/convergance_variation.png')
        plt.close()


