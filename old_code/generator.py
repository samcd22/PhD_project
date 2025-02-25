import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt
import json
from numpyencoder import NumpyEncoder
import matplotlib.colors as colors


from app.controllers.controllers.controller import Controller
from toolboxes.regression_toolbox.parameter import Parameter
from app.toolboxes.regression_toolbox.model import Model
from toolboxes.regression_toolbox.likelihood import Likelihood
from toolboxes.regression_toolbox.sampler import Sampler
from toolboxes.regression_toolbox.visualiser import Visualiser
from toolboxes.data_processing_toolbox.get_data import get_data

# Generator class - generates multiple instances of the sampler based on varying their default parameters
class Generator(Controller):
    # Initialises the Generator class saving all relevant variables and performing some initialising tasks
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
            'RMSE': ('results','misc', 'RMSE'),
            'AIC': ('results','misc','AIC'),
            'BIC': ('results','misc','BIC'),
            'av_divergence': ('results','misc', 'average_diverging'),
        }

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

        # Generates the data construction object
        data_construction = self.get_data_construction()

        # Generates the default_params construction object
        default_params_construction = self.get_default_params_construction()

        # Initialises the construction
        self.init_data_construction(data_construction)
        self.init_default_params_construction(default_params_construction)

    def init_default_params_construction(self, construction):
        self.generators_path = self.data_construction_results_path + '/auto_gen_instances'

        if not os.path.exists(self.generators_path):
            os.makedirs(self.generators_path)
        
        generator_name = self.get_generator_name(self.generators_path, construction)

        self.full_results_path = self.generators_path + '/' + generator_name

        if not os.path.exists(self.full_results_path):
            os.makedirs(self.full_results_path)

        if os.path.exists(self.full_results_path + '/default_params_construction.json'):
            f = open(self.full_results_path + '/default_params_construction.json')
            saved_construction = json.load(f)
            f.close()

            if saved_construction != construction:
                raise Exception('Default generator parameters do not match for this folder name!')
        else:
            with open(self.full_results_path + '/default_params_construction.json', "w") as fp:
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

    # Runs a list of instances with hyperparameters based on the inputted inputs data frame
    def generate(self, inputs, results_path):
        # Creates the instances folder
        instances_path = results_path + '/instances'
        if not os.path.exists(instances_path):
            os.mkdir(instances_path)

        # Creates a parameters series with parameters defined by the hyperparameters of a specific instance in the inputs data frame
        def assign_parameters(input_params, inputs, instance):
            params = pd.Series({},dtype='float64')
            for param_name in input_params.index:
                if '_and_' in param_name:
                    params[param_name] = Parameter(name = param_name.split('_and_')[0], name_2 = param_name.split('_and_')[1],
                                               prior_select=inputs[self.par_to_col[param_name + '_prior']].values[instance-1], 
                                               order = inputs[self.par_to_col[param_name + '_order']].values[instance-1])
                else:
                    params[param_name] = Parameter(name = param_name, 
                                               prior_select=inputs[self.par_to_col[param_name + '_prior']].values[instance-1],
                                               order = inputs[self.par_to_col[param_name + '_order']].values[instance-1])

                params = assign_prior_parameters(input_params, params, param_name, inputs, instance)                
            return params
        
        def assign_prior_parameters(input_params, params, param_name, inputs, instance):
            for prior_param_name in input_params[param_name].prior_params.index:
                param_and_prior_param_name = param_name + '_' + prior_param_name
                uncoupled_prior_names = [x for x in self.par_to_col.keys() if param_and_prior_param_name in x]
                prior_param_elements = [x.replace(param_name + '_', '') for x in uncoupled_prior_names]
                modes = []
                if all('_mode_' in x for x in uncoupled_prior_names):
                    modes = np.unique([int(x.split('_mode_')[1]) for x in uncoupled_prior_names])
                    prior_param_elements = [x.split('_mode_')[0] for x in prior_param_elements]
                
                prior_param_elements = np.unique(prior_param_elements)

                if len(modes) == 0 and len(prior_param_elements) == 1:
                    prior_param_value = inputs[self.par_to_col[param_name + '_' + prior_param_name]].values[instance-1]
                    params[param_name].add_prior_param(prior_param_name, prior_param_value)
                
                elif len(modes) != 0 and len(prior_param_elements) == 1:
                    prior_param_value = []
                    for mode in modes:
                        prior_param_value.append(inputs[self.par_to_col[param_name + '_' + prior_param_name + '_mode_' + str(mode)]].values[instance-1])
                    params[param_name].add_prior_param(prior_param_name, prior_param_value)
                
                elif len(modes) == 0 and len(prior_param_elements) != 1:
                    element_coords_list = []
                    for element in prior_param_elements:
                        element_split = element.split('_')
                        element_coords_list.append([int(x) for x in element_split[1:]])
                    
                    if len(element_coords_list[0]) == 1:
                        prior_param_value = np.zeros(np.max(element_coords_list)+1)
                        for element_coord in element_coords_list:
                            prior_param_value[element_coord[0]] = inputs[self.par_to_col[param_name + '_' + prior_param_name + '_' + str(element_coord[0])]].values[instance-1]

                    elif len(element_coords_list[0]) == 2:
                        prior_param_value = np.zeros((np.max(np.array(element_coords_list)[:, 0])+1, np.max(np.array(element_coords_list)[:, 1])+1))
                        for element_coord in element_coords_list:
                            prior_param_value[element_coord[0], element_coord[1]] = inputs[self.par_to_col[param_name + '_' + prior_param_name + '_' + str(element_coord[0]) + '_' + str(element_coord[1])]].values[instance-1]
                    else:
                        raise Exception('Invalid Prior Shape!')

                    params[param_name].add_prior_param(prior_param_name, prior_param_value.tolist())
                            
                else:
                    prior_param_value = []
                    for mode in modes:
                        element_coords_list = []
                        for element in prior_param_elements:
                            element_split = element.split('_')
                            element_coords_list.append([int(x) for x in element_split[1:]])
                        
                        if len(element_coords_list[0]) == 1:
                            prior_param_mode_value = np.zeros(np.max(element_coords_list)+1)
                            for element_coord in element_coords_list:
                                prior_param_mode_value[element_coord[0]] = inputs[self.par_to_col[param_name + '_' + prior_param_name + '_' + str(element_coord[0]) + '_mode_' + str(mode)]].values[instance-1]

                        elif len(element_coords_list[0]) == 2:
                            prior_param_mode_value = np.zeros((np.max(np.array(element_coords_list)[:, 0])+1, np.max(np.array(element_coords_list)[:, 1])+1))
                            for element_coord in element_coords_list:
                                prior_param_mode_value[element_coord[0], element_coord[1]] = inputs[self.par_to_col[param_name + '_' + prior_param_name + '_' + str(element_coord[0]) + '_' + str(element_coord[1]) + '_mode_' + str(mode)]].values[instance-1]
                        else:
                            raise Exception('Invalid Prior Shape!')

                        prior_param_value.append(prior_param_mode_value.tolist())

                    params[param_name].add_prior_param(prior_param_name, prior_param_value)
            
            return params
        
        # Adds the parameter results to a specified instance of the inputs dataframe
        def output_param_results(param_results, inputs, instance):
            param_list = param_results.index
            new_param_list = []
            for param in param_list:
                if '_and_' in param:
                    new_param_list.append(param.split('_and_')[0])
                    new_param_list.append(param.split('_and_')[1])
                else:
                    new_param_list.append(param)

            for param in new_param_list:
                inputs.loc[instance,self.par_to_col[param + '_lower']] = summary['overall'][param]['lower']
                inputs.loc[instance,self.par_to_col[param + '_mean']] = summary['overall'][param]['mean']
                inputs.loc[instance,self.par_to_col[param + '_upper']] = summary['overall'][param]['upper']
                inputs.loc[instance,self.par_to_col[param + '_tau']] = summary['overall'][param]['tau']
                if 'param_accuracy' in summary['overall'][param]:
                    inputs.loc[instance,self.par_to_col[param + '_param_accuracy']] = summary['overall'][param]['param_accuracy']
            return inputs

        # Batch size for processing instances
        batch_size = 2

        # Loops through each instance in the inputs data frame in batches
        for batch_start in range(0, len(inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(inputs))
            instances_batch = inputs.iloc[batch_start:batch_end]

            for instance in instances_batch.index:
                print('Generating instance ' + str(instance) + '...')

                # Generate the infered parameters for this instance
                infered_model_params = assign_parameters(self.default_params['infered_params']['model_params'], inputs, instance)
                infered_likelihood_params = assign_parameters(self.default_params['infered_params']['likelihood_params'], inputs, instance)
                params = pd.concat([infered_model_params, infered_likelihood_params], axis=0)

                # Generates the Model object for this instances
                model = Model(inputs[self.par_to_col['model_type']].values[instance - 1])
                for model_param_name in self.default_params['model'].model_params.index:
                    model.add_model_param(model_param_name, np.float64(inputs[self.par_to_col['model_' + model_param_name]].values[instance - 1]))

                # Generates the Likelihood object for this instances
                likelihood = Likelihood(inputs[self.par_to_col['likelihood_type']].values[instance - 1])
                for likelihood_param_name in self.default_params['likelihood'].likelihood_params.index:
                    likelihood.add_likelihood_param(likelihood_param_name, np.float64(inputs[self.par_to_col['likelihood_' + likelihood_param_name]].values[instance - 1]))

                # Additions to likelihood object under certain conditions
                if self.data_params['data_type'] == 'simulated_data' and self.data_params['noise_percentage'] == 'NaN':
                    raise Exception('Set the noise percentage!')

                # Generates the sampler variables for this instance
                num_samples = int(inputs[self.par_to_col['n_samples']].values[instance - 1])
                num_chains = int(inputs[self.par_to_col['n_chains']].values[instance - 1])
                thinning_rate = int(inputs[self.par_to_col['thinning_rate']].values[instance - 1])

                # Generates the data for this instance
                data = get_data(self.data_params)
                training_data, testing_data = train_test_split(data, test_size=0.2)

                # Actual parameter values are saved if they are available
                actual_values = []
                if self.data_params['data_type'] == 'simulated_data':
                    for inference_param in self.data_params['model']['inference_params'].keys():
                        actual_values.append(self.data_params['model']['inference_params'][inference_param])

                # Generate and visualize the samples based on the specific instance construction
                sampler = Sampler(params, model, likelihood, training_data, testing_data, num_samples, n_chains=num_chains, thinning_rate=thinning_rate, data_path=instances_path)
                sampler.sample_all()
                visualiser = Visualiser(testing_data, sampler, model, previous_instance=sampler.instance, data_path=instances_path, suppress_prints=True, actual_values=actual_values)
                # Saves the summary and traceplots for this instance
                summary = visualiser.get_summary()
                visualiser.get_traceplot()

                # Adds results to this instance of the inputs dataframe
                inputs = output_param_results(self.default_params['infered_params']['model_params'], inputs, instance)
                inputs = output_param_results(self.default_params['infered_params']['likelihood_params'], inputs, instance)
                inputs.loc[instance, self.par_to_col['RMSE']] = summary['RMSE']
                inputs.loc[instance, self.par_to_col['AIC']] = summary['AIC']
                inputs.loc[instance, self.par_to_col['BIC']] = summary['BIC']
                divergences = []
                for i in range(visualiser.num_chains):
                    divergences.append(summary['chain_' + str(i + 1)]['fields']['diverging'])
                inputs.loc[instance, self.par_to_col['av_divergence']] = np.mean(divergences)

            # Delete sampler and visualiser objects to release memory
            del sampler
            del visualiser

        # Saves the edited inputs file after processing all instances
        inputs.to_csv(results_path + '/results.csv')

        return inputs
    
    # Generates instances while varying one parameter beyond the Generator object's default values
    def vary_one_parameter(self, parameter_name, values, plot = True, xscale = 'log'):
        # Creates folders where neccessary
        results_path = self.full_results_path + '/varying_' + parameter_name
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Checks whether the generator has already been run
        if os.path.exists(results_path +'/results.csv'):
            results = pd.read_excel(results_path +'/results.cvs', header=[0,1,2], index_col=0)
        else:
            # Initialises the inputs object using the default values as references for the column names
            inputs = pd.DataFrame(columns=[self.par_to_col[x] for x in self.default_values.index])

            # Adds instances to the inputs data frame line by line varying certain entries where neccessary
            for val in values:
                one_row = self.default_values
                one_row = pd.Series(index = [self.par_to_col[x] for x in self.default_values.index], data = self.default_values.values)
                one_row[self.par_to_col[parameter_name]] = val
                inputs = pd.concat([inputs, one_row.to_frame().T], ignore_index=True)
            inputs.columns = pd.MultiIndex.from_tuples(inputs.columns)
            inputs.index += 1

            # Uses this inputs object to generate all of the results
            results = self.generate(inputs, results_path)

        # If required, plots results from the generated instances
        if plot:
            self.plot_varying_one(results, parameter_name, results_path, xscale = xscale)

        return results
    
    # Generates instances while varying twp parameters beyond the Generator object's default values
    def vary_two_parameters(self, parameter_name_1, parameter_name_2, values_1, values_2, plot = True, scale_1 = 'log', scale_2 = 'log'):

        # Error handing
        if parameter_name_1 == parameter_name_2:
            raise Exception('Varying parameters must be different!')

        # Creates folders where neccessary
        results_path = self.full_results_path + '/varying_' + parameter_name_1 + '_and_' + parameter_name_2
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        # Checks whether the generator has already been run
        if os.path.exists(results_path +'/results.csv'):
            results = pd.read_csv(results_path +'/results.csv', header=[0,1,2], index_col=0)
        else:
            # Initialises the inputs object using the default values as references for the column names
            inputs = pd.DataFrame(columns=[self.par_to_col[x] for x in self.default_values.index])

            # Formats the varying parameters
            V1, V2 = np.meshgrid(values_1, values_2)
            V1_flattened = V1.flatten()
            V2_flattened = V2.flatten()

            # Adds instances to the inputs data frame line by line varying certain entries where neccessary
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
    
            # Uses this inputs object to generate all of the results
            results = self.generate(inputs, results_path)
        
        # If required, plots results from the generated instances
        if plot:
            self.plot_varying_two(results, parameter_name_1, parameter_name_2, results_path, scale_1 = scale_1, scale_2 = scale_2)

        return results
    
    # Generates 2D colour plots of the results from varying two parameters beyond the Generator object's default values
    def plot_varying_two(self, results, parameter_name_1, parameter_name_2, results_path, scale_1 = 'log', scale_2 = 'log'):
        
        # Sorts the results data frame by varying parameters
        results = results.sort_values([self.par_to_col[parameter_name_1], self.par_to_col[parameter_name_2]])

        # Assigns variables
        param_taus = pd.Series({},dtype='float64')
        param_accuracies = pd.Series({},dtype='float64')
        model_inference_param_names = self.default_params['infered_params']['model_params'].index
        likelihood_inference_param_names = self.default_params['infered_params']['likelihood_params'].index
        inference_param_names = [*model_inference_param_names, *likelihood_inference_param_names]
        varying_parameter_1 = results[self.par_to_col[parameter_name_1]]
        varying_parameter_2 = results[self.par_to_col[parameter_name_2]]

        new_inference_param_names = []
        for param_name in inference_param_names:
            if '_and_' in param_name:
                new_inference_param_names.append(param_name.split('_and_')[0])
                new_inference_param_names.append(param_name.split('_and_')[1])
            else:
                new_inference_param_names.append(param_name)

        inference_param_names = new_inference_param_names

        # Checks whether parameter accuracies should be included in plotted results
        param_accuracy_conditional = 'model' in self.data_params and len(self.data_params['model']['inference_params']) == len(inference_param_names)
        
        new_shape = (np.unique(varying_parameter_1).size, np.unique(varying_parameter_2).size)
        
        for param_name in inference_param_names:
            param_taus[param_name] = results[self.par_to_col[param_name + '_tau']].values.astype('float64')
            if param_accuracy_conditional:
                param_accuracies[param_name] = results[self.par_to_col[param_name + '_param_accuracy']].values.astype('float64')
                param_accuracies[param_name] = np.reshape(param_accuracies[param_name], new_shape)
        
        # Extracts success metrics from generated results
        RMSE_results = results[self.par_to_col['RMSE']].values.astype('float64')
        AIC_results = results[self.par_to_col['AIC']].values.astype('float64')
        BIC_results = results[self.par_to_col['BIC']].values.astype('float64')
        diverging_results = np.around(results[self.par_to_col['av_divergence']].values.astype('float64'))
        tau_av = np.around(np.mean(param_taus.values, axis=0))

        # Reshapes metrics for plotting
        varying_parameter_1 = np.reshape([varying_parameter_1], new_shape)
        varying_parameter_2 = np.reshape([varying_parameter_2], new_shape)
        RMSE_results = np.reshape([RMSE_results], new_shape)
        AIC_results = np.reshape([AIC_results], new_shape)
        BIC_results = np.reshape([BIC_results], new_shape)
        diverging_results = np.reshape([diverging_results], new_shape)
        tau_av = np.reshape([tau_av], new_shape)

        # Extracts and reshapes parameter accuracy metric if neccessary
        # if param_accuracy_conditional:
        #     param_accuracy_av = np.mean(param_accuracies.values, axis=0)
        #     param_accuracy_av = np.reshape([param_accuracy_av], new_shape)

        # Sets x scale for plotting
        if scale_1 == 'log':
            scaled_varying_parameter_1 = np.log10(varying_parameter_1.astype(np.float64))
            x_label = 'log ' + parameter_name_1
        else:
            x_label = parameter_name_1
            scaled_varying_parameter_1 = varying_parameter_1.astype(np.float64)
        
        # Sets y scale for plotting
        if scale_2 == 'log':
            scaled_varying_parameter_2 = np.log10(varying_parameter_2.astype(np.float64))
            y_label = 'log ' + parameter_name_2
        else:
            y_label = parameter_name_2
            scaled_varying_parameter_2 = varying_parameter_2.astype(np.float64)

        # RMSE figure
        fig1 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, RMSE_results, cmap='jet')
        plt.title('RMSE of the algorithm for varying \n' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        plt.tight_layout()
        fig1.savefig(results_path + '/RMSE_plot.png')
        plt.close()

        # AIC figure
        fig2 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, AIC_results, cmap='jet', norm=colors.LogNorm(vmin=AIC_results.min(), vmax=AIC_results.max()))
        plt.title('AIC of the algorithm for varying \n' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        plt.tight_layout()
        fig2.savefig(results_path + '/AIC_plot.png')
        plt.close()

        # BIC figure
        fig3 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, BIC_results, cmap='jet', norm=colors.LogNorm(vmin=AIC_results.min(), vmax=AIC_results.max()))
        plt.title('BIC of the algorithm for varying \n' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        plt.tight_layout()
        fig3.savefig(results_path + '/BIC_plot.png')
        plt.close()

        # Divergences figure
        fig4 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, diverging_results, cmap='jet')
        plt.title('Number of divergences of the algorithm for varying \n' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        plt.tight_layout()
        fig4.savefig(results_path + '/divergence_plot.png')
        plt.close()

        # Convergence variation figure
        fig5 = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, tau_av, cmap='jet')
        plt.title('Tau convergence of the algorithm for varying \n' + parameter_name_1 + ' and ' + parameter_name_2)
        plt.colorbar()
        plt.tight_layout()
        fig5.savefig(results_path + '/convergance_variation.png')
        plt.close()

        # Parameter accuracy figure
        if param_accuracy_conditional:
            # fig6 = plt.figure()
            # plt.xlabel(x_label)
            # plt.ylabel(y_label)
            # plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, param_accuracy_av, cmap='jet')
            # plt.title('Average parameter percentage error for varying ' + parameter_name_1 + ' and ' + parameter_name_2)
            # plt.colorbar()
            # plt.tight_layout()
            # fig6.savefig(results_path + '/param_accuracy_plot.png')
            # plt.close()
            for param_name in inference_param_names:
                fig = plt.figure()
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.pcolor(scaled_varying_parameter_1, scaled_varying_parameter_2, param_accuracies[param_name], cmap='jet')
                plt.title('Percentage error of ' + param_name + ' for varying ' + parameter_name_1 + ' and ' + parameter_name_2)
                plt.colorbar()
                plt.tight_layout()
                fig.savefig(results_path + '/' + param_name +'_accuracy_plot.png')
                plt.close()


    # Generates plots of the results from varying one parameter beyond the Generator object's default values
    def plot_varying_one(self, results, parameter_name, results_path, xscale = 'log'):

        # Sorts the results data frame by varying parameter
        results = results.sort_values(self.par_to_col[parameter_name])

        # Assigns variables
        param_taus = pd.Series({})
        param_accuracies = pd.Series({})
        model_inference_param_names = self.default_params['infered_params']['model_params'].index
        likelihood_inference_param_names = self.default_params['infered_params']['likelihood_params'].index
        inference_param_names = [*model_inference_param_names, *likelihood_inference_param_names]

        # Checks whether parameter accuracies should be included in plotted results
        param_accuracy_conditional = 'model' in self.data_params and len(self.data_params['model']['inference_params']) == len(inference_param_names)

        for param_name in inference_param_names:
            param_taus[param_name] = results[self.par_to_col[param_name + '_tau']].values.astype('float64')
            if param_accuracy_conditional:
                param_accuracies[param_name] = results[self.par_to_col[param_name + '_param_accuracy']].values.astype('float64')

        # Extracts success metrics from generated results
        RMSE_results = results[self.par_to_col['RMSE']].values.astype('float64')
        AIC_results = results[self.par_to_col['AIC']].values.astype('float64')
        BIC_results = results[self.par_to_col['BIC']].values.astype('float64')
        diverging_results = np.around(results[self.par_to_col['av_divergence']].values.astype('float64'))
        
        varying_parameter = results[self.par_to_col[parameter_name]]

        # RMSE figure
        fig1 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('RMSE')
        plt.plot(varying_parameter, RMSE_results)
        plt.xscale(xscale)
        plt.title('RMSE of the algorithm for varying \n' + parameter_name)
        plt.tight_layout()
        fig1.savefig(results_path + '/RMSE_plot.png')
        plt.close()

        # AIC figure
        fig2 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('AIC')
        plt.plot(varying_parameter, AIC_results)
        plt.xscale(xscale)
        plt.title('AIC of the algorithm for varying \n' + parameter_name)
        plt.tight_layout()
        fig2.savefig(results_path + '/AIC_plot.png')
        plt.close()

        # BIC figure
        fig3 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('BIC')
        plt.plot(varying_parameter, BIC_results)
        plt.xscale(xscale)
        plt.title('BIC of the algorithm for varying \n' + parameter_name)
        plt.tight_layout()
        fig3.savefig(results_path + '/BIC_plot.png')
        plt.close()

        # Divergences figure
        fig4 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('Divergences')
        plt.plot(varying_parameter, diverging_results)
        plt.xscale(xscale)
        plt.title('Number of divergences of the algorithm for varying \n' + parameter_name)
        plt.tight_layout()
        fig4.savefig(results_path + '/divergence_plot.png')
        plt.close()

        # Convergence variation figure
        fig5 = plt.figure()
        plt.xlabel(parameter_name)
        plt.ylabel('Tau')
        for param_name in param_taus.index:
            plt.plot(varying_parameter, param_taus[param_name], label = param_name)
        plt.xscale(xscale)
        plt.title('Convergence of the algorithm for varying \n' + parameter_name)
        plt.legend()
        plt.tight_layout()
        fig5.savefig(results_path + '/convergance_variation.png')
        plt.close()

        # Parameter accuracy figure
        if param_accuracy_conditional:
            fig6 = plt.figure()
            plt.xlabel(parameter_name)
            plt.ylabel('Parameter percentage error (%)')
            for param_name in param_accuracies.index:
                plt.plot(varying_parameter, param_accuracies[param_name], label = param_name)
            plt.xscale(xscale)
            plt.title('Average parameter percentage error for varying \n' + parameter_name)
            plt.legend()
            plt.tight_layout()
            fig6.savefig(results_path + '/param_accuracy_plot.png')
            plt.close()


