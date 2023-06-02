import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import os
import json
from numpyencoder import NumpyEncoder
import imageio
from matplotlib.gridspec import GridSpec
from fractions import Fraction


class Visualiser:
    def __init__(self, 
                 test_data,
                 samples, 
                 model, 
                 hyperparams, 
                 chain_samples = -1, 
                 fields = {}, 
                 previous_instance = -1, 
                 data_path = 'results/inference/general_instances', 
                 include_test_points = True, 
                 suppress_prints = False,
                 actual_values = []):
        

        self.suppress_prints = suppress_prints
        self.test_data = test_data

        self.model = model
        self.model_func = model.get_model()
        self.hyperparams = hyperparams
        # self.acceptance_rates = acceptance_rates
        self.data_path = data_path
        self.instance = self.get_instance(previous_instance)
        self.save_hyperparams()
        self.num_chains = 1
        self.include_test_points = include_test_points
        self.fields = fields

        self.sample_data_generated = True
        self.chain_data_generated = True
        self.chain_data_inputted = True

        if type(samples) == list and samples == []:
            self.sample_data_generated = False

        if type(chain_samples) == list and chain_samples == []:
            self.chain_data_generated = False

        if type(chain_samples) == int and chain_samples == -1:
            self.chain_data_inputted = False
        
        if self.chain_data_inputted:
            if not self.chain_data_generated:
                self.chain_samples = self.load_samples(chain=True)
            else:
                self.chain_samples = chain_samples
            self.num_chains = int(np.max(self.chain_samples['chain'].unique()))

        if not self.sample_data_generated:
            self.samples = self.load_samples()
        else:
            self.samples = samples

        self.actual_values = {}
        for i, param in enumerate(self.samples.columns):
            if actual_values == []:
                self.actual_values[param] = 'NaN'
            else:
                self.actual_values[param] = actual_values[i]

        self.params_lower = self.get_ag_samples(self.samples, 0.05)
        self.params_mean = self.get_ag_samples(self.samples, 0.5)
        self.params_upper = self.get_ag_samples(self.samples, 0.95)

        self.RMSE = 'n/a'
        if self.include_test_points:
            mean_test_pred_C = self.model_func(self.params_mean, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            mean_test_actual_C = self.test_data['Concentration']
            self.RMSE = np.sqrt(np.mean((mean_test_pred_C-mean_test_actual_C)**2))

        self.calculate_autocorrs()

        self.save_samples()

    def get_traceplot(self):
        traceplot_folder = self.data_path + '/instance_' + str(self.instance) + '/traceplots'
        if not os.path.exists(traceplot_folder):
            os.mkdir(traceplot_folder)

        for chain in range(self.num_chains):
            full_path = traceplot_folder + '/traceplot_' + str(chain + 1) + '.png'
            if self.num_chains == 1:
                title = 'MCMC samples'
                samples = self.samples
            else:
                title = 'MCMC samples for chain ' + str(chain + 1)
                samples = self.chain_samples[self.chain_samples['chain'] == chain + 1].drop(columns = ['chain', 'sample_index'])
            
            if os.path.exists(full_path):
                if not self.suppress_prints:
                    print('Traceplot ' + str(chain + 1) + ' already exists')
            else:
                tp = self.traceplots(samples.values, xnames = self.samples.columns, title = title)
                tp.savefig(full_path)

    def traceplots(self, x, xnames = None, title = None):
        _, d = x.shape
        fig = plt.figure()
        left, tracewidth, histwidth = 0.1, 0.65, 0.15
        bottom, rowheight = 0.1, 0.8/d

        for i in range(d):
            rowbottom = bottom + i * rowheight
            rect_trace = (left, rowbottom, tracewidth, rowheight)
            rect_hist = (left + tracewidth, rowbottom, histwidth, rowheight)

            if i == 0:
                ax_trace = fig.add_axes(rect_trace)
                ax_trace.plot(x[:,i])
                ax_trace.set_xlabel("Sample Count")
                ax_tr0 = ax_trace

            elif i > 0:
                ax_trace = fig.add_axes(rect_trace, sharex=ax_tr0)
                ax_trace.plot(x[:,i])
                plt.setp(ax_trace.get_xticklabels(), visible=False)

            if i == d-1 and title is not None:
                plt.title(title)

            if xnames is not None:
                ax_trace.set_ylabel(xnames[i])

            ax_hist = fig.add_axes(rect_hist, sharey=ax_trace)
            try:
                ax_hist.hist(x[:,i], orientation='horizontal', bins=50)
            except:
                if not self.suppress_prints:
                    print('Traceplot histogram invalid!')

            plt.setp(ax_hist.get_xticklabels(), visible=False)
            plt.setp(ax_hist.get_yticklabels(), visible=False)
            xlim = ax_hist.get_xlim()
            ax_hist.set_xlim([xlim[0], 1.1*xlim[1]])
            plt.close()
        return fig

    def visualise_results(self, domain, name, plot_type = '3D', log_results = True, title = 'Concentration of Droplets'):
        points = domain.create_domain()
        if os.path.exists(self.data_path + '/instance_' + str(self.instance) + '/' + name):
            if not self.suppress_prints:
                print('Plots already exist!')

        elif plot_type == '3D':
            X, Y, Z = points[:,0], points[:,1], points[:,2]
            C_lower = self.model_func(self.params_lower, X,Y,Z)
            C_mean = self.model_func(self.params_mean, X,Y,Z)
            C_upper = self.model_func(self.params_upper, X,Y,Z)

            results_df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten(), 'C_lower': C_lower, 'C_mean': C_mean,'C_upper': C_upper})

            self.threeD_plots(results_df, name, log_results=log_results, title=title)

    def threeD_plots(self, results, name, q=10, log_results = True, title = 'Concentration of Droplets'):

        X = results.x
        Y = results.y
        Z = results.z
        C_lower = results.C_lower
        C_mean = results.C_mean
        C_upper = results.C_upper

        # Modifying C depending on if the results need logging
        if log_results:
            C_lower[C_lower<1] = 1
            C_lower = np.log10(C_lower)
            
            C_mean[C_mean<1] = 1
            C_mean = np.log10(C_mean)
            
            C_upper[C_upper<1] = 1
            C_upper = np.log10(C_upper)
        
        # Define the bin numbers and their labels
        lower_bin_nums = pd.qcut(C_lower,q, labels = False, duplicates='drop')
        lower_bin_labs = np.array(pd.qcut(C_mean,q, duplicates='drop').unique())

        mean_bin_nums = pd.qcut(C_mean,q, labels = False, duplicates='drop')
        mean_bin_labs = np.array(pd.qcut(C_mean,q, duplicates='drop').unique())

        upper_bin_nums = pd.qcut(C_upper,q, labels = False, duplicates='drop')
        upper_bin_labs = np.array(pd.qcut(C_mean,q, duplicates='drop').unique())

        # Define the dataframe of results with bin numbers attached
        lower_conc_and_bins = pd.DataFrame([X, Y, Z, C_lower, lower_bin_nums]).T
        lower_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        mean_conc_and_bins = pd.DataFrame([X, Y, Z, C_mean, mean_bin_nums]).T
        mean_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        upper_conc_and_bins = pd.DataFrame([X, Y, Z, C_upper, upper_bin_nums]).T
        upper_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        # Define min and max values for colorbar
        min_val = np.percentile([C_lower, C_mean, C_upper],10)
        max_val = np.percentile([C_lower, C_mean, C_upper],90)

        # I NEED TO EDIT THIS TO IT CHANGES FOR EACH PLOT!!!!!!
        # Calculates the percentage differances and RMSE if the test points are set to be included
        if self.include_test_points:
            lower_test_pred_C = self.model_func(self.params_lower, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            lower_test_actual_C = self.test_data['Concentration']
            lower_percentage_difference = 2*np.abs(lower_test_actual_C-lower_test_pred_C)/(lower_test_actual_C + lower_test_pred_C) * 100

            mean_test_pred_C = self.model_func(self.params_mean, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            mean_test_actual_C = self.test_data['Concentration']
            mean_percentage_difference = 2*np.abs(mean_test_actual_C-mean_test_pred_C)/(mean_test_actual_C + mean_test_pred_C) * 100

            upper_test_pred_C = self.model_func(self.params_upper, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            upper_test_actual_C = self.test_data['Concentration']
            upper_percentage_difference = 2*np.abs(lower_test_actual_C-upper_test_pred_C)/(upper_test_actual_C + upper_test_pred_C) * 100

     
            
        # Creates a directory for the instance if the save parameter is selected
        os.mkdir(self.data_path + '/instance_' + str(self.instance) + '/' + str(name))
        os.mkdir(self.data_path + '/instance_' + str(self.instance) + '/' + str(name) + '/figures')
        
        # Loops through each bin number and generates a figure with the bin data 
        for bin_num in np.sort(np.unique(mean_bin_nums)):

            lower_bin_data = lower_conc_and_bins[lower_conc_and_bins['bin'] >= bin_num]
            mean_bin_data = mean_conc_and_bins[mean_conc_and_bins['bin'] >= bin_num]
            upper_bin_data = upper_conc_and_bins[upper_conc_and_bins['bin'] >= bin_num]

            fin_alpha = len(self.params_mean)*0.75

            y = 12
            r = 0.8
            alpha = (1-r)*y

            delta_alpha = fin_alpha - alpha

            y_prime = y + delta_alpha
            r_prime = r*y/y_prime

            r_prime_frac = Fraction(r_prime)

            fig = plt.figure(constrained_layout = True, figsize = (24,y_prime))
            spec = GridSpec(2, 6, figure = fig, height_ratios= [r_prime_frac.numerator, r_prime_frac.denominator - r_prime_frac.numerator])

            ax1 = fig.add_subplot(spec[0,:2], projection='3d')
            ax2 = fig.add_subplot(spec[0,2:4], projection='3d')
            ax3 = fig.add_subplot(spec[0,4:], projection='3d')
            ax4 = fig.add_subplot(spec[1,:3])
            ax5 = fig.add_subplot(spec[1,3:])
            
            # Defines the lower bound subplot
            plot_1 = ax1.scatter(lower_bin_data.x, lower_bin_data.y, lower_bin_data.z, c = lower_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax1.set_title('Generated by the lower bound parameters')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_xlim(np.min(X), np.max(X))
            ax1.set_ylim(np.min(Y), np.max(Y))
            ax1.set_zlim(np.min(Z), np.max(Z))

            # Defines the mean subplot
            ax2.scatter(mean_bin_data.x, mean_bin_data.y, mean_bin_data.z, c = mean_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax2.set_title('Generated by the mean parameters')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax2.set_xlim(np.min(X), np.max(X))
            ax2.set_ylim(np.min(Y), np.max(Y))
            ax2.set_zlim(np.min(Z), np.max(Z))

            # Defines the upper bound subplot
            ax3.scatter(upper_bin_data.x, upper_bin_data.y, upper_bin_data.z, c = upper_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax3.set_title('Generated by the upper bound parameters')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')
            ax3.set_xlim(np.min(X), np.max(X))
            ax3.set_ylim(np.min(Y), np.max(Y))
            ax3.set_zlim(np.min(Z), np.max(Z))

            # Generates the test point data on each graph
            if self.include_test_points:
                pd_min = np.min([lower_percentage_difference, mean_percentage_difference, upper_percentage_difference])
                pd_max = np.min([lower_percentage_difference, mean_percentage_difference, upper_percentage_difference])

                plot_2 = ax1.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 10*np.log10(lower_test_pred_C), c = lower_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                ax2.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 10*np.log10(mean_test_pred_C), c = mean_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                ax3.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 10*np.log10(upper_test_pred_C), c = upper_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                formatter = "{:.2e}" 
                if  np.floor(np.log10(self.RMSE)) < 2: formatter = "{:.2f}" 
                sampling_data_string = 'RMSE = ' + formatter.format(self.RMSE)
            else:
                sampling_data_string = 'RMSE = n/a'

            # Creates an axis to display the parameter values
            param_string = self.get_param_string()
            ax4.text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
            ax4.set_xticks([])
            ax4.set_yticks([])

            param_accuracy_string = self.get_param_accuracy_string()

            # Creates an axis to display the sampling information
            ax5.text(0.5,0.5, sampling_data_string + '\n' + param_accuracy_string, fontsize = 30, va = "center", ha = 'center')
            ax5.set_xticks([])
            ax5.set_yticks([])

            # Defines the two colorbars
            plt.colorbar(plot_1, ax = ax4, location = 'top', shrink = 2)
            plt.colorbar(plot_2, ax = ax5, location = 'top', shrink = 2)

            # Defines the overall title, including the range of values for each plot
            if mean_bin_labs[bin_num].left < 0:
                left_bound = 0
            else:
                left_bound = mean_bin_labs[bin_num].left
            fig.suptitle(title + '\nValues for mean plot greater than ' + "{:.2f}".format(left_bound), fontsize = 32)

            # Saves the figures if required
            fig_name = 'fig_' + str(bin_num + 1) + '_of_' + str(np.max(mean_bin_nums + 1)) + '.png'
            full_path = self.data_path + '/instance_' + str(self.instance) + '/' + name + '/figures/' + fig_name
            if not os.path.exists(full_path):
                fig.savefig(full_path)
            plt.close()

    def get_param_accuracy_string(self):
        param_accuracy_string_array = []
        for param in self.samples.columns:
            if not self.actual_values[param] == 'NaN':
                percentage_error = 200*np.abs(self.params_mean[param]-self.actual_values[param])/(self.params_mean[param] + self.actual_values[param])
                param_accuracy_string_array.append(param + ' error = ' + f'{percentage_error:.3}' + '%')
        return ('\n').join(param_accuracy_string_array)


    def get_param_string(self):
        param_string_array = []
        for param in self.params_mean.index:
            if np.floor(np.log10(self.params_mean[param])) < 2:
                lower_string =  "{:.2f}".format(self.params_lower[param])
                mean_string = "{:.2f}".format(self.params_mean[param])
                upper_string = "{:.2f}".format(self.params_upper[param])
            else:
                lower_string =  "{:.2e}".format(self.params_lower[param])
                mean_string = "{:.2e}".format(self.params_mean[param])
                upper_string = "{:.2e}".format(self.params_upper[param])
            param_string_array.append(param + ' = [' + lower_string + ', ' + mean_string + ', ' + upper_string + ']')

        return ('\n').join(param_string_array)
        
    def get_instance(self, previous_instance):
        if previous_instance != -1:
            instance = previous_instance
        else:
            instance_folders = os.listdir(self.data_path)
            instances = [int(x.split('_')[1]) for x in instance_folders]
            missing_elements = []
            if len(instances) == 0:
                instance = 1
            else:
                for el in range(1,np.max(instances) + 2):
                    if el not in instances:
                        missing_elements.append(el)
                instance = np.min(missing_elements)

        instance_path = self.data_path + '/instance_' + str(instance)
        if not os.path.exists(instance_path):
            if not self.suppress_prints:
                print('Creating instance')
            os.mkdir(instance_path)
        return instance

    def save_hyperparams(self):
        with open(self.data_path + '/instance_' + str(self.instance) + '/hyperparams.json', "w") as fp:
            json.dump(self.hyperparams,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)
        
    def animate(self, name, frame_dur = 500):
        folder_name = 'instance_' + str(self.instance) + '/' + name + '/figures'
        gif_name = name + '.gif'
        gif_path = self.data_path + '/' + 'instance_' + str(self.instance) + '/' + name + '/' + gif_name
        if not os.path.exists(self.data_path + '/' + folder_name):
            if not self.suppress_prints:
                print('Images for animation do not exist')
        elif os.path.exists(gif_path):
            if not self.suppress_prints:
                print('Animation already exist!')
        else:
            files = os.listdir(self.data_path + '/' + folder_name)

            images = []
            for i in range(len(files))[::-1]:
                images.append(imageio.imread(self.data_path + '/' + folder_name + '/' + files[i]))

            imageio.mimsave(gif_path, images, duration = frame_dur, loop=0)
    
    def save_samples(self):
        full_path = self.data_path + '/instance_' + str(self.instance) + '/samples.csv'
        if type(self.samples) == list and self.samples == []:
            raise Exception('Samples data is empty!')    
        pd.DataFrame(self.samples).to_csv(full_path, index=False)
        if self.chain_data_inputted:
            chain_full_path = self.data_path + '/instance_' + str(self.instance) + '/chain_samples.csv'
            if type(self.chain_samples) == list and self.chain_samples == []:
                raise Exception('Samples data is empty!')    
            pd.DataFrame(self.chain_samples).to_csv(chain_full_path, index=False)
        
    def load_samples(self, chain = False):
        if chain:
            chain_full_path = self.data_path + '/instance_' + str(self.instance) + '/chain_samples.csv'
            if os.path.exists(chain_full_path):
                if not self.suppress_prints:
                    print('Loading Chain Samples...')
                return pd.read_csv(chain_full_path)
            else:
                raise Exception('Chain samples file does not exist!')
        else:
            full_path = self.data_path + '/instance_' + str(self.instance) + '/samples.csv'
            if os.path.exists(full_path):
                if not self.suppress_prints:
                    print('Loading Samples...')
                return pd.read_csv(full_path)
            else:
                raise Exception('Samples file does not exist!')

    def get_autocorrelations(self):
        autocorr_folder = self.data_path + '/instance_' + str(self.instance) + '/autocorrelations'
        if not os.path.exists(autocorr_folder):
            os.mkdir(autocorr_folder)

        for chain in range(self.num_chains):
            full_path = autocorr_folder + '/autocorrelations_' + str(chain + 1) + '.png'
            
            if self.num_chains == 1:
                title = 'MCMC autocorrelations'
            else:
                title = 'MCMC autocorrelations for chain ' + str(chain + 1)

            if os.path.exists(full_path):
                if not self.suppress_prints:
                    print('Traceplot ' + str(chain + 1) + ' already exists')
            else:
                ac = self.autocorr_fig(chain, title = title)
                ac.savefig(full_path)
    
    def autocorr_fig(self, chain_num = 1, title = ''):
        fig = plt.figure(figsize=(6,4))
        for param in self.samples.columns:
            autocorrelations = self.autocorrs['chain_' + str(chain_num + 1)][param]['Ct']
            tau = self.autocorrs['chain_' + str(chain_num + 1)][param]['tau']
            plt.plot(autocorrelations, label = param + ', tau = ' + str(tau))
        plt.legend()
        plt.xlabel('Sample number')
        plt.ylabel('Autocorrelation')
        plt.title(title + '\nDiscrete Autocorrelation')
        plt.close()

        return fig
    
    def calculate_autocorrs(self, D=-1):
        self.autocorrs = {}
        for chain_num in range(self.num_chains):
            if self.num_chains > 1:
                samples = self.chain_samples[self.chain_samples['chain'] == chain_num + 1].drop(columns = ['chain', 'sample_index'])
            else:
                samples = self.samples

            if D == -1:
                D = int(samples.shape[0])
            xp = np.atleast_2d(samples)
            z = (xp-np.mean(xp, axis=0))/np.std(xp, axis=0)
            Ct = np.ones((D, z.shape[1]))
            Ct[1:,:] = np.array([np.mean(z[i:]*z[:-i], axis=0) for i in range(1,D)])
            tau_hat = 1 + 2*np.cumsum(Ct, axis=0)
            Mrange = np.arange(len(tau_hat))
            tau = np.argmin(Mrange[:,None] - 5*tau_hat, axis=0)

            self.autocorrs['chain_' + str(chain_num+1)] = {}
            for i, param in enumerate(self.samples.columns):
                self.autocorrs['chain_' + str(chain_num+1)][param] = {}
                self.autocorrs['chain_' + str(chain_num+1)][param]['tau'] = tau[i]
                self.autocorrs['chain_' + str(chain_num+1)][param]['Ct'] = Ct[:,i]
        
        self.autocorrs['overall'] = {}
        for param in self.samples.columns:
            tau_overall = np.mean([self.autocorrs['chain_' + str(x + 1)][param]['tau'] for x in range(self.num_chains)])
            self.autocorrs['overall'][param] = {}
            self.autocorrs['overall'][param]['tau'] = tau_overall
    
    def get_ag_samples(self,samples, q_val):
        ags = pd.Series({},dtype='float64')
        for col in samples.columns:
            param_samples = samples[col]
            ag = np.quantile(param_samples, q_val)
            ags[col] = ag
        return ags
    
    def get_fields(self, chain_num):
        output = {}
        if self.fields.keys() != []:
            for key in self.fields.keys():
                field_output = self.fields[key][chain_num]
                if key == 'diverging':
                    field_output = sum(field_output.tolist())

                output[key] = field_output
        return output

    def get_summary(self):
        summary = {}
        full_path = self.data_path + '/instance_' + str(self.instance) + '/summary.json'
        if os.path.exists(full_path):
            f = open(full_path)
            summary = json.load(f)
            f.close()
        else:
            summary['RMSE'] = self.RMSE
            if self.chain_data_inputted:
                for chain_num in range(self.num_chains):
                    summary['chain_' + str(chain_num + 1)] = {}
                    samples = self.chain_samples[self.chain_samples['chain'] == chain_num + 1].drop(columns = ['chain', 'sample_index'])
                    params_lower = self.get_ag_samples(samples, 0.05)
                    params_mean = self.get_ag_samples(samples, 0.5)
                    params_upper = self.get_ag_samples(samples, 0.95)

                    summary['chain_' + str(chain_num + 1)]['fields'] = self.get_fields(chain_num)

                    for param in self.samples.columns:
                        summary['chain_' + str(chain_num + 1)][param] = {}
                        
                        summary['chain_' + str(chain_num + 1)][param]['lower'] = params_lower[param]
                        summary['chain_' + str(chain_num + 1)][param]['mean'] = params_mean[param]
                        summary['chain_' + str(chain_num + 1)][param]['upper'] = params_upper[param]
                        
                        summary['chain_' + str(chain_num + 1)][param]['tau'] = self.autocorrs['chain_' + str(chain_num + 1)][param]['tau']
                        
                        if self.actual_values[param] !='NaN':
                            proposed = params_mean[param]
                            actual = self.actual_values[param]
                            summary['chain_' + str(chain_num + 1)][param]['param_accuracy'] = 200*np.abs(proposed-actual)/(proposed + actual)

                
            overall_samples = self.samples
            summary['overall'] = {}

            overall_params_lower = self.get_ag_samples(overall_samples, 0.05)
            overall_params_mean = self.get_ag_samples(overall_samples, 0.5)
            overall_params_upper = self.get_ag_samples(overall_samples, 0.95)

            summary['chain_' + str(chain_num + 1)]['fields'] = self.get_fields(0)

            for param in self.samples.columns:
                summary['overall'][param] = {}

                summary['overall'][param]['lower'] = overall_params_lower[param]
                summary['overall'][param]['mean'] = overall_params_mean[param]
                summary['overall'][param]['upper'] = overall_params_upper[param]

                if self.chain_data_inputted:
                    summary['overall'][param]['tau'] = self.autocorrs['overall'][param]['tau']
                else:
                    summary['overall'][param]['tau'] = self.autocorrs['chain_' + str(1)][param]['tau']

                if self.actual_values[param] !='NaN':
                    summary['overall'][param]['param_accuracy'] = np.abs(overall_params_mean[param]-self.actual_values[param])/np.mean([overall_params_mean[param],self.actual_values[param]])*100


            with open(self.data_path + '/instance_' + str(self.instance) + '/summary.json', "w") as fp:
                json.dump(summary,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

        return summary