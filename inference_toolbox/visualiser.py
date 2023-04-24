import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import os
import json
from numpyencoder import NumpyEncoder
import imageio
from matplotlib.gridspec import GridSpec

class Visualiser:
    def __init__(self, test_data, params_all, model, hyperparams, acceptance_rate):
        self.test_data = test_data
        self.params_lower = params_all[0]
        self.params_mean = params_all[1]
        self.params_upper = params_all[2]

        self.model = model
        self.model_func = model.get_model()
        self.hyperparams = hyperparams
        self.acceptance_rate = acceptance_rate
        self.data_path = 'results/inference'

    def get_traceplot(self, samples, instance, save = False):
        # Format samples and construct traceplots
        if type(samples) == list and samples == []:
            print('Data already exists!!')
        else:
            params_samples_formatted = samples.copy()
            for col in params_samples_formatted:
                params_samples_formatted[col] = params_samples_formatted[col].apply(lambda x: x.val)
            tp = self.traceplots(np.array(params_samples_formatted), xnames = params_samples_formatted.columns, title = 'MCMC samples')
            if save:
                full_path = self.data_path + '/instance_' + str(instance) + '/traceplot.png'
                tp.savefig(full_path)
            else:
                tp.show()

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
            ax_hist.hist(x[:,i], orientation='horizontal', bins=50)
            plt.setp(ax_hist.get_xticklabels(), visible=False)
            plt.setp(ax_hist.get_yticklabels(), visible=False)
            xlim = ax_hist.get_xlim()
            ax_hist.set_xlim([xlim[0], 1.1*xlim[1]])
        return fig

    def visualise_results(self, domain, three_D = True, include_test_points = True, save = True):
        if type(self.params_mean) == list and self.params_mean == []:
            print('Figures not generated! Data already exists!')
            return -1

        elif three_D:
            X, Y, Z = np.meshgrid(domain[:,0], domain[:,1], domain[:,2])
            C_lower = self.model_func(self.params_lower, X,Y,Z)
            C_mean = self.model_func(self.params_mean, X,Y,Z)
            C_upper = self.model_func(self.params_upper, X,Y,Z)

            return self.threeD_plots(X.flatten(), Y.flatten(), Z.flatten(), C_lower.flatten(), C_mean.flatten(), C_upper.flatten(), include_test_points = include_test_points, save = save)

    def threeD_plots(self, X, Y, Z, C_lower, C_mean, C_upper, q=10, include_test_points = True, save = False, log_results = True):

        # Modifying C depending on if the results need logging
        if log_results:
            C_lower[C_lower<1] = 1
            C_lower = np.log10(C_lower)
            
            C_mean[C_mean<1] = 1
            C_mean = np.log10(C_mean)
            
            C_upper[C_upper<1] = 1
            C_upper = np.log10(C_upper)
            
            prefix = 'Log c'
        else:
            prefix = 'C'
        
        # Define the bin numbers and their labels
        lower_bin_nums = pd.qcut(C_lower,q, labels = False, duplicates='drop')
        lower_bin_labs = np.array(pd.qcut(C_mean,q, duplicates='drop').unique())

        mean_bin_nums = pd.qcut(C_mean,q, labels = False, duplicates='drop')
        mean_bin_labs = np.array(pd.qcut(C_mean,q, duplicates='drop').unique())

        upper_bin_nums = pd.qcut(C_upper,q, labels = False, duplicates='drop')
        upper_bin_labs = np.array(pd.qcut(C_mean,q, duplicates='drop').unique())

        # Define the dataframe of results with bin numbers attached
        lower_conc_and_bins = pd.DataFrame([X.reshape(lower_bin_nums.shape), Y.reshape(lower_bin_nums.shape), Z.reshape(lower_bin_nums.shape), C_lower, lower_bin_nums]).T
        lower_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        mean_conc_and_bins = pd.DataFrame([X.reshape(mean_bin_nums.shape), Y.reshape(mean_bin_nums.shape), Z.reshape(mean_bin_nums.shape), C_mean, mean_bin_nums]).T
        mean_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        upper_conc_and_bins = pd.DataFrame([X.reshape(upper_bin_nums.shape), Y.reshape(upper_bin_nums.shape), Z.reshape(upper_bin_nums.shape), C_upper, upper_bin_nums]).T
        upper_conc_and_bins.columns=['x', 'y', 'z', 'conc', 'bin']

        # Define min and max values for colorbar
        min_val = np.percentile([C_lower, C_mean, C_upper],5)
        max_val = np.percentile([C_lower, C_mean, C_upper],95)

        # I NEED TO EDIT THIS TO IT CHANGES FOR EACH PLOT!!!!!!
        # Calculates the percentage differances and RMSE if the test points are set to be included
        if include_test_points:
            lower_test_pred_C = self.model_func(self.params_lower, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            lower_test_actual_C = self.test_data['Concentration']
            lower_percentage_difference = 2*np.abs(lower_test_actual_C-lower_test_pred_C)/(lower_test_actual_C + lower_test_pred_C) * 100

            mean_test_pred_C = self.model_func(self.params_mean, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            mean_test_actual_C = self.test_data['Concentration']
            mean_percentage_difference = 2*np.abs(mean_test_actual_C-mean_test_pred_C)/(mean_test_actual_C + mean_test_pred_C) * 100

            upper_test_pred_C = self.model_func(self.params_upper, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            upper_test_actual_C = self.test_data['Concentration']
            upper_percentage_difference = 2*np.abs(lower_test_actual_C-upper_test_pred_C)/(upper_test_actual_C + upper_test_pred_C) * 100

            RMSE = np.sqrt(np.mean((mean_test_pred_C-mean_test_actual_C)**2))
        else:
            RMSE = 'n/a'     
            
        # Creates a directory for the instance if the save parameter is selected
        if save:
            instance = self.get_instance()
            os.mkdir(self.data_path + '/instance_' + str(instance))
            self.save_hyperparams(instance)
            os.mkdir(self.data_path + '/instance_' + str(instance) + '/figures')
        
        # Loops through each bin number and generates a figure with the bin data 
        for bin_num in np.sort(np.unique(mean_bin_nums)):

            lower_bin_data = lower_conc_and_bins[lower_conc_and_bins['bin'] >= bin_num]
            mean_bin_data = mean_conc_and_bins[mean_conc_and_bins['bin'] >= bin_num]
            upper_bin_data = upper_conc_and_bins[upper_conc_and_bins['bin'] >= bin_num]

            fig = plt.figure(figsize = (20,10), layout = 'constrained')
            gs = GridSpec(10, 6, figure=fig)
            
            # Defines the lower bound subplot
            ax1 = fig.add_subplot(gs[:-2,:2], projection = '3d')
            plot_1 = ax1.scatter(lower_bin_data.x, lower_bin_data.y, lower_bin_data.z, c = lower_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax1.set_title(prefix + 'oncentration generated \nby the lower bound parameters')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_xlim(np.min(X), np.max(X))
            ax1.set_ylim(np.min(Y), np.max(Y))
            ax1.set_zlim(np.min(Z), np.max(Z))

            # Defines the mean subplot
            ax2 = fig.add_subplot(gs[:-2,2:4], projection = '3d')
            ax2.scatter(mean_bin_data.x, mean_bin_data.y, mean_bin_data.z, c = mean_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax2.set_title(prefix + 'oncentration generated \nby the mean parameters')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax2.set_xlim(np.min(X), np.max(X))
            ax2.set_ylim(np.min(Y), np.max(Y))
            ax2.set_zlim(np.min(Z), np.max(Z))

            # Defines the upper bound subplot
            ax3 = fig.add_subplot(gs[:-2,4:], projection = '3d')
            ax3.scatter(upper_bin_data.x, upper_bin_data.y, upper_bin_data.z, c = upper_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
            ax3.set_title(prefix + 'oncentration generated \nby the upper bound parameters')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')
            ax3.set_xlim(np.min(X), np.max(X))
            ax3.set_ylim(np.min(Y), np.max(Y))
            ax3.set_zlim(np.min(Z), np.max(Z))

            # Generates the test point data on each graph
            if include_test_points:
                pd_min = np.min([lower_percentage_difference, mean_percentage_difference, upper_percentage_difference])
                pd_max = np.min([lower_percentage_difference, mean_percentage_difference, upper_percentage_difference])

                plot_2 = ax1.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 10*np.log10(lower_test_pred_C), c = lower_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                ax2.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 10*np.log10(mean_test_pred_C), c = mean_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                ax3.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 10*np.log10(upper_test_pred_C), c = upper_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                sampling_data_string = 'RMSE = ' + "{:.3e}".format(RMSE) + '\nAcceptance Rate = ' + "{:.2f}".format(self.acceptance_rate) + '%' 
            else:
                sampling_data_string = 'RMSE = n/a\nAcceptance Rate = ' + "{:.2f}".format(self.acceptance_rate) + '%' 

            # Creates an axis to display the parameter values
            ax4 = fig.add_subplot(gs[-2:,:3])
            param_string = self.get_param_string()
            ax4.text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
            ax4.set_xticks([])
            ax4.set_yticks([])

            # Creates an axis to display the sampling information
            ax5 = fig.add_subplot(gs[-2:,3:])
            ax5.text(0.5,0.5, sampling_data_string, fontsize = 30, va = "center", ha = 'center')
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
            fig.suptitle('Values for mean plot greater than ' + "{:.2f}".format(left_bound), fontsize = 32)

            # Saves the figures if required
            if save:
                fig_name = 'fig_' + str(bin_num + 1) + '_of_' + str(np.max(mean_bin_nums + 1)) + '.png'
                full_path = self.data_path + '/instance_' + str(instance) + '/figures/' + fig_name
                if not os.path.exists(full_path):
                    fig.savefig(full_path)
            else:
                plt.show()
        
        # Returns the instance number
        if save:
            return instance
        else:
            return -1

    def get_param_string(self):
        param_string_array = []
        for param in self.params_mean.index:
            if np.floor(np.log10(self.params_mean[param].val)) < 2:
                lower_string =  "{:.2f}".format(self.params_lower[param].val)
                mean_string = "{:.2f}".format(self.params_mean[param].val)
                upper_string = "{:.2f}".format(self.params_upper[param].val)
            else:
                lower_string =  "{:.2e}".format(self.params_lower[param].val)
                mean_string = "{:.2e}".format(self.params_mean[param].val)
                upper_string = "{:.2e}".format(self.params_upper[param].val)
            param_string_array.append(param + ' = [' + lower_string + ', ' + mean_string + ', ' + upper_string + ']')

        return ('\n').join(param_string_array)
        
    def get_instance(self):
        instance_folders = os.listdir(self.data_path)
        instances = [int(x.split('_')[1]) for x in instance_folders]
        missing_elements = []
        if len(instances) == 0:
            return 1
        for el in range(1,np.max(instances) + 2):
            if el not in instances:
                missing_elements.append(el)
        instance = np.min(missing_elements)
        return instance

    def save_hyperparams(self, instance):
        with open(self.data_path + '/instance_' + str(instance) + '/hyperparams.json', "w") as fp:
            json.dump(self.hyperparams,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)
        
    def animate(self, instance, frame_dur = 1):
        if instance == -1:
            print('Animation not generated! Data already exists!')
        else:
            folder_name = 'instance_' + str(instance)
            files = os.listdir(self.data_path + '/' + folder_name + '/figures')

            images = []
            for i in range(len(files))[::-1]:
                images.append(imageio.imread(self.data_path + '/' + folder_name + '/figures/' + files[i]))
            gif_name = folder_name + '.gif'
            gif_path = self.data_path + '/' + folder_name + '/' + gif_name
            if os.path.exists(gif_path):
                os.remove(gif_path)

            imageio.mimsave(gif_path, images, duration = frame_dur)
