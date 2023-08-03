import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio
import json
from numpyencoder import NumpyEncoder
from fractions import Fraction

# Visualiser class - generates the plots from the predictions of the trained gaussian processor
class Visualiser:
    # Initialises the Visualiser class saving all relevant variables and performing some initialising tasks
    def __init__(self, 
                 test_data, 
                 trainer, 
                 kernel_type,
                 data_path= 'results/GP_results/simulated_data',
                 include_test_points = True,
                 suppress_print = False):

        # Save variables
        self.suppress_print = suppress_print
        self.test_data = test_data
        self.data_path = data_path
        self.include_test_points = include_test_points
        self.kernel = kernel_type
        self.results_path = self.data_path + '/' + self.kernel
        self.model = trainer.model

        self.kernel_dict = {'matern_white': 'Matern and White Noise', 'rbf': 'Radial Basis Function', 'matern':'Matern'}

        # Filters the relevant fitted gaussian processor parameters
        params = trainer.params
        params = self.remove_keys_containing_substring(params, 'bounds')
        self.params = {key: value for key, value in params.items() if not ('_' not in key and 'k' in key)}

        # Creates folders if they don't exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    # Function for filtering gaussian processor parameters
    def remove_keys_containing_substring(self, dictionary, substring):
        return {key: value for key, value in dictionary.items() if substring not in key}

    # Formats the gaussian processor fitted parameters to be displayed on the figure
    def get_GP_params_string(self):
        params_string_array = []
        for key in self.params.keys():
            formatter = "{:.2e}" 
            if  np.floor(np.log10(self.params[key])) < 2 or np.floor(np.log10(self.params[key])) > -2: formatter = "{:.2f}" 
            one_param_string = key  + ' = ' + formatter.format(self.params[key])
            params_string_array.append(one_param_string)
        return ('\n').join(params_string_array)
    
    # Formats the RMSE to be displayed on the figure
    def get_RMSE_string(self):
        formatter = "{:.2e}" 
        if  np.floor(np.log10(self.RMSE)) < 2: formatter = "{:.2f}" 
        return 'RMSE = ' + formatter.format(self.RMSE)

    # Outputs the plots for the predicted values from the gaussian processor, based on an inputted domain
    # There are multiple ways of visualising these results
    def visualise_results(self, domain, name, plot_type = '3D', title = 'Concentration of Droplets'):
        # Generates domain plots
        points = domain.create_domain()

        # Checks whether plots exist
        if os.path.exists(self.results_path + '/' + name):
            if not self.suppress_print:
                print('Plots already exist!')

        # Output plots are 3D
        elif plot_type == '3D':
            X, Y, Z = points[:,0], points[:,1], points[:,2]
            grid_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

            if self.model == None:
                raise Exception('Model has not been trained!')
            C_mean, sigma_pred = self.model.predict(grid_points, return_std=True)
            C_lower = C_mean - 2*sigma_pred
            C_upper = C_mean + 2*sigma_pred

            results_df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten(), 'C_lower': C_lower, 'C_mean': C_mean,'C_upper': C_upper})
            
            self.threeD_plots(results_df, name,  title=title)

    # Plotting function for 3D plots
    def threeD_plots(self, results, name, q=10, title = 'Concentration of Droplets'):
        X = results.x
        Y = results.y
        Z = results.z
        C_lower = results.C_lower
        C_mean = results.C_mean
        C_upper = results.C_upper

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

        # Define min and max values for lower colorbar
        min_val_lower = np.percentile(C_lower,10)
        max_val_lower = np.percentile(C_lower,90)

        # Define min and max values for mean colorbar
        min_val_mean = np.percentile(C_mean,10)
        max_val_mean = np.percentile(C_mean,90)

        # Define min and max values for upper colorbar
        min_val_upper = np.percentile(C_upper,10)
        max_val_upper = np.percentile(C_upper,90)

        # Calculates the percentage differances and RMSE if the test points are set to be included
        if self.include_test_points:
            test_grid_points = np.column_stack((self.test_data.x,self.test_data.y, self.test_data.z))

            test_pred_C, test_sigma = self.model.predict(test_grid_points, return_std=True)

            lower_test_pred_C = test_pred_C - 2*test_sigma
            lower_test_actual_C = self.test_data['Concentration']
            lower_percentage_difference = 2*np.abs(lower_test_actual_C-lower_test_pred_C)/(lower_test_actual_C + lower_test_pred_C) * 100

            mean_test_pred_C = test_pred_C
            mean_test_actual_C = self.test_data['Concentration']
            mean_percentage_difference = 2*np.abs(mean_test_actual_C-mean_test_pred_C)/(mean_test_actual_C + mean_test_pred_C) * 100

            upper_test_pred_C = test_pred_C + 2*test_sigma
            upper_test_actual_C = self.test_data['Concentration']
            upper_percentage_difference = 2*np.abs(lower_test_actual_C-upper_test_pred_C)/(upper_test_actual_C + upper_test_pred_C) * 100
            
            self.RMSE = np.mean((mean_test_pred_C-mean_test_actual_C)**2)

        # Creates a directory for the instance if the save parameter is selected
        os.mkdir(self.results_path + '/' + str(name))
        os.mkdir(self.results_path + '/' + str(name) + '/figures')
        
        # Loops through each bin number and generates a figure with the bin data 
        for bin_num in np.sort(np.unique(mean_bin_nums)):

            lower_bin_data = lower_conc_and_bins[lower_conc_and_bins['bin'] >= bin_num]
            mean_bin_data = mean_conc_and_bins[mean_conc_and_bins['bin'] >= bin_num]
            upper_bin_data = upper_conc_and_bins[upper_conc_and_bins['bin'] >= bin_num]

            fin_alpha = len(self.params)*0.75

            y = 12
            r = 0.8
            alpha = (1-r)*y

            delta_alpha = fin_alpha - alpha

            y_prime = y + delta_alpha
            r_prime = r*y/y_prime

            r_prime_frac = Fraction(r_prime)

            fig = plt.figure(constrained_layout = True, figsize = (24, y_prime))
            spec = GridSpec(2, 6, figure = fig, height_ratios= [r_prime_frac.numerator, r_prime_frac.denominator - r_prime_frac.numerator])

            ax1 = fig.add_subplot(spec[0,:2], projection='3d')
            ax2 = fig.add_subplot(spec[0,2:4], projection='3d')
            ax3 = fig.add_subplot(spec[0,4:], projection='3d')
            ax4 = fig.add_subplot(spec[1,:3])
            ax5 = fig.add_subplot(spec[1,3:])
            
            # Defines the lower bound subplot
            plot_1 = ax1.scatter(lower_bin_data.x, lower_bin_data.y, lower_bin_data.z, c = lower_bin_data.conc, cmap='jet', vmin=min_val_lower, vmax=max_val_lower, alpha = 0.3, s=1)
            ax1.set_title('Gaussian Process Upper Bound Predictions', fontsize = 20)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_xlim(np.min(X), np.max(X))
            ax1.set_ylim(np.min(Y), np.max(Y))
            ax1.set_zlim(np.min(Z), np.max(Z))

            # Defines the mean subplot
            plot_2 = ax2.scatter(mean_bin_data.x, mean_bin_data.y, mean_bin_data.z, c = mean_bin_data.conc, cmap='jet', vmin=min_val_mean, vmax=max_val_mean, alpha = 0.3, s=1)
            ax2.set_title('Gaussian Process Mean Predictions', fontsize = 20)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax2.set_xlim(np.min(X), np.max(X))
            ax2.set_ylim(np.min(Y), np.max(Y))
            ax2.set_zlim(np.min(Z), np.max(Z))

            # Defines the upper bound subplot
            plot_3 = ax3.scatter(upper_bin_data.x, upper_bin_data.y, upper_bin_data.z, c = upper_bin_data.conc, cmap='jet', vmin=min_val_upper, vmax=max_val_upper, alpha = 0.3, s=1)
            ax3.set_title('Gaussian Process Lower Bound Predictions', fontsize = 20)
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

                plot_4 = ax1.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = lower_test_pred_C, c = lower_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                ax2.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = mean_test_pred_C, c = mean_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                ax3.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = upper_test_pred_C, c = upper_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                
                RMSE_string = self.get_RMSE_string()

            else:
                RMSE_string = 'RMSE = n/a'

            # Creates an axis to display the parameter values
            param_string = self.get_GP_params_string()
            ax4.text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
            ax4.set_xticks([])
            ax4.set_yticks([])

            # Creates an axis to display the sampling information
            ax5.text(0.5,0.5, RMSE_string, fontsize = 30, va = "center", ha = 'center')
            ax5.set_xticks([])
            ax5.set_yticks([])

            # Defines the colorbars
            cbar1 = plt.colorbar(plot_1, ax = ax1, location = 'bottom', shrink = 1)
            cbar2 = plt.colorbar(plot_2, ax = ax2, location = 'bottom', shrink = 1)
            cbar3 = plt.colorbar(plot_3, ax = ax3, location = 'bottom', shrink = 1)
            cbar4 = plt.colorbar(plot_4, ax = np.array([ax4,ax5]), location = 'top', fraction = 0.8)
            cbar1.ax.set_title('Lower Bound Predicted Concentration', fontsize = 10)
            cbar2.ax.set_title('Mean Predicted Concentration', fontsize = 10)
            cbar3.ax.set_title('Upper Bound Predicted Concentration', fontsize = 10)
            cbar4.ax.set_title('Percemtage Difference in Test Data', fontsize = 10)

            # Defines the overall title, including the range of values for each plot
            if mean_bin_labs[bin_num].left < 0:
                left_bound = 0
            else:
                left_bound = mean_bin_labs[bin_num].left
            fig.suptitle(title + '\nValues for mean plot greater than ' + "{:.2f}".format(left_bound) + '\n', fontsize = 32)

            # Saves the figures if required
            fig_name = 'fig_' + str(bin_num + 1) + '_of_' + str(np.max(mean_bin_nums + 1)) + '.png'
            full_path = self.results_path + '/' + name + '/figures/' + fig_name
            if not os.path.exists(full_path):
                fig.savefig(full_path)
            plt.close()

    # Gathers all of the figures under the inputted name, creates an animation of them and saves that animation
    def animate(self, name, frame_dur = 500):
        folder_name = self.results_path + '/' + name + '/figures'
        gif_name = name + '.gif'
        gif_path = self.results_path + '/' + name + '/' + gif_name
        if not os.path.exists(folder_name):
            if not self.suppress_prints:
                print('Images for animation do not exist')
        elif os.path.exists(gif_path):
            if not self.suppress_prints:
                print('Animation already exist!')
        else:
            files = os.listdir(folder_name)

            images = []
            for i in range(len(files))[::-1]:
                images.append(imageio.imread(folder_name + '/' + files[i]))

            imageio.mimsave(gif_path, images, duration = frame_dur, loop=0)
 