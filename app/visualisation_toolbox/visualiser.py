import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
from numpyencoder import NumpyEncoder
import imageio.v2 as imageio
from matplotlib.gridspec import GridSpec
from fractions import Fraction
from labellines import labelLines
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from bs4 import BeautifulSoup
import base64
from sympy.printing.latex import latex
from sympy import Eq, symbols
import jax.numpy as jnp
import io
from abc import abstractmethod

from regression_toolbox.sampler import Sampler
from gaussian_process_toolbox.gaussian_processor import GP
from visualisation_toolbox.domain import Domain
from data_processing.sim_data_processor import SimDataProcessor
from data_processing.raw_data_processor import RawDataProcessor

class Visualiser:
    """
    Visualiser class - Used for processing and visualising the results from the sampler. The class contains methods for generating traceplots of the samples, visualising the modelled system and summarising the results and success of the sampler.

    Attributes:
        - data_processor (SimDataProcessor or RawDataProcessor): Data processor object
        - training_data (pd.DataFrame): Training data
        - testing_data (pd.DataFrame): Testing data
        - dependent_variable (str): Dependent variable of the model
        - independent_variables (list): Independent variables of the model
        - results_path (str): Path to save results
        - instance (int): Instance number
        - RMSE (float): Root Mean Squared Error
        - AIC (float): Akaike Information Criterion
        - BIC (float): Bayesian Information Criterion
    """

    def __init__(self, input_obj: Sampler | GP):
        """
        Initialises the Visualiser class saving all relevant variables and performing some initialising tasks

        Args:
            - input (Sampler | GP): The sampler or GP object

        """
        if isinstance(input_obj, Sampler):
            self.sampler = input_obj
        elif isinstance(input_obj, GP):
            self.gp_obj = input_obj

        self.data_processor = input_obj.data_processor
        
        self.training_data = input_obj.training_data
        self.testing_data = input_obj.testing_data
        
        self.dependent_variable = input_obj.dependent_variable
        self.independent_variables = input_obj.independent_variables
        
        self.results_path = input_obj.results_path
        self.instance = input_obj.instance

        # log_likelihood = self.likelihood_func(test_predictions, self.params_median).log_prob(jnp.array(test_measured)).sum()
        
        # self.RMSE = np.sqrt(np.mean((test_predictions-test_measured)**2))
        # self.AIC = 2*self.params_median.size - log_likelihood
        # self.BIC = self.params_median.size*np.log(self.testing_data.shape[0]) - 2*log_likelihood

        # self.autocorrs = self._calculate_autocorrs()

        self.construction = input_obj.get_construction()

        # self.prediction_plots = []

    @abstractmethod
    def _make_predictions(self, domain_points):
        pass

    def show_predictions(self, domain: Domain, plot_name: str, plot_type: str = '3D', title: str = None, cross_section_params = None):
        """
        Outputs the plots for visualising the modelled system based on the concluded lower, median and upper bound parameters and an inputted domain. Plots are saved and include a summary of the parameter values and the sampling success.

        Args:
            - domain (Domain): Domain object, must be built
            - plot_name (str): Name of the plot for saving purposes
            - plot_type (str, optional): Type of plot. Defaults to '3D'. Options are: 
                - '1D': 1D plot
                - '2D': 2D plot
                - '3D': 3D plot
                - '2D_cross_sections': 2D cross sections plot

            - title (str, optional): Overall title of the plot. Defaults to None.
            - cross_section_params (dict, optional): Dictionary containing the cross section parameters. Defaults to None.

        """

        if domain.points is None:
            raise Exception('Visualiser - domain has not been built, run domain.build_domain() outside of the visualiser!')
        
        domain_points = domain.points

        if not all(self.independent_variables == domain_points.columns):
            raise Exception('Visualiser - domain columns do not match the independent variables!')

        if plot_type == '3D':
            self._show_3D_predictions(domain_points, plot_name, title)
        elif plot_type == '2D':
            self._show_2D_predictions(domain_points, plot_name, title)
        elif plot_type == '1D':
            self._show_1D_predictions(domain_points, plot_name, title)
        elif plot_type == '2D_cross_sections':
            self._show_2D_cross_sections_predictions(domain, plot_name, title, cross_section_params)
        elif plot_type == '3D_fixed':
            self._show_3D_fixed_predictions(domain_points, plot_name, title)
    
    def _construct_figure(self, project_3D = False):
        """
        Constructs the figure for the visualisation
        """

        if isinstance(self, RegressionVisualiser):
            fin_alpha = len(self.params_median)*0.75
        else:
            fin_alpha = 3

        y = 12
        r = 0.8
        alpha = (1-r)*y

        delta_alpha = fin_alpha - alpha

        y_prime = y + delta_alpha
        r_prime = r*y/y_prime

        r_prime_frac = Fraction(r_prime)

        fig = plt.figure(constrained_layout = True, figsize = (24,y_prime))
        spec = GridSpec(2, 6, figure = fig, height_ratios= [r_prime_frac.numerator, r_prime_frac.denominator - r_prime_frac.numerator])
        axes = [None]*5
        
        projection = None
        if project_3D:
            projection = '3d'
        
        axes[0] = fig.add_subplot(spec[0,:2], projection = projection)
        axes[1] = fig.add_subplot(spec[0,2:4], projection = projection)
        axes[2] = fig.add_subplot(spec[0,4:], projection = projection)
        axes[3] = fig.add_subplot(spec[1,:3])
        axes[4] = fig.add_subplot(spec[1,3:])

        return fig, axes

    def _fill_figure_metadata(self, fig, axes):
        RMSE_formatter = "{:.2e}" 
        AIC_formatter = "{:.2e}"
        BIC_formatter = "{:.2e}"

        if self.RMSE < 0:
            raise Exception('Visualiser - RMSE is negative!')
        if  np.floor(np.log10(self.RMSE)) < 2 and np.floor(np.log10(self.RMSE)) > -2: RMSE_formatter = "{:.2f}"
        if  np.floor(np.log10(np.abs(self.AIC))) < 2 and np.floor(np.log10(np.abs(self.AIC))) > -2: AIC_formatter = "{:.2f}"
        if  np.floor(np.log10(np.abs(self.BIC))) < 2 and np.floor(np.log10(np.abs(self.BIC))) > -2: BIC_formatter = "{:.2f}"
        RMSE_string = 'RMSE = ' + RMSE_formatter.format(self.RMSE)
        AIC_string = 'AIC = ' + AIC_formatter.format(self.AIC)
        BIC_string = 'BIC = ' + BIC_formatter.format(self.BIC)

        if isinstance(self, RegressionVisualiser):
            # Creates an axis to display the parameter values
            param_string = self._get_param_string()
            axes[3].text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
            
            param_accuracy_string = self._get_param_accuracy_string()

            # Creates an axis to display the sampling information
            axes[4].text(0.5,0.5, RMSE_string + ',   ' + AIC_string + ',   ' + BIC_string + '\n' + param_accuracy_string, fontsize = 30, va = "center", ha = 'center')
        
        elif isinstance(self, GPVisualiser):
            # Creates an axis to display the parameter values
            param_string = self._get_param_string()
            print(param_string)
            axes[3].text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')

            # Creates an axis to display the success information
            axes[4].text(0.5,0.5, RMSE_string + ',   ' + AIC_string + ',   ' + BIC_string, fontsize = 30, va = "center", ha = 'center')

        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[4].set_xticks([])
        axes[4].set_yticks([])

        return fig, axes

    def _show_1D_predictions(self, domain_points, plot_name, title):

        dir_name = self.results_path + '/instance_' + str(self.instance) + '/' + plot_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        full_path = dir_name + '/predictions.png'
        if not os.path.exists(full_path):

            lower_pred, median_pred, upper_pred = self._make_predictions(domain_points)

            results_df = domain_points.copy()
            results_df['lower_res'] = lower_pred
            results_df['mean_res'] = median_pred
            results_df['upper_res'] = upper_pred
            
            fig, axes = self._construct_figure()

            if isinstance(self, RegressionVisualiser):
                title_lower = 'Generated by the lower bound parameters'
                title_median = 'Generated by the median parameters'
                title_upper = 'Generated by the upper bound parameters'
            else:
                title_lower = 'Lower bound GP prediction'
                title_median = 'Median GP prediction'
                title_upper = 'Upper bound GP prediction'

            # Defines the lower bound subplot
            axes[0].plot(results_df[self.independent_variables[0]], lower_pred, label='Predicted ' + self.dependent_variable[0])
            axes[0].set_title(title_lower, fontsize = 20)
            axes[0].set_xlabel(self.independent_variables[0])
            axes[0].set_ylabel(self.dependent_variable)

            # Defines the median subplot
            axes[1].plot(results_df[self.independent_variables[0]], median_pred, label='Predicted ' + self.dependent_variable)
            axes[1].set_title(title_median, fontsize = 20)
            axes[1].set_xlabel(self.independent_variables[0])
            axes[1].set_ylabel(self.dependent_variable)
            # Fill the area between the lower and upper bound predictions
            axes[1].fill_between(results_df[self.independent_variables[0]], lower_pred, upper_pred, color='gray', alpha=0.3, label='Prediction Interval')

            # Defines the upper subplot
            axes[2].plot(results_df[self.independent_variables[0]], upper_pred, label='Predicted ' + self.dependent_variable)
            axes[2].set_title(title_upper, fontsize = 20)
            axes[2].set_xlabel(self.independent_variables[0])
            axes[2].set_ylabel(self.dependent_variable)

            axes[0].scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.dependent_variable], s = 20, c = 'r', label='Measured ' + self.dependent_variable + ' test values')
            axes[1].scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.dependent_variable], s = 20, c = 'r', label='Measured ' + self.dependent_variable + ' test values')
            axes[2].scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.dependent_variable], s = 20, c = 'r', label='Measured ' + self.dependent_variable + ' test values')
            
            # Get axis limits from the median plot
            ylim = axes[1].get_ylim()

            # Apply the same limits to all plots
            for ax in axes:
                ax.set_ylim(ylim)

            axes[0].legend()
            axes[1].legend()
            axes[2].legend()

            fig.suptitle(title, fontsize = 40)

            fig, axes = self._fill_figure_metadata(fig, axes)

            fig.savefig(full_path)
            plt.close()

    def _show_2D_predictions(self, domain_points, plot_name, title):
        dir_name = self.results_path + '/instance_' + str(self.instance) + '/' + plot_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        full_path = dir_name + '/predictions.png'
        if not os.path.exists(full_path):

            lower_pred, median_pred, upper_pred = self._make_predictions(domain_points)

            results_df = domain_points.copy()
            results_df['lower_res'] = lower_pred
            results_df['mean_res'] = median_pred
            results_df['upper_res'] = upper_pred
            
            fig, axes = self._construct_figure()

            # Define min and max values for colorbar
            min_val = np.percentile([lower_pred, median_pred, upper_pred], 10)
            max_val = np.percentile([lower_pred, median_pred, upper_pred], 90)

            X = results_df[self.independent_variables[0]]
            Y = results_df[self.independent_variables[1]]

            if isinstance(self, RegressionVisualiser):
                title_lower = 'Generated by the lower bound parameters'
                title_median = 'Generated by the median parameters'
                title_upper = 'Generated by the upper bound parameters'
                
                lower_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_lower, self.testing_data))
                median_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_median, self.testing_data))
                upper_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_upper, self.testing_data))

                # Generates the test point data on each graph
                plot_2 = axes[0].scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.independent_variables[1]], s = 20, c = lower_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                axes[1].scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.independent_variables[1]], s = 20, c = median_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                axes[2].scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.independent_variables[1]], s = 20, c = upper_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                cbar2 = plt.colorbar(plot_2, ax=axes[4], location='top', shrink=2)
                cbar2.ax.set_title('Error in Predictions (Modeled - Measured)', fontsize=10)
            else:
                title_lower = 'Lower bound GP prediction'
                title_median = 'Median GP prediction'
                title_upper = 'Upper bound GP prediction'


            # Defines the lower bound subplot
            plot_1 = axes[0].tricontourf(X, Y, lower_pred, vmin=min_val, vmax=max_val, levels=100, cmap = 'jet', alpha = 0.7)
            axes[0].set_title(title_lower, fontsize=20)
            axes[0].set_xlabel(self.independent_variables[0])
            axes[0].set_ylabel(self.independent_variables[1])
            axes[0].set_xlim(np.min(X), np.max(X))
            axes[0].set_ylim(np.min(Y), np.max(Y))

            # Defines the mean subplot
            axes[1].tricontourf(X, Y, median_pred, vmin=min_val, vmax=max_val, levels=100, cmap = 'jet', alpha = 0.7)
            axes[1].set_title(title_median, fontsize=20)
            axes[1].set_xlabel(self.independent_variables[0])
            axes[1].set_ylabel(self.independent_variables[1])
            axes[1].set_xlim(np.min(X), np.max(X))
            axes[1].set_ylim(np.min(Y), np.max(Y))

            # Defines the upper bound subplot
            axes[2].tricontourf(X, Y, upper_pred, vmin=min_val, vmax=max_val, levels=100, cmap = 'jet', alpha = 0.7)
            axes[2].set_title(title_upper, fontsize=20)
            axes[2].set_xlabel(self.independent_variables[0])
            axes[2].set_ylabel(self.independent_variables[1])
            axes[2].set_xlim(np.min(X), np.max(X))
            axes[2].set_ylim(np.min(Y), np.max(Y))

            # Defines the two colorbars
            cbar1 = plt.colorbar(plot_1, ax=axes[3], location='top', shrink=2)
            cbar1.ax.set_title('Predicted ' + self.dependent_variable, fontsize=10)

            fig.suptitle(title, fontsize = 40)

            fig, axes = self._fill_figure_metadata(fig, axes)

            fig.savefig(full_path)
            plt.close()

    def _show_3D_predictions(self, domain_points, plot_name, title):
        dir_name = self.results_path + '/instance_' + str(self.instance) + '/' + plot_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        full_path = dir_name + '/predictions.gif'
        if not os.path.exists(full_path):

            lower_pred, median_pred, upper_pred = self._make_predictions(domain_points)

            results_df = domain_points.copy()
            results_df['lower_res'] = lower_pred
            results_df['mean_res'] = median_pred
            results_df['upper_res'] = upper_pred

            X = results_df[self.independent_variables[0]]
            Y = results_df[self.independent_variables[1]]
            Z = results_df[self.independent_variables[2]]

            lower_bin_nums = pd.qcut(lower_pred, 10, labels=False, duplicates='drop')
            mean_bin_nums = pd.qcut(median_pred, 10, labels=False, duplicates='drop')
            upper_bin_nums = pd.qcut(upper_pred, 10, labels=False, duplicates='drop')

            lower_conc_and_bins = pd.DataFrame(
                np.column_stack((X, Y, Z, lower_pred, lower_bin_nums)),
                columns=[self.independent_variables[0], self.independent_variables[1], self.independent_variables[2], 'val', 'bin']
            )

            mean_conc_and_bins = pd.DataFrame(
                np.column_stack((X, Y, Z, median_pred, mean_bin_nums)),
                columns=[self.independent_variables[0], self.independent_variables[1], self.independent_variables[2], 'val', 'bin']
            )

            upper_conc_and_bins = pd.DataFrame(
                np.column_stack((X, Y, Z, upper_pred, upper_bin_nums)),
                columns=[self.independent_variables[0], self.independent_variables[1], self.independent_variables[2], 'val', 'bin']
            )

            min_val = np.percentile([lower_pred, median_pred, upper_pred], 10)
            max_val = np.percentile([lower_pred, median_pred, upper_pred], 90)

            if isinstance(self, RegressionVisualiser):
                title_lower = 'Generated by the lower bound parameters'
                title_median = 'Generated by the median parameters'
                title_upper = 'Generated by the upper bound parameters'
                lower_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_lower, self.testing_data))
                median_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_median, self.testing_data))
                upper_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_upper, self.testing_data))
            else:
                title_lower = 'Lower bound GP prediction'
                title_median = 'Median GP prediction'
                title_upper = 'Upper bound GP prediction'


            images = []
            for bin_num in np.sort(np.unique(mean_bin_nums)):
                fig, axes = self._construct_figure(project_3D = True)

                lower_bin_data = lower_conc_and_bins[lower_conc_and_bins['bin'] >= bin_num]
                mean_bin_data = mean_conc_and_bins[mean_conc_and_bins['bin'] >= bin_num]
                upper_bin_data = upper_conc_and_bins[upper_conc_and_bins['bin'] >= bin_num]

                # Defines the lower bound subplot
                plot_1 = axes[0].scatter(lower_bin_data[self.independent_variables[0]], lower_bin_data[self.independent_variables[1]], lower_bin_data[self.independent_variables[2]], c = lower_bin_data.val, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
                axes[0].set_title(title_lower, fontsize = 20)
                axes[0].set_xlabel(self.independent_variables[0])
                axes[0].set_ylabel(self.independent_variables[1])
                axes[0].set_zlabel(self.independent_variables[2])
                axes[0].set_xlim(np.min(X), np.max(X))
                axes[0].set_ylim(np.min(Y), np.max(Y))
                axes[0].set_zlim(np.min(Z), np.max(Z))

                # Defines the mean subplot
                axes[1].scatter(mean_bin_data[self.independent_variables[0]], mean_bin_data[self.independent_variables[1]], mean_bin_data[self.independent_variables[2]], c = mean_bin_data.val, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
                axes[1].set_title(title_median, fontsize = 20)
                axes[1].set_xlabel(self.independent_variables[0])
                axes[1].set_ylabel(self.independent_variables[1])
                axes[1].set_zlabel(self.independent_variables[2])
                axes[1].set_xlim(np.min(X), np.max(X))
                axes[1].set_ylim(np.min(Y), np.max(Y))
                axes[1].set_zlim(np.min(Z), np.max(Z))

                # Defines the upper bound subplot
                axes[2].scatter(upper_bin_data[self.independent_variables[0]], upper_bin_data[self.independent_variables[1]], upper_bin_data[self.independent_variables[2]], c = upper_bin_data.val, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
                axes[2].set_title(title_upper, fontsize = 20)
                axes[2].set_xlabel(self.independent_variables[0])
                axes[2].set_ylabel(self.independent_variables[1])
                axes[2].set_zlabel(self.independent_variables[2])
                axes[2].set_xlim(np.min(X), np.max(X))
                axes[2].set_ylim(np.min(Y), np.max(Y))
                axes[2].set_zlim(np.min(Z), np.max(Z))

                if isinstance(self, RegressionVisualiser):
                    plot_2 = axes[0].scatter(self.testing_data[self.independent_variables[0]], self.testing_data[self.independent_variables[1]], self.testing_data[self.independent_variables[2]], s = 20, c = lower_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                    axes[1].scatter(self.testing_data[self.independent_variables[0]], self.testing_data[self.independent_variables[1]], self.testing_data[self.independent_variables[2]], s = 20, c = median_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                    axes[2].scatter(self.testing_data[self.independent_variables[0]], self.testing_data[self.independent_variables[1]], self.testing_data[self.independent_variables[2]], s = 20, c = upper_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                    cbar2 = plt.colorbar(plot_2, ax=axes[4], location='top', shrink=2)
                    cbar2.ax.set_title('Error in Predictions (Modeled - Measured)', fontsize=10)
                else:
                    print('GP visualisation section not yet implemented')
                    
                cbar1 = plt.colorbar(plot_1, ax=axes[3], location='top', shrink=2)
                cbar1.ax.set_title('Predicted ' + self.dependent_variable, fontsize=10)

                fig.suptitle(title, fontsize = 40)

                fig, axes = self._fill_figure_metadata(fig, axes)

                images.append(fig)

                plt.close()
            self._animate_figures(images, full_path)
            plt.close()

    def _show_3D_fixed_predictions(self, domain_points, plot_name, title):
        dir_name = self.results_path + '/instance_' + str(self.instance) + '/' + plot_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        full_path = dir_name + '/fixed_predictions.png'
        if not os.path.exists(full_path):

            lower_pred, median_pred, upper_pred = self._make_predictions(domain_points)

            results_df = domain_points.copy()
            results_df['lower_res'] = lower_pred
            results_df['mean_res'] = median_pred
            results_df['upper_res'] = upper_pred

            X = results_df[self.independent_variables[0]]
            Y = results_df[self.independent_variables[1]]
            Z = results_df[self.independent_variables[2]]

            min_val = np.percentile([lower_pred, median_pred, upper_pred], 10)
            max_val = np.percentile([lower_pred, median_pred, upper_pred], 90)


            fig, axes = self._construct_figure(project_3D = True)

            if isinstance(self, RegressionVisualiser):
                title_lower = 'Generated by the lower bound parameters'
                title_median = 'Generated by the median parameters'
                title_upper = 'Generated by the upper bound parameters'

                lower_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_lower, self.testing_data))
                median_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_median, self.testing_data))
                upper_test_error = np.abs(self.testing_data[self.dependent_variable]-self.model_func(self.params_upper, self.testing_data))
                plot_4 = axes[0].scatter(self.testing_data[self.independent_variables[0]], self.testing_data[self.independent_variables[1]], self.testing_data[self.independent_variables[2]], s = 20, c = lower_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                axes[1].scatter(self.testing_data[self.independent_variables[0]], self.testing_data[self.independent_variables[1]], self.testing_data[self.independent_variables[2]], s = 20, c = median_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                axes[2].scatter(self.testing_data[self.independent_variables[0]], self.testing_data[self.independent_variables[1]], self.testing_data[self.independent_variables[2]], s = 20, c = upper_test_error, label='Measured ' + self.dependent_variable + ' test values', cmap = 'binary', edgecolors='k')
                cbar2 = plt.colorbar(plot_4, ax=axes[4], location='top', shrink=2)
                cbar2.ax.set_title('Error in Predictions (Modeled - Measured)', fontsize=10)
                
            else:
                title_lower = 'Lower bound GP prediction'
                title_median = 'Median GP prediction'
                title_upper = 'Upper bound GP prediction'

            plot_1 = axes[0].scatter(X, Y, Z, c = lower_pred, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
            axes[0].set_title(title_lower, fontsize = 20)
            axes[0].set_xlabel(self.independent_variables[0])
            axes[0].set_ylabel(self.independent_variables[1])
            axes[0].set_zlabel(self.independent_variables[2])
            axes[0].set_xlim(np.min(X), np.max(X))
            axes[0].set_ylim(np.min(Y), np.max(Y))
            axes[0].set_zlim(np.min(Z), np.max(Z))
            
            axes[1].scatter(X, Y, Z, c = median_pred, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
            axes[1].set_title(title_median, fontsize = 20)
            axes[1].set_xlabel(self.independent_variables[0])
            axes[1].set_ylabel(self.independent_variables[1])
            axes[1].set_zlabel(self.independent_variables[2])
            axes[1].set_xlim(np.min(X), np.max(X))
            axes[1].set_ylim(np.min(Y), np.max(Y))
            axes[1].set_zlim(np.min(Z), np.max(Z))
            
            axes[2].scatter(X, Y, Z, c = upper_pred, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
            axes[2].set_title(title_upper, fontsize = 20)
            axes[2].set_xlabel(self.independent_variables[0])
            axes[2].set_ylabel(self.independent_variables[1])
            axes[2].set_zlabel(self.independent_variables[2])
            axes[2].set_xlim(np.min(X), np.max(X))
            axes[2].set_ylim(np.min(Y), np.max(Y))
            axes[2].set_zlim(np.min(Z), np.max(Z))
            
            cbar1 = plt.colorbar(plot_1, ax=axes[3], location='top', shrink=2)
            cbar1.ax.set_title('Predicted ' + self.dependent_variable, fontsize=10)
            
            fig.suptitle(title, fontsize = 40)

            fig, axes = self._fill_figure_metadata(fig, axes)
            
            fig.savefig(full_path)

    def _animate_figures(self, images, filename):
        frames = []
            
        for fig in images[::-1]:
            buf = io.BytesIO()  # Create an in-memory buffer
            fig.savefig(buf, format="png")  # Save figure to buffer
            buf.seek(0)  # Reset buffer position
            img = imageio.imread(buf)  # Read as an image
            frames.append(img)  # Store the image

        # Save as GIF or MP4
        imageio.mimsave(filename, frames, fps=1)  # Use '.mp4' for video

        print(f"Animation saved as {filename}")

    def _show_2D_cross_sections_predictions(self, domain, plot_name, title, cross_section_params):
        cross_sections = domain.cross_sections
        dir_name = self.results_path + '/instance_' + str(self.instance) + '/' + plot_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        for i, cross_section in enumerate(cross_sections):
            full_path = dir_name + '/cross_section_' + str(i+1) + '.png'
            all_points = cross_section['points']
            projected_points = cross_section['projected_points']
            if not os.path.exists(full_path):


                lower_pred, median_pred, upper_pred = self._make_predictions(all_points)
                
                u_name = projected_points.columns[0]
                v_name = projected_points.columns[1]

                results_df = projected_points.copy()
                results_df['lower_res'] = lower_pred
                results_df['mean_res'] = median_pred
                results_df['upper_res'] = upper_pred

                X = results_df[u_name]
                Y = results_df[v_name]

                min_val = np.percentile([lower_pred, median_pred, upper_pred], 10)
                max_val = np.percentile([lower_pred, median_pred, upper_pred], 90)

                fig, axes = self._construct_figure()

                if isinstance(self, RegressionVisualiser):
                    title_lower = 'Generated by the lower bound parameters'
                    title_median = 'Generated by the median parameters'
                    title_upper = 'Generated by the upper bound parameters'

                else:
                    title_lower = 'Lower bound GP prediction'
                    title_median = 'Median GP prediction'
                    title_upper = 'Upper bound GP prediction'

                plot_1 = axes[0].tricontourf(X, Y, lower_pred, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
                axes[0].set_title(title_lower, fontsize = 20)
                axes[0].set_xlabel(u_name)
                axes[0].set_ylabel(v_name)
                axes[0].set_xlim(np.min(X), np.max(X))
                axes[0].set_ylim(np.min(Y), np.max(Y))

                axes[1].tricontourf(X, Y, median_pred, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.7, s=1)
                axes[1].set_title(title_median, fontsize = 20)
                axes[1].set_xlabel(u_name)
                axes[1].set_ylabel(v_name)
                axes[1].set_xlim(np.min(X), np.max(X))
                axes[1].set_ylim(np.min(Y), np.max(Y))

                axes[2].tricontourf(X, Y, upper_pred, cmap='jet', vmin = min_val, vmax = max_val, alpha = 0.7, s=1)
                axes[2].set_title(title_upper, fontsize = 20)
                axes[2].set_xlabel(u_name)
                axes[2].set_ylabel(v_name)
                axes[2].set_xlim(np.min(X), np.max(X))
                axes[2].set_ylim(np.min(Y), np.max(Y))
                
                cbar1 = plt.colorbar(plot_1, ax=axes[3], location='top', shrink=2)
                cbar1.ax.set_title('Predicted ' + self.dependent_variable, fontsize=10)
                
                fig.suptitle(f'{title}\nCross Section {i+1}', fontsize = 40)
                
                fig, axes = self._fill_figure_metadata(fig, axes)

                fig.savefig(full_path)
                plt.close()
                
    def generate_report(self, title, include = 'all'):
        HTML_string = '''
        <!DOCTYPE html>
        <html>
        <head>
            <script type="text/javascript" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
            </script>
        </head>
        <body>
        '''
        HTML_string_section = []
        HTML_string_section.append(f"<h1>{title}</h1>")
        HTML_string_section.append(self._generate_input_summary())

        if include == 'all':
            HTML_string_section.append(self._data_visualisation_report_section())
            HTML_string_section.append(self._traceplots_report_section())
            HTML_string_section.append(self._priors_report_section())
            HTML_string_section.append(self._posteriors_report_section())
            HTML_string_section.append(self._autocorrelations_report_section())
            HTML_string_section.append(self._predictions_report_section())
            HTML_string_section.append(self._summary_report_section())
        elif isinstance(include, list):
            for item in include:
                if item == 'data':
                    HTML_string_section.append(self._data_visualisation_report_section())
                if item == 'prior':
                    HTML_string_section.append(self._priors_report_section())
                elif item == 'posterior':
                    HTML_string_section.append(self._posteriors_report_section())
                elif item == 'autocorrelation':
                    HTML_string_section.append(self._autocorrelations_report_section())
                elif item == 'summary':
                    HTML_string_section.append(self._summary_report_section())
                elif item == 'traceplots':
                    HTML_string_section.append(self._traceplots_report_section())
                elif item == 'predictions':
                    HTML_string_section.append(self._predictions_report_section())
                else:
                    raise ValueError("Visualiser - Invalid report item. Expected 'prior', 'posterior', 'autocorrelation', 'summary' or 'all'.")
        else:
            raise ValueError("Visualiser - Invalid report item. Expected a list or 'all'.")
    
        HTML_string += "\n<hr>\n".join(HTML_string_section)
        HTML_string += '</body></html>'
        report_path = self.results_path + '/instance_' + str(self.instance) + '/report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(HTML_string)

    def _priors_report_section(self):
        HTML_string = "<h2>Priors</h2>"
        prior_folder = self.results_path + '/instance_' + str(self.instance) + '/prior_dists'
        
        if not os.path.exists(prior_folder):
            HTML_string += "<p>No prior plots generated.</p>"
        else:
            # Start a div container for Flexbox to arrange images side by side
            HTML_string += '<div style="display: flex; justify-content: space-between;">\n'
            
            for param_name in self.inference_params.keys():
                if self.inference_params[param_name].n_dims == 1:
                    filename = 'prior_dist_' + param_name + '.png'
                elif self.inference_params[param_name].n_dims == 2:
                    param_1_name = self.inference_params[param_name].name[0]
                    param_2_name = self.inference_params[param_name].name[1]
                    filename = 'prior_dist_' + param_1_name + '_' + param_2_name + '.png'
                else:
                    continue

                full_path = prior_folder + '/' + filename
                rel_path = './prior_dists/' + filename

                if os.path.exists(full_path):
                    # Add each image to the div with a smaller width and auto height
                    HTML_string += f'<img src="{rel_path}" alt="Prior distribution for {param_name}" style="width: 30%; height: auto; margin-right: 10px;">\n'
                else:
                    HTML_string += f"<p>Prior plot for {param_name} not found.</p>\n"
            
            # Close the div container
            HTML_string += '</div><br>\n'
        
        return HTML_string
    
    def _posteriors_report_section(self):
        HTML_string = "<h2>Posteriors</h2>"
        posterior_folder = self.results_path + '/instance_' + str(self.instance) + '/posterior_dists'

        if not os.path.exists(posterior_folder):
            HTML_string += "<p>No posterior plots generated.</p>"
        else:
            # Start a div container for Flexbox to arrange images side by side
            HTML_string += '<div style="display: flex; justify-content: space-between;">\n'
            
            for param_name in self.inference_params.keys():
                if self.inference_params[param_name].n_dims == 1:
                    filename = 'posterior_dist_' + param_name + '.png'
                elif self.inference_params[param_name].n_dims == 2:
                    param_1_name = self.inference_params[param_name].name[0]
                    param_2_name = self.inference_params[param_name].name[1]
                    filename = 'posterior_dist_' + param_1_name + '_' + param_2_name + '.png'
                else:
                    continue

                full_path = posterior_folder + '/' + filename
                rel_path = './posterior_dists/' + filename

                if os.path.exists(full_path):
                    # Add each image to the div with a smaller width and auto height
                    HTML_string += f'<img src="{rel_path}" alt="Posterior distribution for {param_name}" style="width: 30%; height: auto; margin-right: 10px;">\n'
                else:
                    HTML_string += f"<p>Posterior plot for {param_name} not found.</p>\n"
            
            # Close the div container
            HTML_string += '</div><br>\n'
        return HTML_string
    
    def _autocorrelations_report_section(self):
        HTML_string = "<h2>Autocorrelations</h2>"
        autocorr_folder = self.results_path + '/instance_' + str(self.instance) + '/autocorrelations'

        if not os.path.exists(autocorr_folder):
            HTML_string += "<p>No autocorrelation plots generated.</p>"
        else:
            for chain in range(self.n_chains):
                HTML_string += f"<h3>Chain {chain + 1}</h3>"
                HTML_string += '<div style="display: flex; flex-wrap: wrap;">\n'
                
                for param in self.samples.columns:
                    filename = f'autocorrelations_{param}_chain_{chain + 1}.png'
                    full_path = autocorr_folder + '/' + filename
                    rel_path = './autocorrelations/' + filename

                    if os.path.exists(full_path):
                        HTML_string += f'<img src="{rel_path}" alt="Autocorrelation for {param} in chain {chain + 1}" style="width: {100/(len(self.samples.columns)+1)}%; height: auto; margin-right: 10px; margin-bottom: 10px;">\n'
                    else:
                        HTML_string += f"<p>Autocorrelation plot for {param} in chain {chain + 1} not found.</p>\n"
                
                HTML_string += '</div><br>\n'
        return HTML_string 
    
    def _summary_report_section(self):
        HTML_string = "<h2>Results Summary</h2>"

        summary = self.get_summary()
           
        # Overall statistics
        HTML_string += f"<p><strong>RMSE:</strong> {summary['RMSE']:.2f}</p>"
        HTML_string += f"<p><strong>AIC:</strong> {summary['AIC']:.2f}</p>"
        HTML_string += f"<p><strong>BIC:</strong> {summary['BIC']:.2f}</p>"
        
        # Start a container for side-by-side tables
        HTML_string += "<div style='display: flex; flex-wrap: wrap;'>"

        # Chain data side by side
        for chain_name, chain_data in summary.items():
            if chain_name not in ["RMSE", "AIC", "BIC", "overall"]:
                HTML_string += f"<div style='flex: 1; margin-right: 20px;'>"
                HTML_string += f"<h3>{chain_name.capitalize()}</h3>"
                HTML_string += "<table border='1' style='border-collapse: collapse;'><tr><th>Parameter</th><th>Lower</th><th>Mean</th><th>Upper</th><th>Tau</th></tr>"
                for param_name, param_data in chain_data.items():
                    HTML_string += f"<tr><td>{param_name.capitalize()}</td>"
                    HTML_string += f"<td>{param_data['lower']:.2g}</td>"
                    HTML_string += f"<td>{param_data['mean']:.2g}</td>"
                    HTML_string += f"<td>{param_data['upper']:.2g}</td>"
                    HTML_string += f"<td>{param_data['tau']:.2g}</td></tr>"
                HTML_string += "</table></div>"
        
        # Close the div for side-by-side layout
        HTML_string += "</div>"

        # Overall statistics for parameters
        if 'overall' in summary:
            HTML_string += "<div style='margin-top: 20px;'>"
            HTML_string += "<h3>Overall</h3>"
            HTML_string += "<table border='1' style='border-collapse: collapse;'><tr><th>Parameter</th><th>Lower</th><th>Mean</th><th>Upper</th><th>Tau</th></tr>"
            for param_name, param_data in summary['overall'].items():
                HTML_string += f"<tr><td>{param_name.capitalize()}</td>"
                HTML_string += f"<td>{param_data['lower']:.2g}</td>"
                HTML_string += f"<td>{param_data['mean']:.2g}</td>"
                HTML_string += f"<td>{param_data['upper']:.2g}</td>"
                HTML_string += f"<td>{param_data['tau']:.2g}</td></tr>"
            HTML_string += "</table></div>"
        
        return HTML_string
    
    def _traceplots_report_section(self):
        HTML_string = "<h2>Traceplots</h2>"
        traceplot_folder = self.results_path + '/instance_' + str(self.instance) + '/traceplots'

        if not os.path.exists(traceplot_folder):
            HTML_string += "<p>No traceplots generated.</p>"
        else:
            HTML_string += '<div style="display: flex; justify-content: space-between;">'
            for chain in range(self.n_chains):
                
                filename = f'traceplot_{chain + 1}.png'
                full_path = traceplot_folder + '/' + filename
                rel_path = './traceplots/' + filename

                if os.path.exists(full_path):
                    HTML_string += f'<img src="{rel_path}" alt="Traceplot for chain {chain + 1}" style="width: {100/self.n_chains}%; height: auto; margin-right: 10px;">\n'
                else:
                    HTML_string += f"<p>Traceplot for chain {chain + 1} not found.</p>\n"
                
            HTML_string += '</div><br>\n'
        return HTML_string
  
    def _data_visualisation_report_section(self):
        HTML_string = "<h2>Data</h2>"
        if type(self.data_processor) == SimDataProcessor:
            data_vis_file= self.data_processor.data_path + '/processed_sim_data/' + self.data_processor.processed_data_name + '/data_plot.png'
            if not os.path.exists(data_vis_file):
                HTML_string += "<p>No data visualisation plot generated.</p>"
            else:
                rel_data_vis_file = os.path.relpath(data_vis_file, start=self.results_path + '/instance_' + str(self.instance))
                HTML_string += f'<img src="{rel_data_vis_file}" alt="Data visualisation plot" style="width: 50%; height: auto;">\n'
            return HTML_string
    
        elif type(self.data_processor) == RawDataProcessor:
            data_vis_folder = self.data_processor.data_path + '/processed_raw_data/' + self.data_processor.processed_data_name
            if not os.path.exists(data_vis_folder):
                HTML_string += "<p>No data visualisation plots generated.</p>"
            else:
                HTML_string += '<div style="display: flex; flex-wrap: wrap;">\n'
                num_figures = len([file for file in os.listdir(data_vis_folder) if file.endswith('.png')])
                for file in os.listdir(data_vis_folder):
                    
                    if file.endswith('.png'):
                        rel_path = os.path.relpath(os.path.join(data_vis_folder, file), start=self.results_path + '/instance_' + str(self.instance))
                        HTML_string += f'<img src="{rel_path}" alt="Data visualisation plot" style="width: {100/(num_figures+1)}%; height: auto; margin-right: 10px; margin-bottom: 10px;">\n'
                HTML_string += '</div><br>\n'
            return HTML_string
        else:
            raise ValueError("Visualiser - Data processor must be a RawDataProcessor or SimDataProcessor.")

    def _predictions_report_section(self):
        HTML_string = "<h2>Predictions</h2>"
        predictions_folder = self.results_path + '/instance_' + str(self.instance) + '/predictions'
        for prediction in self.prediction_plots:
            plots_folder = predictions_folder + '/' +prediction['name'] + '_' + prediction['plot_type']
            if prediction['plot_type'] =='1D':
                pass
            elif prediction['plot_type'] == '2D':
                pass
            elif prediction['plot_type'] == '3D':
                pass
            elif prediction['plot_type'] == '2D_cross_sections':
                pass
 
        return HTML_string
    
    def _format_prior_params(self, params, modes = 1):
        """
        Format prior parameters to display them tidily inside the table.
        Handles arrays, lists, or individual values.
        """
        if modes > 1:
            # Format multiple modes in a structured way
            modes_array = []
            for i in range (modes):
                mode_str = ""
                mode_str += f"<strong>Mode {i + 1}</strong><br>"
                formatted_dict = []
                for key, value in params.items():
                    if key == 'overall_scale':
                        formatted_dict.append(f"Overall Scale: {value}")
                    else:
                        formatted_dict.append(f"{key.capitalize()}: {value[i]}")
                mode_str += "<br>".join(formatted_dict)
                modes_array.append(mode_str)
            return "<br><br>".join(modes_array)

        else:
            # Format a dictionary by separating key-value pairs into new lines
            formatted_dict = []
            for key, value in params.items():
                formatted_dict.append(f"{key.capitalize()}: {value}")
            return "<br>".join(formatted_dict)

    def _generate_input_summary(self):
        HTML_string = "<h2>Input Summary</h2>"
        construction = self.construction

        equation = Eq(symbols(construction['model']['dependent_variable']), self.model_sum_expr)

        latex_model_expr = latex(equation)

        # Model section
        HTML_string += "<h3>Model</h3>"
        HTML_string += f'<p style = "font-size:150%;">\\[{latex_model_expr}\\]</p>'
        HTML_string += "<p>Independent Variables: " + ", ".join(construction['model']['independent_variables']) + "</p>"
        HTML_string += "<p>Dependent Variables: " + ", ".join(construction['model']['dependent_variable']) + "</p>"
        
        # Inference Parameters section
        HTML_string += "<h3>Inference Parameters</h3>"
        HTML_string += "<table border='1' style='border-collapse: collapse;'>"
        HTML_string += "<tr><th>Name</th><th>Prior Selection</th><th>Prior Parameters</th><th>Order</th><th>Multi Mode</th></tr>"

        for param in construction['inference_params']:
            # Handle list of names vs. single name
            param_name = ', '.join(param['name']) if isinstance(param['name'], list) else param['name']
            
            # Handle multi_mode formatting
            if param['multi_mode']:
                n_modes = len(param['prior_params'][list(param['prior_params'].keys())[0]])
                if n_modes == 1:
                    raise ValueError("Visualiser - Must have multiple modes, given multi_mode is True.")
                prior_param_formatted = self._format_prior_params(param['prior_params'], modes = n_modes)
                prior_param_cell = f"<td>{prior_param_formatted}</td>"
            else:
                prior_params = self._format_prior_params(param['prior_params'])
                prior_param_cell = f"<td>{prior_params}</td>"

            # Build the table rows dynamically
            HTML_string += f"<tr><td>{param_name}</td>"
            HTML_string += f"<td>{param['prior_select']}</td>"
            HTML_string += prior_param_cell  # Prior parameter section
            HTML_string += f"<td>{param['order']}</td>"
            HTML_string += f"<td>{'Yes' if param['multi_mode'] else 'No'}</td></tr>"

        HTML_string += "</table>"

        # Likelihood section
        HTML_string += "<h3>Likelihood</h3>"
        HTML_string += f"<p>Likelihood Selection: {construction['likelihood']['likelihood_select']}</p>"

        # Data Processor section
        HTML_string += "<h3>Data Processor</h3>"

        if type(self.data_processor) == SimDataProcessor:
            HTML_string += f"<p>Section not yet implemented</p>"

        elif type(self.data_processor) == RawDataProcessor:
            HTML_string += f"<p>Raw Data Filename: {construction['data_processor']['raw_data_filename']}</p>"
            HTML_string += f"<p>Processed Data Name: {construction['data_processor']['processed_data_name']}</p>"
            HTML_string += f"<p>Processor Selection: {construction['data_processor']['processor_select']}</p>"
            HTML_string += f"<p>Experiments List: {', '.join(construction['data_processor']['processor_params']['experiments_list'])}</p>"
            HTML_string += f"<p>Metadata Selection: {construction['data_processor']['processor_params']['meta_data_select']}</p>"
            HTML_string += f"<p>Input Header: {construction['data_processor']['processor_params']['input_header']}</p>"
            HTML_string += f"<p>Output Header: {construction['data_processor']['processor_params']['output_header']}</p>"
            HTML_string += f"<p>Log Output Data: {construction['data_processor']['processor_params']['log_output_data']}</p>"
            HTML_string += f"<p>Gridding: {', '.join(map(str, construction['data_processor']['processor_params']['gridding']))}</p>"
        else:
            raise ValueError("Visualiser - Data processor must be a RawDataProcessor or SimDataProcessor.")
        

        # Sampling Information
        HTML_string += "<h3>Sampling Information</h3>"
        HTML_string += f"<p>Number of Samples: {construction['n_samples']}</p>"
        HTML_string += f"<p>Number of Chains: {construction['n_chains']}</p>"
        HTML_string += f"<p>Thinning Rate: {construction['thinning_rate']}</p>"
        HTML_string += f"<p>Warmup Proportion: {construction['p_warmup']}</p>"

        return HTML_string

    def embed_report(self):

        """
        Embeds the HTML report in a Jupyter notebook.
        """

        full_path = self.results_path + '/instance_' + str(self.instance) + '/report.html'
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                html_string = f.read()
        else:
            raise Exception('Visualiser - HTML report not generated!')
        # Function to convert image to Base64
        def convert_image_to_base64(image_path):
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Parse the HTML content
        soup = BeautifulSoup(html_string, 'html.parser')

        # Find all <img> tags
        img_tags = soup.find_all('img')

        # Process each image tag
        for img in img_tags:
            img_src = img['src']  # Get the src attribute of the image

            # Ensure we are dealing with a local image, not an external URL
            if not img_src.startswith('http'):  # Skip external URLs
                # Make sure the image path is relative to the HTML file directory
                img_abs_path = os.path.join(os.path.dirname(full_path), img_src)

                # Check if the image file exists
                if os.path.exists(img_abs_path):
                    # Convert image to Base64
                    img_format = img_abs_path.split('.')[-1].lower()  # Get image format (e.g., png, jpg)
                    try:
                        img_base64 = convert_image_to_base64(img_abs_path)
                        img['src'] = f"data:image/{img_format};base64,{img_base64}"  # Replace src with Base64
                    except Exception as e:
                        print(f"Error embedding image {img_src}: {e}")
                else:
                    print(f"Image file not found: {img_abs_path}")

        # Save the modified HTML content with embedded images
        output_html_file = self.results_path + '/instance_' + str(self.instance) + '/embedded_report.html'
        with open(output_html_file, 'w', encoding='utf-8') as file:
            file.write(str(soup))

class RegressionVisualiser(Visualiser):
    """
    A class to handle the visualisation of the results from a regression model. This class inherits from the Visualiser class.

    Args:
        - sampler (Sampler): The sampler object

    Attributes:
        - n_samples (int): Number of samples
        - n_chains (int): Number of chains
        - samples (pd.DataFrame): Samples from the sampler
        - chain_samples (pd.DataFrame): Samples from each chain
        - fields (list): Fields of the sampler
        - data_processor (SimDataProcessor or RawDataProcessor): Data processor object
        - training_data (pd.DataFrame): Training data
        - testing_data (pd.DataFrame): Testing data
        - model_func (function): Model function
        - dependent_variable (str): Dependent variable of the model
        - independent_variables (list): Independent variables of the model
        - inference_params (pd.Series): Inference parameters. This is a series containing the Parameter objects
        - likelihood_func (function): Likelihood function
        - results_path (str): Path to save results
        - instance (int): Instance number
        - params_min (pd.Series): Minimum parameter values from the sampled results
        - params_lower (pd.Series): Lower bound parameter values from the sampled results
        - params_median (pd.Series): Median parameter values from the sampled results
        - params_upper (pd.Series): Upper bound parameter values from the sampled results
        - params_max (pd.Series): Maximum parameter values from the sampled results
        - RMSE (float): Root Mean Squared Error
        - AIC (float): Akaike Information Criterion
        - BIC (float): Bayesian Information Criterion
        - autocorrs (dict): A dictionary containing the autocorrelation values for each parameter

    """

    def __init__(self, sampler: Sampler):
        """
        Initialises the Visualiser class saving all relevant variables and performing some initialising tasks

        Args:
            - sampler (Sampler): The sampler object

        """

        super().__init__(sampler)

        if not sampler.sampled:
            raise Exception('Visualiser - sampler has not been run!')
        
        self.n_samples = sampler.n_samples
        self.n_chains = sampler.n_chains
        self.samples = sampler.samples
        self.chain_samples = sampler.chain_samples
        self.fields = sampler.fields
        
        self.model_func = sampler.model_func
        self.model_sum_expr = sampler.model.sum_expr
        self.inference_params = sampler.inference_params

        self.likelihood_func= sampler.likelihood_func
        
        self.params_min = self.samples.min()
        self.params_lower = self._get_ag_samples(self.samples, 0.05)
        self.params_median = self._get_ag_samples(self.samples, 0.5)
        self.params_upper = self._get_ag_samples(self.samples, 0.95)
        self.params_max = self.samples.max()

        test_predictions = self.model_func(self.params_median, self.testing_data)
        test_measured = self.testing_data[self.dependent_variable[0]]

        log_likelihood = self.likelihood_func(test_predictions, self.params_median).log_prob(jnp.array(test_measured)).sum()
        
        self.RMSE = np.sqrt(np.mean((test_predictions-test_measured)**2))
        self.AIC = 2*self.params_median.size - log_likelihood
        self.BIC = self.params_median.size*np.log(self.testing_data.shape[0]) - 2*log_likelihood

        self.autocorrs = self._calculate_autocorrs()

        self.prediction_plots = []

    def _make_predictions(self, domain_points):
        lower_pred = self.model_func(self.params_lower, domain_points)
        median_pred = self.model_func(self.params_median, domain_points)
        upper_pred = self.model_func(self.params_upper, domain_points)

        return lower_pred, median_pred, upper_pred

    def get_traceplots(self):
        """
        Generates and saves the traceplots from the sampled results. If there are multiple chains of samples then multiple traceplots are generated and saved. One file is saved per chain, including the traceplot for each inferred parameter.
        """
        traceplot_folder = self.results_path + '/instance_' + str(self.instance) + '/traceplots'
        if not os.path.exists(traceplot_folder):
            os.mkdir(traceplot_folder)

        for chain in range(self.n_chains):
            full_path = traceplot_folder + '/traceplot_' + str(chain + 1) + '.png'
            if self.n_chains == 1:
                title = 'MCMC samples'
                samples = self.samples
            else:
                title = 'MCMC samples for chain ' + str(chain + 1)
                samples = self.chain_samples[self.chain_samples['chain'] == chain + 1].drop(columns=['chain', 'sample_index'])
            
            if not os.path.exists(full_path):
                tp = self._traceplots(samples.values, xnames=self.samples.columns, title=title)
                tp.savefig(full_path)

    def _traceplots(self, x, xnames=None, title=None):
        """
        Generates a traceplot based on inputted samples

        Args:
            - x (np.ndarray): Input samples
            - xnames (list): Names of the variables
            - title (str): Title of the traceplot

        Returns:
            matplotlib.figure.Figure: The generated traceplot figure
        """
        _, d = x.shape
        fig = plt.figure(figsize=(9, 6))
        left, tracewidth, histwidth = 0.1, 0.65, 0.15
        bottom, rowheight = 0.1, 0.8 / d

        for i in range(d):
            rowbottom = bottom + i * rowheight
            rect_trace = (left, rowbottom, tracewidth, rowheight * 0.8)
            rect_hist = (left + tracewidth, rowbottom, histwidth, rowheight * 0.8)

            if i == 0:
                ax_trace = fig.add_axes(rect_trace)
                ax_trace.plot(x[:, i])
                ax_trace.set_xlabel("Sample Count")
                ax_tr0 = ax_trace

            elif i > 0:
                ax_trace = fig.add_axes(rect_trace, sharex=ax_tr0)
                ax_trace.plot(x[:, i])
                plt.setp(ax_trace.get_xticklabels(), visible=False)

            if i == d - 1 and title is not None:
                plt.title(title)

            if xnames is not None:
                ax_trace.set_ylabel(xnames[i])

            ax_hist = fig.add_axes(rect_hist, sharey=ax_trace)

            try:
                ax_hist.hist(x[:, i], orientation='horizontal', bins=50)
            except Exception as e:
                raise Exception('Visualiser - Traceplot histogram invalid!') from e

            plt.setp(ax_hist.get_xticklabels(), visible=False)
            plt.setp(ax_hist.get_yticklabels(), visible=False)
            xlim = ax_hist.get_xlim()
            ax_hist.set_xlim([xlim[0], 1.1 * xlim[1]])

        # Adjust layout to add more padding between subplots
        plt.subplots_adjust(hspace=0.5, left=0.15, right=0.85, top=0.95, bottom=0.1)

        plt.close()
        return fig
    
    def _get_param_accuracy_string(self):
        """
        Returns a string representation of the accuracy of each parameter in the samples.

        The accuracy is calculated as the percentage error between the mean value of the parameter
        in the samples and the corresponding fixed model parameter value.

        """
        param_accuracy_string_array = []
        for param in self.samples.columns:
            if type(self.data_processor) == SimDataProcessor and param in self.data_processor.model.fixed_model_params:
                value1 = self.params_median[param]
                value2 = self.data_processor.model.fixed_model_params[param]
                if value1 == 0 or value2 == 0:
                    print('Visualiser - cannot calculate percentage error for ' + param)
                elif abs(value1 + value2)/2 < 1e-6:
                    print('Visualiser - cannot calculate percentage error for ' + param)
                elif value1 < 0 or value2 < 0:
                    print("Visualiser - Negative values encountered; ensure the context makes sense for percentage difference.")

                percentage_error = 100 * abs(value1 - value2) / ((value1 + value2) / 2)
                param_accuracy_string_array.append(param + ' error = ' + f'{percentage_error:.3}' + '%')
        return ('\n').join(param_accuracy_string_array)

    def _get_param_string(self):
        """
        Returns a formatted string representation of the parameters.

        Returns:
            str: A string representation of the parameters in the format:
                 "<param_name> = [<lower_value>, <mean_value>, <upper_value>]"
        """
        param_string_array = []
        for param in self.params_median.index:
            if np.floor(np.log10(self.params_median[param])) < 2:
                lower_string =  "{:.3f}".format(self.params_lower[param])
                mean_string = "{:.3f}".format(self.params_median[param])
                upper_string = "{:.3f}".format(self.params_upper[param])
            else:
                lower_string =  "{:.3e}".format(self.params_lower[param])
                mean_string = "{:.3e}".format(self.params_median[param])
                upper_string = "{:.3e}".format(self.params_upper[param])
            param_string_array.append(param + ' = [' + lower_string + ', ' + mean_string + ', ' + upper_string + ']')

        return ('\n').join(param_string_array)
        
    def _get_ag_samples(self, samples, q_val):
        """
        Calculate the aggregated samples for each column in the given DataFrame.

        Parameters:
            - samples (pd.DataFrame): The DataFrame containing the samples.
            - q_val (float): The quantile value used to calculate the aggregated sample.

        Returns:
            - pd.Series: A Series containing the aggregated samples for each column.
        """
        ags = pd.Series({}, dtype='float64')
        for col in samples.columns:
            param_samples = samples[col]
            ag = np.quantile(param_samples, q_val)
            ags[col] = ag
        return ags
    
    def _animate(self, name, frame_dur=500):
        """
        Gathers all of the figures under the inputted name, creates an animation of them and saves that animation

        Args:
            - name (str): The name of the animation.
            - frame_dur (int, optional): The duration of each frame in milliseconds. Defaults to 500.

        """
        folder_name = 'instance_' + str(self.instance) + '/predictions/' + name + '/figures'
        gif_name = name + '.gif'
        gif_path = self.results_path + '/' + 'instance_' + str(self.instance) + '/predictions/' + name + '/' + gif_name

        if not os.path.exists(self.results_path + '/' + folder_name):
            raise Exception('Visualiser - Images for animation do not exist')
        elif os.path.exists(gif_path):
            pass
        else:
            files = os.listdir(self.results_path + '/' + folder_name)

            images = []
            for i in range(len(files))[::-1]:
                images.append(imageio.imread(self.results_path + '/' + folder_name + '/' + files[i]))

            imageio.mimsave(gif_path, images, duration=frame_dur, loop=0)

    def plot_prior(self, param_name: str, param_range: list, references: dict = None, show_estimate: bool = False):
        """
        Plots the prior distribution for a given parameter. Either a 1D or 2D plot is generated based on the number of dimensions of the parameter. The plot is saved in the results directory.

        Args:
            - param_name (str): The name of the parameter. If the parameter is a 2D parameter, the name should be the name of the parameter is the one given as the key in the inference_params series.
            - param_range (list): The range of the parameter values for the plot. Indicates the maximum and minimum values of the parameter to plot the prior between. The range should be a list containing two floats or integers in the 1D case and a list of of two lists, each containing two floats or integers in the 2D case.
            - references (dict, optional): A dictionary containing reference values for the parameter. The dictionary must have keys 'labels' and 'vals', which are lists of labels and values respectively. Defaults to None.
            - show_estimate (bool, optional): Whether to show the estimated median value from the sampled posterior. Defaults to False.
        """
        if param_name not in self.inference_params:
            raise Exception('Visualiser - parameter not listed as an inference parameter!')
        if self.inference_params[param_name].n_dims == 1:
            self._plot_one_prior(param_name, param_range, references, show_estimate)
        elif self.inference_params[param_name].n_dims == 2:
            self._plot_two_priors(param_name, param_range, references, show_estimate)
        else:
            raise Exception('Visualiser - visualising more than 2 dimensions is not supported!')
            
    def _plot_two_priors(self, param_name, param_range, references=None, show_estimate = False):
        """
        Plot the two priors for a given parameter.

        Args:
            - param_name (str): The name of the parameter.
            - param_range (list): A list of two lists, each containing two floats or integers, representing the range of the parameter.
            - references (dict, optional): A dictionary containing references for the parameter. 
                - It should have keys 'labels' and 'vals', where 'labels' is a list of strings and 'vals' is a list of lists, each containing two floats or integers. Defaults to None.
            - show_estimate (bool, optional): Whether to show the estimated median value. Defaults to False.

        """
        plt.figure(figsize=(8, 8))
        handles = []

        if references is not None:
            if not isinstance(references, dict):
                raise ValueError("Visualiser - Invalid references format. Expected references to be a dictionary.")
            if "labels" not in references or "vals" not in references:
                raise ValueError("Visualiser - Invalid references format. Expected references to have keys 'labels' and 'vals'.")
            if not isinstance(references["labels"], list) or not isinstance(references["vals"], list):
                raise ValueError("Visualiser - Invalid references format. Expected 'labels' and 'vals' to be lists.")
            if len(references["labels"]) != len(references["vals"]):
                raise ValueError("Visualiser - Invalid references format. Expected 'labels' and 'vals' to have the same length.")
            reference_labels = references["labels"]
            reference_vals = np.array(references["vals"])
            if not all(isinstance(label, str) for label in references["labels"]):
                raise ValueError("Visualiser - Invalid references format. Expected 'labels' to contain only strings.")
            if not all(isinstance(val, list) and len(val) == 2 and all(isinstance(sub_val,(int,float)) for sub_val in val) for val in references["vals"]):
                raise ValueError("Visualiser - Invalid references format. Expected 'vals' to contain only lists of two floats or integers.")
            
            refs = plt.plot(reference_vals[:,0], reference_vals[:,1], ".r", label="References")
            handles.append(refs[0])
            for i, (reference_x_point, reference_y_point) in enumerate(zip(reference_vals[:,0], reference_vals[:,1])):
                label = reference_labels[i]
                plt.annotate(label, (reference_x_point, reference_y_point), textcoords="offset points", 
                                xytext=(0, 5), ha="center", color="r", zorder=3)

        if not isinstance(param_range, list) or len(param_range) != 2 or not isinstance(param_range[0], list) or not isinstance(param_range[1], list) or len(param_range[0]) != 2 or len(param_range[1]) != 2 or not all(isinstance(val, (float, int)) for val in param_range[0]) or not all(isinstance(val, (float, int)) for val in param_range[1]):
            raise ValueError("Invalid param_range format. Expected param_range to be a list of two lists, each containing two floats or integers.")
        param_1_range = [param_range[0][0], param_range[0][1]]
        param_2_range = [param_range[1][0], param_range[1][1]]

        param_1_name = self.inference_params[param_name].name[0]
        param_2_name = self.inference_params[param_name].name[1]

        if param_1_range[0] > self.params_median[param_1_name]:
            param_1_range[0] = self.params_median[param_1_name]
        elif param_1_range[1] < self.params_median[param_1_name]:
            param_1_range[1] = self.params_median[param_1_name]

        if param_2_range[0] > self.params_median[param_2_name]:
            param_2_range[0] = self.params_median[param_2_name]
        elif param_2_range[1] < self.params_median[param_2_name]:
            param_2_range[1] = self.params_median[param_2_name]

        param_1_linspace = np.linspace(param_1_range[0], param_1_range[1], 100)
        param_2_linspace = np.linspace(param_2_range[0], param_2_range[1], 100)
        param_1_mesh, param_2_mesh = np.meshgrid(param_1_linspace, param_2_linspace)
        shape = param_1_mesh.shape

        parameter = self.inference_params[param_name]
        prior_dist = parameter.get_prior_function()

        log_P = prior_dist.log_prob(np.array([param_1_mesh.flatten()/(10**parameter.order), param_2_mesh.flatten()/(10**parameter.order)]).T)
        log_P = np.reshape(log_P, shape)

        plt.contourf(param_1_mesh, param_2_mesh, np.exp(log_P), levels=30, cmap='Greys', vmin = np.percentile(np.exp(log_P),5))
        plt.xlabel(param_1_name, fontsize = 15)
        plt.ylabel(param_2_name, fontsize = 15)
        plt.title('Joint prior distribution of ' + param_1_name + ' and ' + param_2_name, fontsize = 20)

        if show_estimate:
            est_median_val = plt.scatter([self.params_median[param_1_name]], [self.params_median[param_2_name]], color = 'w', marker='s', edgecolors='k', s=30, label = 'Estimated Median Value')
            handles.append(est_median_val)

            if self.data_processor and type(self.data_processor) == SimDataProcessor and param_1_name in self.data_processor.model.fixed_model_params and param_2_name in self.data_processor.model.fixed_model_params:
                act_val = plt.scatter([self.data_processor.model.fixed_model_params[param_1_name]], [self.data_processor.model.fixed_model_params[param_2_name]], color = 'white', marker='*', edgecolors='black', label = "Actual Value")
                handles.append(act_val)

        plt.legend(handles = handles, fontsize = 15)

        filename = 'prior_dist_' + param_1_name + '_' + param_2_name +'.png'
        folder_name = self.results_path + '/instance_' + str(self.instance) + '/prior_dists'
        full_path = folder_name + '/' + filename
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig(full_path)
        plt.close()

    def _plot_one_prior(self, param_name, param_range, references=None, show_estimate = False):
        """
        Plot the prior distribution for a given parameter.

        Args:
            - param_name (str): The name of the parameter.
            - param_range (list): A list with two floats or integers indicating the maximum and minimum values to plot the prior between.
            - references (dict, optional): A dictionary containing reference values for the parameter. The dictionary must have keys 'labels' and 'vals', which are lists of labels and values respectively. Defaults to None.
        """
        if not isinstance(param_range, list) or len(param_range) != 2 or not all(isinstance(val, (float, int)) for val in param_range):
            raise Exception('Visualiser - Invalid param_range. It should be a list with two floats or integers indicating the maximum and minimum values to plot the prior between.')

        if references:
            if not isinstance(references, dict):
                raise ValueError("Visualiser - References must be a dictionary.")
            if "labels" not in references or "vals" not in references:
                raise ValueError("Visualiser - References dictionary must contain keys 'labels' and 'vals'.")
            if not isinstance(references["labels"], list) or not isinstance(references["vals"], list):
                raise ValueError("Visualiser - References 'labels' and 'vals' must be lists.")
            if len(references["labels"]) != len(references["vals"]):
                raise ValueError("Visualiser - References 'labels' and 'vals' must have the same length.")
            if not all(isinstance(label, str) for label in references["labels"]):
                raise ValueError("Visualiser - References 'labels' must contain only strings.")
            if not all(isinstance(val, (int, float)) for val in references["vals"]):
                raise ValueError("Visualiser - References 'vals' must contain only integers or floats.")

        # Correction to fit estimated value into distribution
        if param_range[0] > self.params_median[param_name]:
            param_range[0] = self.params_median[param_name]
        elif param_range[1] < self.params_median[param_name]:
            param_range[1] = self.params_median[param_name]

        param_linspace = np.linspace(param_range[0], param_range[1], 500)

        parameter = self.inference_params[param_name]
        prior_func = parameter.get_prior_function()
        log_prior_P = prior_func.log_prob(param_linspace/(10**parameter.order))
        
        plt.figure(figsize=(8, 8))
        dist_plot = plt.plot(param_linspace, np.exp(log_prior_P), color='k', label='Prior Distribution')
        plt.xlabel(param_name, fontsize=15)
        plt.ylabel('log(P)', fontsize=15)
        plt.title('Prior distribution of ' + param_name, fontsize=20)
        plt.yticks([])

        handles = [dist_plot[0]]

        if references:
            reference_labels = references['labels']
            reference_x = references['vals']
            refs = []

            for i, reference_x_point in enumerate(reference_x):
                label = reference_labels[i]

                ref = plt.axvline(reference_x_point, color='r', linestyle='dotted', label=label)
                refs.append(ref)

            offset = np.linspace(-np.percentile(log_prior_P,99)/2,np.percentile(log_prior_P,99)/2,len(reference_labels))
            labelLines(refs, zorder=2.5, align=True)

            ref.set_label('References')
            handles.append(ref)


        if show_estimate:
            est_val = plt.axvline(self.params_median[param_name], color='b', linestyle='dashed', label='Estimated Value')
            handles.append(est_val)

            if type(self.data_processor) == SimDataProcessor and param_name in self.data_processor.model.fixed_model_params:
                act_val = plt.axvline(self.data_processor.model.fixed_model_params[param_name], color='g', linestyle='dashed', label='Actual Value')
                handles.append(act_val)

        plt.legend(handles=handles, fontsize=15)

        filename = 'prior_dist_' + param_name + '.png'
        folder_name = self.results_path + '/instance_' + str(self.instance) + '/prior_dists'
        full_path = folder_name + '/' + filename
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig(full_path)
        plt.close()

    def get_autocorrelations(self):
        """
        Generates and saves autocorrelation plots for each parameter of the MCMC samples. A plot is generated for each parameter within each chain.
        """
        autocorr_folder = self.results_path + '/instance_' + str(self.instance) + '/autocorrelations'
        if not os.path.exists(autocorr_folder):
            os.mkdir(autocorr_folder)

        for chain in range(self.n_chains):
            for param in self.samples.columns:
                full_path = autocorr_folder + '/autocorrelations_' + param + '_chain_' + str(chain + 1) + '.png'

                if self.n_chains == 1:
                    title = 'MCMC autocorrelations for ' + param
                else:
                    title = 'MCMC autocorrelations for ' + param + ', chain ' + str(chain + 1)

                if not os.path.exists(full_path):
                    fig = plt.figure(figsize=(6,4))
                    autocorrelations = self.autocorrs['chain_' + str(chain + 1)][param]['Ct']
                    tau = self.autocorrs['chain_' + str(chain + 1)][param]['tau']
                    ci = self.autocorrs['chain_' + str(chain + 1)][param]['ci']
                    formatted_tau = "{:.2f}".format(tau)
                    plt.bar(range(autocorrelations.size), autocorrelations, label = param + ', tau = ' + formatted_tau)
                    plt.axhline(y = ci, color = 'r', linestyle = '--')
                    plt.axhline(y = -ci, color = 'r', linestyle = '--')

                    plt.legend(fontsize=15)
                    plt.xlabel('Lag', fontsize=15)
                    plt.ylabel('Autocorrelation', fontsize=15)
                    plt.title(title + '\nDiscrete Autocorrelation', fontsize=20)
                    plt.tight_layout()
                    plt.close()

                    fig.savefig(full_path)
    
    def _calculate_autocorrs(self, D=-1):
        """
        Calculate autocorrelation values for the samples.

        Args:
            - D (int, optional): The number of samples to consider for autocorrelation calculation. If not provided, it defaults to the total number of samples.

        Returns:
            - dict: A dictionary containing autocorrelation values for each chain and overall.

        """
        autocorrs = {}
        for chain_num in range(self.n_chains):
            if self.n_chains > 1:
                samples = self.chain_samples[self.chain_samples['chain'] == chain_num + 1].drop(columns=['chain', 'sample_index'])
            else:
                samples = self.samples

            if D == -1:
                D = int(samples.shape[0])

            autocorrs['chain_' + str(chain_num + 1)] = {}
            for col in samples.columns:
                x = samples[col]
                acf = sm.tsa.acf(x)
                ci = np.sqrt(1 / x.size * (1 + 2 * np.sum(acf)))
                tau = 1 + 2 * sum(acf)
                autocorrs['chain_' + str(chain_num + 1)][col] = {}
                autocorrs['chain_' + str(chain_num + 1)][col]['tau'] = tau
                autocorrs['chain_' + str(chain_num + 1)][col]['Ct'] = acf
                autocorrs['chain_' + str(chain_num + 1)][col]['ci'] = ci

        autocorrs['overall'] = {}
        for param in self.samples.columns:
            tau_overall = np.mean([autocorrs['chain_' + str(x + 1)][param]['tau'] for x in range(self.n_chains)])
            autocorrs['overall'][param] = {}
            autocorrs['overall'][param]['tau'] = tau_overall

        return autocorrs

    def get_summary(self) -> dict:
        """
        Generates and saves the summary of the inference results. This is a JSON file which includes:
            - RMSE, AIC and BIC values
            - The lower, median and upper values for each parameter in each chain
            - The autocorrelation values for each parameter in each chain
            - The lower, median and upper values for each parameter overall
            - The autocorrelation values for each parameter overall
        Returns:
            dict: A dictionary containing the summary of the inference results.
        """
        summary = {}
        full_path = self.results_path + '/instance_' + str(self.instance) + '/summary.json'
        if os.path.exists(full_path):
            with open(full_path) as f:
                summary = json.load(f)
        else:
            summary['RMSE'] = self.RMSE
            summary['AIC'] = self.AIC
            summary['BIC'] = self.BIC
            for chain_num in range(self.n_chains):
                summary['chain_' + str(chain_num + 1)] = {}
                samples = self.chain_samples[self.chain_samples['chain'] == chain_num + 1].drop(columns=['chain', 'sample_index'])
                params_lower = self._get_ag_samples(samples, 0.05)
                params_median = self._get_ag_samples(samples, 0.5)
                params_upper = self._get_ag_samples(samples, 0.95)

                for param in self.samples.columns:
                    summary['chain_' + str(chain_num + 1)][param] = {}
                    
                    summary['chain_' + str(chain_num + 1)][param]['lower'] = params_lower[param]
                    summary['chain_' + str(chain_num + 1)][param]['mean'] = params_median[param]
                    summary['chain_' + str(chain_num + 1)][param]['upper'] = params_upper[param]
                    
                    summary['chain_' + str(chain_num + 1)][param]['tau'] = self.autocorrs['chain_' + str(chain_num + 1)][param]['tau']
                    
                    if type(self.data_processor) == SimDataProcessor and param in self.data_processor.model.fixed_model_params:
                        proposed = params_median[param]
                        actual = self.data_processor.model.fixed_model_params[param]
                        summary['chain_' + str(chain_num + 1)][param]['param_accuracy'] = np.abs(100 * np.abs(proposed - actual) / (self.params_max[param] - self.params_min[param]))

                
            overall_samples = self.samples
            summary['overall'] = {}

            overall_params_lower = self._get_ag_samples(overall_samples, 0.05)
            overall_params_median = self._get_ag_samples(overall_samples, 0.5)
            overall_params_upper = self._get_ag_samples(overall_samples, 0.95)

            for param in self.samples.columns:
                summary['overall'][param] = {}

                summary['overall'][param]['lower'] = overall_params_lower[param]
                summary['overall'][param]['mean'] = overall_params_median[param]
                summary['overall'][param]['upper'] = overall_params_upper[param]
                summary['overall'][param]['tau'] = self.autocorrs['overall'][param]['tau']

                if type(self.data_processor) == SimDataProcessor and param in self.data_processor.model.fixed_model_params:
                    proposed = overall_params_median[param]
                    actual = self.data_processor.model.fixed_model_params[param]
                    summary['overall'][param]['param_accuracy'] = np.abs(np.abs(proposed - actual) / (self.params_max[param] - self.params_min[param]) * 100)

            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                elif isinstance(obj, (np.ndarray, np.generic)):  # Handles numpy arrays and scalars
                    return obj.tolist() if isinstance(obj, np.ndarray) else float(obj)
                elif isinstance(obj, (float, int, str)):  # Already serializable types
                    return obj
                elif hasattr(obj, "tolist"):  # For objects like ArrayImpl
                    return obj.tolist()
                elif hasattr(obj, "item"):  # For numpy scalars
                    return obj.item()
                else:
                    return obj  # Default case, leave it as-is

            # Convert your summary dictionary
            summary_serializable = convert_to_serializable(summary)
            with open(self.results_path + '/instance_' + str(self.instance) + '/summary.json', "w") as fp:
                json.dump(summary_serializable, fp, cls=NumpyEncoder, separators=(', ', ': '), indent=4)

        return summary

    def plot_posterior(self, param_name: str, param_range: list, references: dict = None):
        """
        Plots the posterior distribution for a given parameter. Either a 1D or 2D plot is generated based on the number of dimensions of the parameter. The plot is saved in the results directory.

        Args:
            - param_name (str): The name of the parameter. If the parameter is a 2D parameter, the name should be the name of the parameter is the one given as the key in the inference_params series.
            - param_range (list): The range of the parameter values for the plot. Indicates the maximum and minimum values of the parameter to plot the sampled posterior between. The range should be a list containing two floats or integers in the 1D case and a list of of two lists, each containing two floats or integers in the 2D case.
            - references (dict, optional): A dictionary containing reference values for the parameter. The dictionary must have keys 'labels' and 'vals', which are lists of labels and values respectively. Defaults to None.
        """

        if param_name not in self.inference_params:
            raise Exception('Visualiser - parameter not listed as an inference parameter!')
        if self.inference_params[param_name].n_dims == 1:
            self._plot_one_posterior(param_name, param_range, references)
        elif self.inference_params[param_name].n_dims == 2:
            self._plot_two_posteriors(param_name, param_range, references)
        else:
            raise Exception('Visualiser - visualising more than 2 dimensions is not supported!')
    
    def _plot_one_posterior(self, param_name, param_range, references=None):
        if not isinstance(param_range, list) or len(param_range) != 2 or not all(isinstance(val, (float, int)) for val in param_range):
            raise Exception('Visualiser - Invalid param_range. It should be a list with two floats or integers indicating the maximum and minimum values to plot the prior between.')

        if references:
            if not isinstance(references, dict):
                raise ValueError("Visualiser - References must be a dictionary.")
            if "labels" not in references or "vals" not in references:
                raise ValueError("Visualiser - References dictionary must contain keys 'labels' and 'vals'.")
            if not isinstance(references["labels"], list) or not isinstance(references["vals"], list):
                raise ValueError("Visualiser - References 'labels' and 'vals' must be lists.")
            if len(references["labels"]) != len(references["vals"]):
                raise ValueError("Visualiser - References 'labels' and 'vals' must have the same length.")
            if not all(isinstance(label, str) for label in references["labels"]):
                raise ValueError("Visualiser - References 'labels' must contain only strings.")
            if not all(isinstance(val, (int, float)) for val in references["vals"]):
                raise ValueError("Visualiser - References 'vals' must contain only integers or floats.")

        # Correction to fit estimated value into distribution
        if param_range[0] > self.params_median[param_name]:
            param_range[0] = self.params_median[param_name]
        elif param_range[1] < self.params_median[param_name]:
            param_range[1] = self.params_median[param_name]

        # Distribute self.samples[param_name] into multiple bins of equal size
        bins = np.linspace(param_range[0], param_range[1], num=self.samples.shape[0]//100)
        counts, _ = np.histogram(self.samples[param_name], bins=bins)
        

        plt.figure(figsize=(8, 8))
        dist_plot = plt.plot(bins[:-1], counts, color='k', label='Sampled Posterior Distribution')
        plt.xlabel(param_name, fontsize=15)
        plt.ylabel('Sampled Probability', fontsize=15)
        plt.title('Sampled posterior distribution of ' + param_name, fontsize=20)
        plt.yticks([])

        handles = [dist_plot[0]]

        if references:
            reference_labels = references['labels']
            reference_x = references['vals']
            refs = []

            for i, reference_x_point in enumerate(reference_x):
                label = reference_labels[i]

                ref = plt.axvline(reference_x_point, color='r', linestyle='dotted', label=label)
                refs.append(ref)

            # offset = np.linspace(-np.percentile(log_prior_P,99)/2,np.percentile(log_prior_P,99)/2,len(reference_labels))
            labelLines(refs, zorder=2.5, align=True)

            ref.set_label('References')
            handles.append(ref)

        est_val = plt.axvline(self.params_median[param_name], color='b', linestyle='dashed', label='Estimated Value')
        handles.append(est_val)

        if type(self.data_processor) == SimDataProcessor and param_name in self.data_processor.model.fixed_model_params:
            act_val = plt.axvline(self.data_processor.model.fixed_model_params[param_name], color='g', linestyle='dashed', label='Actual Value')
            handles.append(act_val)

        plt.legend(handles=handles, fontsize=15)

        filename = 'posterior_dist_' + param_name + '.png'
        folder_name = self.results_path + '/instance_' + str(self.instance) + '/posterior_dists'
        full_path = folder_name + '/' + filename
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig(full_path)
        plt.close()

    def _plot_two_posteriors(self, param_name, param_range, references=None):
        plt.figure(figsize=(8, 8))
        handles = []

        if references is not None:
            if not isinstance(references, dict):
                raise ValueError("Visualiser - Invalid references format. Expected references to be a dictionary.")
            if "labels" not in references or "vals" not in references:
                raise ValueError("Visualiser - Invalid references format. Expected references to have keys 'labels' and 'vals'.")
            if not isinstance(references["labels"], list) or not isinstance(references["vals"], list):
                raise ValueError("Visualiser - Invalid references format. Expected 'labels' and 'vals' to be lists.")
            if len(references["labels"]) != len(references["vals"]):
                raise ValueError("Visualiser - Invalid references format. Expected 'labels' and 'vals' to have the same length.")
            reference_labels = references["labels"]
            reference_vals = np.array(references["vals"])
            if not all(isinstance(label, str) for label in references["labels"]):
                raise ValueError("Visualiser - Invalid references format. Expected 'labels' to contain only strings.")
            if not all(isinstance(val, list) and len(val) == 2 and all(isinstance(sub_val,(int,float)) for sub_val in val) for val in references["vals"]):
                raise ValueError("Visualiser - Invalid references format. Expected 'vals' to contain only lists of two floats or integers.")
            
            refs = plt.plot(reference_vals[:,0], reference_vals[:,1], ".r", label="References")
            handles.append(refs[0])
            for i, (reference_x_point, reference_y_point) in enumerate(zip(reference_vals[:,0], reference_vals[:,1])):
                label = reference_labels[i]
                plt.annotate(label, (reference_x_point, reference_y_point), textcoords="offset points", 
                                xytext=(0, 5), ha="center", color="r", zorder=3)

        if not isinstance(param_range, list) or len(param_range) != 2 or not isinstance(param_range[0], list) or not isinstance(param_range[1], list) or len(param_range[0]) != 2 or len(param_range[1]) != 2 or not all(isinstance(val, (float, int)) for val in param_range[0]) or not all(isinstance(val, (float, int)) for val in param_range[1]):
            raise ValueError("Invalid param_range format. Expected param_range to be a list of two lists, each containing two floats or integers.")
        param_1_range = [param_range[0][0], param_range[0][1]]
        param_2_range = [param_range[1][0], param_range[1][1]]

        param_1_name = self.inference_params[param_name].name[0]
        param_2_name = self.inference_params[param_name].name[1]

        if param_1_range[0] > self.params_median[param_1_name]:
            param_1_range[0] = self.params_median[param_1_name]
        elif param_1_range[1] < self.params_median[param_1_name]:
            param_1_range[1] = self.params_median[param_1_name]

        if param_2_range[0] > self.params_median[param_2_name]:
            param_2_range[0] = self.params_median[param_2_name]
        elif param_2_range[1] < self.params_median[param_2_name]:
            param_2_range[1] = self.params_median[param_2_name]

        param_1_linspace = np.linspace(param_1_range[0], param_1_range[1], 100)
        param_2_linspace = np.linspace(param_2_range[0], param_2_range[1], 100)

        x_vals = self.samples[param_1_name]
        y_vals = self.samples[param_2_name]

        counts, _, _ = np.histogram2d(x_vals, y_vals, bins=[100, 100], range=[(param_1_range[0], param_1_range[1]), (param_2_range[0], param_2_range[1])])

        plt.contourf(param_1_linspace, param_2_linspace, counts.T, levels=20, cmap='Greys', vmin = np.percentile(counts,5))
        plt.xlabel(param_1_name, fontsize = 15)
        plt.ylabel(param_2_name, fontsize = 15)
        plt.title('Sampled joint posterior distribution of ' + param_1_name + ' and ' + param_2_name, fontsize = 20)

        est_val = plt.scatter([self.params_median[param_1_name]], [self.params_median[param_2_name]], color = 'w', marker='s', edgecolors='k', label = 'Estimated Value')
        handles.append(est_val)
        
        if self.data_processor and type(self.data_processor) == SimDataProcessor and param_1_name in self.data_processor.model.fixed_model_params and param_2_name in self.data_processor.model.fixed_model_params:
            act_val = plt.scatter([self.data_processor.model.fixed_model_params[param_1_name]], [self.data_processor.model.fixed_model_params[param_2_name]], color = 'w', marker='*', edgecolors='black', label = "Actual Value")
            handles.append(act_val)

        plt.legend(handles = handles, fontsize = 15)

        filename = 'posterior_dist_' + param_1_name + '_' + param_2_name +'.png'
        folder_name = self.results_path + '/instance_' + str(self.instance) + '/posterior_dists'
        full_path = folder_name + '/' + filename
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig(full_path)
        plt.close()

class GPVisualiser(Visualiser):
    """
    A class to visualise the results of a Gaussian Process inference.

    Attributes:
        - results_path (str): The path to the directory where the results are saved.
        - instance (int): The instance number of the inference.
        - params_upper (pd.Series): A Series containing the upper values of the parameters.
        - data_processor (DataProcessor): The DataProcessor object used to process the data.
    """

    def __init__(self, gp_obj):
        """
        Initialises the GPVisualiser object.

        Args:
            - gp_obj (GP): The GaussianProcess object used to perform the inference.
        """
        super().__init__(gp_obj)

        X_test = gp_obj.X_test
        y_test = gp_obj.y_test
        test_predictions, test_std = self.gp_obj.predict(X_test)

        num_kernel_params = self.gp_obj.num_params

        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * test_std**2) + (y_test - test_predictions)**2 / test_std**2)
        self.RMSE = np.sqrt(np.mean((test_predictions-y_test)**2))
        self.AIC = 2*num_kernel_params - log_likelihood
        self.BIC = num_kernel_params*np.log(self.testing_data.shape[0]) - 2*log_likelihood

    def _get_param_string(self, kernel_dict=None):
        """
        Generate a string describing the parameters of the GP.

        Args:
            kernel_dict (dict, optional): Dictionary representation of the kernel.
                                        Defaults to self.gp_obj.optimized_params if None.

        Returns:
            str: A formatted string containing the kernel parameters and variable names.
        """
        if kernel_dict is None:
            kernel_dict = self.gp_obj.optimized_params  # Use the instance's kernel dictionary

        param_str_list = []

        def format_value(value):
            """Format numbers dynamically to improve readability."""
            if isinstance(value, list):
                formatted_vals = []
                for val in value:
                    formatted_vals.append(format_value(val))
                return formatted_vals

            if isinstance(value, (float, int)):
                abs_val = abs(value)
                if abs_val == 0:
                    return "0"
                elif abs_val >= 1:
                    return f"{value:.3g}"  # 3 significant figures for large numbers
                else:
                    return f"{value:.2g}"  # 2 significant figures for small numbers
            return value  # Return as is if not a number

        def get_variable_names(dims):
            """Get variable names from self.independent_variables given dims indices."""
            if isinstance(dims, list) and hasattr(self, "independent_variables"):
                return ", ".join(
                    self.independent_variables[i] for i in dims if i < len(self.independent_variables)
                )
            return "unknown_variable"

        def extract_params(k_dict, prefix=""):
            """Recursive function to extract kernel parameters from nested structures."""
            if "sub_kernels" in k_dict:  # Handle composite kernels (Sum/Product)
                for sub_kernel in k_dict["sub_kernels"]:
                    extract_params(sub_kernel, prefix)

            elif "base_kernel" in k_dict:  # Handle transformed kernels (_KernelTransformer)
                dims = k_dict.get("dims", [])
                extract_params(k_dict["base_kernel"], f"{get_variable_names(dims)} ")

            elif "parameters" in k_dict:  # Handle base kernels (RBF, Matern, etc.)
                kernel_name = k_dict["type"]  # Get base kernel type (e.g., RBF, Matern)
                variables = prefix.strip() if prefix else "unknown_variable"

                for param_name, value in k_dict["parameters"].items():
                    if "bounds" not in param_name:  # Ignore bounds
                        param_str_list.append(f"{kernel_name} ({variables}): {param_name} = {format_value(value)}")

        extract_params(kernel_dict)

        return "\n".join(param_str_list)

    def _make_predictions(self, domain_points):

        medium_pred, std_pred = self.gp_obj.predict(domain_points.values)
        lower_pred = medium_pred - 1.96 * std_pred
        upper_pred = medium_pred + 1.96 * std_pred
        return lower_pred, medium_pred, upper_pred
    
