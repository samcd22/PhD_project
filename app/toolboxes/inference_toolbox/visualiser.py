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

from toolboxes.inference_toolbox.sampler import Sampler
from toolboxes.plotting_toolbox.domain import Domain
from toolboxes.data_processing_toolbox.sim_data_processor import SimDataProcessor

class Visualiser:
    """
    Visualiser class - Used for processing and visualising the results from the sampler

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
    - dependent_variables (list): Dependent variables of the model
    - independent_variables (list): Independent variables of the model
    - inference_params (list): Inference parameters
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
    - autocorrs (dict): Autocorrelations

    Methods:
    - get_traceplots: Checks whether traceplots exist, if not, generates and saves them
    - show_predictions: Outputs the plots for visualising the modelled system
    - get_autocorrs: Generates the autocorrelation plots
    - get_summary: Generates a summary of the results
    - plot_prior: Plots the prior distribution for the inputted parameter

    """

    def __init__(self, sampler: Sampler):
        """
        Initialises the Visualiser class saving all relevant variables and performing some initialising tasks

        Args:
            sampler (Sampler): The sampler object

        Raises:
            Exception: If the sampler has not been run
        """
        if not sampler.sampled:
            raise Exception('Visualiser - sampler has not been run!')
        
        self.n_samples = sampler.n_samples
        self.n_chains = sampler.n_chains
        self.samples = sampler.samples
        self.chain_samples = sampler.chain_samples
        self.fields = sampler.fields

        self.data_processor = sampler.data_processor
        
        self.training_data = sampler.training_data
        self.testing_data = sampler.testing_data
        
        self.model_func = sampler.model_func
        self.dependent_variables = sampler.dependent_variables
        self.independent_variables = sampler.independent_variables
        self.inference_params = sampler.inference_params

        self.likelihood_func= sampler.likelihood_func
        
        self.results_path = sampler.results_path
        self.instance = sampler.instance

        self.params_min = self.samples.min()
        self.params_lower = self._get_ag_samples(self.samples, 0.05)
        self.params_median = self._get_ag_samples(self.samples, 0.5)
        self.params_upper = self._get_ag_samples(self.samples, 0.95)
        self.params_max = self.samples.max()

        test_predictions = self.model_func(self.params_median, self.testing_data)
        test_measured = self.testing_data[self.dependent_variables[0]]
        log_likelihood = self.likelihood_func(test_predictions, self.params_median).log_prob(test_measured).sum()
        
        self.RMSE = np.sqrt(np.mean((test_predictions-test_measured)**2))
        self.AIC = 2*self.params_median.size - log_likelihood
        self.BIC = self.params_median.size*np.log(self.testing_data.shape[0]) - 2*log_likelihood

        self.autocorrs = self._calculate_autocorrs()

    def get_traceplots(self):
        """
        Generates and saves the traceplots from the sampled results
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
            x (np.ndarray): Input samples
            xnames (list): Names of the variables
            title (str): Title of the traceplot

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


    def show_predictions(self, domain: Domain, plot_name: str, plot_type: str = '3D', title: str = None):
        """
        Outputs the plots for visualising the modelled system based on the concluded lower, median and upper bound parameters and an inputted domain

        Args:
            domain (Domain): The domain object
            plot_name (str): Name of the plot for saving purposes
            plot_type (str, optional): Type of plot. Defaults to '3D'. Options are:
                - '3D': 3D plot
                - '2D_slice': 2D slice plot
            title (str, optional): Overall title of the plot. Defaults to None.

        """
        if plot_type == '3D':
            if domain.n_dims != 3:
                raise Exception('Visualiser - domain does not have the correct number of spatial dimensions!')
            
            points = domain.create_domain()

            results_df = pd.DataFrame({})
            for i, var in enumerate(self.independent_variables):
                results_df[var] = points[:,i].flatten()

            lower_res = self.model_func(self.params_lower, results_df)
            mean_res = self.model_func(self.params_median, results_df)
            upper_res = self.model_func(self.params_upper, results_df)

            results_df['lower_res'] = lower_res
            results_df['mean_res'] = mean_res
            results_df['upper_res'] = upper_res
            
            self._threeD_plots(results_df, plot_name, title=title)

        elif plot_type == '2D_slice':
            if domain.n_dims != 3:
                raise Exception('Visualiser - domain does not have the correct number of spatial dimensions!')
            count = 0
            for slice_name in [indep_var + '_slice' for indep_var in self.independent_variables]:
                if slice_name in domain.domain_params:
                    count+=1
                    points = domain.create_domain_slice(slice_name)

                    results_df = pd.DataFrame({})
                    for i, var in enumerate(self.independent_variables):
                        results_df[var] = points[:,i].flatten()

                    lower_res = self.model_func(self.params_lower, results_df)
                    mean_res = self.model_func(self.params_median, results_df)
                    upper_res = self.model_func(self.params_upper, results_df)

                    results_df['lower_res'] = lower_res
                    results_df['mean_res'] = mean_res
                    results_df['upper_res'] = upper_res

                    self._twoD_slice_plots(results_df, plot_name,  slice_name = slice_name, title=title)

            if count == 0:
                raise Exception('No slice parameter inputted')
            
        elif plot_type == '1D':
            if domain.n_dims != 1:
                raise Exception('Visualiser - domain does not have the correct number of spatial dimensions!')
            points = domain.create_domain()

            results_df = pd.DataFrame({})
            for i, var in enumerate(self.independent_variables):
                if points.ndim == 1:
                    if len(self.independent_variables) == 1:
                        results_df[var] = points
                    else:
                        raise Exception('Visualiser - domain does not have the correct number of spatial dimensions!')
                else:
                    results_df[var] = points[:,i].flatten()

            lower_res = self.model_func(self.params_lower, results_df)
            mean_res = self.model_func(self.params_median, results_df)
            upper_res = self.model_func(self.params_upper, results_df)

            results_df['lower_res'] = lower_res
            results_df['mean_res'] = mean_res
            results_df['upper_res'] = upper_res

            self._oneD_plots(results_df, plot_name, title=title)
        else:
            raise Exception('Visualiser - plot type not recognised!')
        
    def _oneD_plots(self, results, name, title=None):
        """
        Plotting function for 1D plots

        Args:
            results (pd.DataFrame): Results dataframe
            name (str): Name of the plot for saving purposes
            title (str, optional): Title of the plot. Defaults to None.
        """
        results[self.independent_variables[0]]
        lower_res = results.lower_res
        mean_res = results.mean_res
        upper_res = results.upper_res

        full_path = self.results_path + '/instance_' + str(self.instance) + '/' + name + '_1D_scatter.png'
        if not os.path.exists(full_path):
            fin_alpha = len(self.params_median)*0.75

            y = 12
            r = 0.8
            alpha = (1-r)*y

            delta_alpha = fin_alpha - alpha

            y_prime = y + delta_alpha
            r_prime = r*y/y_prime

            r_prime_frac = Fraction(r_prime)

            fig = plt.figure(constrained_layout = True, figsize = (24,y_prime))
            spec = GridSpec(2, 6, figure = fig, height_ratios= [r_prime_frac.numerator, r_prime_frac.denominator - r_prime_frac.numerator])

            ax1 = fig.add_subplot(spec[0,:2])
            ax2 = fig.add_subplot(spec[0,2:4])
            ax3 = fig.add_subplot(spec[0,4:])
            ax4 = fig.add_subplot(spec[1,:3])
            ax5 = fig.add_subplot(spec[1,3:])
                
            # Defines the lower bound subplot
            ax1.plot(results[self.independent_variables[0]], lower_res, label='Predicted ' + self.dependent_variables[0])
            ax1.set_title('Generated by the lower bound parameters', fontsize = 20)
            ax1.set_xlabel(self.independent_variables[0])
            ax1.set_ylabel(self.dependent_variables[0])

            # Defines the median subplot
            ax2.plot(results[self.independent_variables[0]], mean_res, label='Predicted ' + self.dependent_variables[0])
            ax2.set_title('Generated by the median parameters', fontsize = 20)
            ax2.set_xlabel(self.independent_variables[0])
            ax2.set_ylabel(self.dependent_variables[0])

            # Defines the upper subplot
            ax3.plot(results[self.independent_variables[0]], upper_res, label='Predicted ' + self.dependent_variables[0])
            ax3.set_title('Generated by the upper bound parameters', fontsize = 20)
            ax3.set_xlabel(self.independent_variables[0])
            ax3.set_ylabel(self.dependent_variables[0])

            ax1.scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.dependent_variables[0]], s = 20, c = 'r', label='Measured ' + self.dependent_variables[0] + ' test values')
            ax2.scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.dependent_variables[0]], s = 20, c = 'r', label='Measured ' + self.dependent_variables[0] + ' test values')
            ax3.scatter(self.testing_data[self.independent_variables[0]],self.testing_data[self.dependent_variables[0]], s = 20, c = 'r', label='Measured ' + self.dependent_variables[0] + ' test values')
            
            ax1.legend()
            ax2.legend()
            ax3.legend()

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

            # Creates an axis to display the parameter values
            param_string = self._get_param_string()
            ax4.text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
            ax4.set_xticks([])
            ax4.set_yticks([])

            param_accuracy_string = self._get_param_accuracy_string()

            # Creates an axis to display the sampling information
            ax5.text(0.5,0.5, RMSE_string + ',   ' + AIC_string + ',   ' + BIC_string + '\n' + param_accuracy_string, fontsize = 30, va = "center", ha = 'center')
            ax5.set_xticks([])
            ax5.set_yticks([])

            # Defines the overall title, including the range of values for each plot
            fig.suptitle(title, fontsize = 40)

            # Saves the figures if required
            if not os.path.exists(full_path):
                fig.savefig(full_path)
            plt.close()
                              

    def _threeD_plots(self, results, name, q=10, title=None):
        """
        Plotting function for 3D plots

        Args:
            results (pd.DataFrame): Results dataframe
            name (str): Name of the plot for saving purposes
            q (int, optional): Number of bins i.e. number of figures generated. Defaults to 10.
            title (str, optional): Title of the plot. Defaults to None.
        """
        X = results[self.independent_variables[0]]
        Y = results[self.independent_variables[1]]
        Z = results[self.independent_variables[2]]
        lower_res = results.lower_res
        mean_res = results.mean_res
        upper_res = results.upper_res

        lower_bin_nums = pd.qcut(lower_res,q, labels = False, duplicates='drop')

        mean_bin_nums = pd.qcut(mean_res,q, labels = False, duplicates='drop')
        mean_bin_labs = np.array(pd.qcut(mean_res,q, duplicates='drop').unique())

        upper_bin_nums = pd.qcut(upper_res,q, labels = False, duplicates='drop')

        lower_conc_and_bins = pd.DataFrame([X, Y, Z, lower_res, lower_bin_nums]).T
        lower_conc_and_bins.columns=[self.independent_variables[0],self.independent_variables[1], self.independent_variables[2], 'val', 'bin']

        mean_conc_and_bins = pd.DataFrame([X, Y, Z, mean_res, mean_bin_nums]).T
        mean_conc_and_bins.columns=[self.independent_variables[0],self.independent_variables[1], self.independent_variables[2], 'val', 'bin']

        upper_conc_and_bins = pd.DataFrame([X, Y, Z, upper_res, upper_bin_nums]).T
        upper_conc_and_bins.columns=[self.independent_variables[0],self.independent_variables[1], self.independent_variables[2], 'val', 'bin']

        # Define min and max values for colorbar
        min_val = np.percentile([lower_res, mean_res, upper_res],10)
        max_val = np.percentile([lower_res, mean_res, upper_res],90)

        # Calculates the percentage differances and RMSE if the test points are set to be included
        lower_test_pred = self.model_func(self.params_lower, self.testing_data)
        lower_test_measured = self.testing_data[self.dependent_variables[0]]
        lower_percentage_difference = 2*np.abs(lower_test_measured-lower_test_pred)/(lower_test_measured + lower_test_pred) * 100

        mean_test_pred = self.model_func(self.params_median, self.testing_data)
        mean_test_measured = self.testing_data[self.dependent_variables[0]]
        mean_percentage_difference = 2*np.abs(mean_test_measured-mean_test_pred)/(mean_test_measured + mean_test_pred) * 100

        upper_test_pred = self.model_func(self.params_upper, self.testing_data)
        upper_test_measured = self.testing_data[self.dependent_variables[0]]
        upper_percentage_difference = 2*np.abs(upper_test_measured-upper_test_pred)/(upper_test_measured + upper_test_pred) * 100

        # Creates a directory for the instance if the save parameter is selected
        if not os.path.exists(self.results_path + '/instance_' + str(self.instance) + '/' + str(name) + '_3D_scatter/figures'):
            os.makedirs(self.results_path + '/instance_' + str(self.instance) + '/' + str(name) + '_3D_scatter/figures')
        
        # Loops through each bin number and generates a figure with the bin data 
        for bin_num in np.sort(np.unique(mean_bin_nums)):
            fig_name = 'fig_' + str(bin_num + 1) + '_of_' + str(np.max(mean_bin_nums + 1)) + '.png'
            full_path = self.results_path + '/instance_' + str(self.instance) + '/' + name + '_3D_scatter' + '/figures/' + fig_name
            if not os.path.exists(full_path):
                lower_bin_data = lower_conc_and_bins[lower_conc_and_bins['bin'] >= bin_num]
                mean_bin_data = mean_conc_and_bins[mean_conc_and_bins['bin'] >= bin_num]
                upper_bin_data = upper_conc_and_bins[upper_conc_and_bins['bin'] >= bin_num]

                fin_alpha = len(self.params_median)*0.75

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
                plot_1 = ax1.scatter(lower_bin_data[self.independent_variables[0]], lower_bin_data[self.independent_variables[1]], lower_bin_data[self.independent_variables[2]], c = lower_bin_data.val, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
                ax1.set_title('Generated by the lower bound parameters', fontsize = 20)
                ax1.set_xlabel(self.independent_variables[0])
                ax1.set_ylabel(self.independent_variables[1])
                ax1.set_zlabel(self.independent_variables[2])
                ax1.set_xlim(np.min(X), np.max(X))
                ax1.set_ylim(np.min(Y), np.max(Y))
                ax1.set_zlim(np.min(Z), np.max(Z))

                # Defines the mean subplot
                ax2.scatter(mean_bin_data[self.independent_variables[0]], mean_bin_data[self.independent_variables[1]], mean_bin_data[self.independent_variables[2]], c = mean_bin_data.val, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
                ax2.set_title('Generated by the mean parameters', fontsize = 20)
                ax2.set_xlabel(self.independent_variables[0])
                ax2.set_ylabel(self.independent_variables[1])
                ax2.set_zlabel(self.independent_variables[2])
                ax2.set_xlim(np.min(X), np.max(X))
                ax2.set_ylim(np.min(Y), np.max(Y))
                ax2.set_zlim(np.min(Z), np.max(Z))

                # Defines the upper bound subplot
                ax3.scatter(upper_bin_data[self.independent_variables[0]], upper_bin_data[self.independent_variables[1]], upper_bin_data[self.independent_variables[2]], c = upper_bin_data.val, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
                ax3.set_title('Generated by the upper bound parameters', fontsize = 20)
                ax3.set_xlabel(self.independent_variables[0])
                ax3.set_ylabel(self.independent_variables[1])
                ax3.set_zlabel(self.independent_variables[2])
                ax3.set_xlim(np.min(X), np.max(X))
                ax3.set_ylim(np.min(Y), np.max(Y))
                ax3.set_zlim(np.min(Z), np.max(Z))

                # Generates the test point data on each graph
                pd_min = np.min([lower_percentage_difference, mean_percentage_difference, upper_percentage_difference])
                pd_max = np.min([lower_percentage_difference, mean_percentage_difference, upper_percentage_difference])

                all_test_data = self.testing_data
                all_test_data['lower_p_diff'] = lower_percentage_difference
                all_test_data['mean_p_diff'] = mean_percentage_difference
                all_test_data['upper_p_diff'] = upper_percentage_difference

                all_test_data = all_test_data[all_test_data[self.independent_variables[0]]<np.max(X)]
                all_test_data = all_test_data[all_test_data[self.independent_variables[0]]>np.min(X)]
                all_test_data = all_test_data[all_test_data[self.independent_variables[1]]<np.max(Y)]
                all_test_data = all_test_data[all_test_data[self.independent_variables[1]]>np.min(Y)]
                all_test_data = all_test_data[all_test_data[self.independent_variables[2]]>np.min(Z)]
                all_test_data = all_test_data[all_test_data[self.independent_variables[2]]<np.max(Z)]

                plot_2 = ax1.scatter(all_test_data[self.independent_variables[0]],all_test_data[self.independent_variables[1]],all_test_data[self.independent_variables[2]], s = 20, c = all_test_data['lower_p_diff'], cmap='jet', vmin = pd_min, vmax = pd_max)
                ax2.scatter(all_test_data[self.independent_variables[0]],all_test_data[self.independent_variables[1]],all_test_data[self.independent_variables[2]], s = 20, c = all_test_data['mean_p_diff'], cmap='jet', vmin = pd_min, vmax = pd_max)
                ax3.scatter(all_test_data[self.independent_variables[0]],all_test_data[self.independent_variables[1]],all_test_data[self.independent_variables[2]], s = 20, c = all_test_data['upper_p_diff'], cmap='jet', vmin = pd_min, vmax = pd_max)
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

                # Creates an axis to display the parameter values
                param_string = self._get_param_string()
                ax4.text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
                ax4.set_xticks([])
                ax4.set_yticks([])

                param_accuracy_string = self._get_param_accuracy_string()

                # Creates an axis to display the sampling information
                ax5.text(0.5,0.5, RMSE_string + ',   ' + AIC_string + ',   ' + BIC_string + '\n' + param_accuracy_string, fontsize = 30, va = "center", ha = 'center')
                ax5.set_xticks([])
                ax5.set_yticks([])

                # Defines the two colorbars
                cbar1 = plt.colorbar(plot_1, ax = ax4, location = 'top', shrink = 2)
                cbar2 = plt.colorbar(plot_2, ax = ax5, location = 'top', shrink = 2)

                cbar1.ax.set_title('Predicted Concentration', fontsize = 20)
                cbar2.ax.set_title('Percentage Difference in Test Data', fontsize = 20)


                # Defines the overall title, including the range of values for each plot
                if mean_bin_labs[bin_num].left < 0:
                    left_bound = 0
                else:
                    left_bound = mean_bin_labs[bin_num].left
                fig.suptitle(title + '\nValues for mean plot greater than ' + "{:.2f}".format(left_bound) + '\n', fontsize = 32)

                # Saves the figures if required
                fig_name = 'fig_' + str(bin_num + 1) + '_of_' + str(np.max(mean_bin_nums + 1)) + '.png'
                full_path = self.results_path + '/instance_' + str(self.instance) + '/' + name + '_3D_scatter' + '/figures/' + fig_name
                if not os.path.exists(full_path):
                    fig.savefig(full_path)
                plt.close()
        
        self._animate(name = name + '_3D_scatter')

    def _get_param_accuracy_string(self):
        """
        Returns a string representation of the accuracy of each parameter in the samples.

        The accuracy is calculated as the percentage error between the mean value of the parameter
        in the samples and the corresponding fixed model parameter value.

        Returns:
            str: A string representation of the accuracy of each parameter.
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
        samples (pd.DataFrame): The DataFrame containing the samples.
        q_val (float): The quantile value used to calculate the aggregated sample.

        Returns:
        pd.Series: A Series containing the aggregated samples for each column.
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
            name (str): The name of the animation.
            frame_dur (int, optional): The duration of each frame in milliseconds. Defaults to 500.

        Raises:
            Exception: If the images for animation do not exist.

        Returns:
            None
        """
        folder_name = 'instance_' + str(self.instance) + '/' + name + '/figures'
        gif_name = name + '.gif'
        gif_path = self.results_path + '/' + 'instance_' + str(self.instance) + '/' + name + '/' + gif_name

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
        Plots the prior distribution for a given parameter.

        Args:
            param_name (str): The name of the parameter.
            param_range (list): The range of the parameter values.
            references (dict, optional): A dictionary of reference values for the parameter. Defaults to None.
            show_estimate (bool, optional): Whether to show the estimated median value. Defaults to False.
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
            param_name (str): The name of the parameter.
            param_range (list): A list of two lists, each containing two floats or integers, representing the range of the parameter.
            references (dict, optional): A dictionary containing references for the parameter. 
                It should have keys 'labels' and 'vals', where 'labels' is a list of strings and 'vals' is a list of lists, 
                each containing two floats or integers. Defaults to None.
            show_estimate (bool, optional): Whether to show the estimated median value. Defaults to False.

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
            param_name (str): The name of the parameter.
            param_range (list): A list with two floats or integers indicating the maximum and minimum values to plot the prior between.
            references (dict, optional): A dictionary containing reference values for the parameter. The dictionary must have keys 'labels' and 'vals', which are lists of labels and values respectively. Defaults to None.
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

    def _twoD_slice_plots(self, results, name, slice_name=None, title=None):
        """
        Generates plots for a 2D slice of the 3D modelled system based on the concluded lower, median and upper bound parameters and an inputted domain

        Args:
            results (dict): The results data containing the independent variables and their corresponding values.
            name (str): The name of the plot.
            slice_name (str, optional): The name of the slice variable. Defaults to None.
            title (str, optional): The title of the plot. Defaults to None.

        """
        full_path = self.results_path + '/instance_' + str(self.instance) + '/' + name + '_2D_' + slice_name + '.png'
        if not os.path.exists(full_path):
            slice_var = slice_name.split('_')[0]
            if slice_var in self.independent_variables:
                other_vars = [var for var in self.independent_variables if var != slice_var]

                X = results[other_vars[0]]
                Y = results[other_vars[1]]
                final_title = title + '\nslice at ' + slice_var + ' = ' + str(results[slice_var][0])
                xlab = other_vars[0]
                ylab = other_vars[1]
            else:
                raise Exception('Visualiser - Invalid slice inputted!')

            lower_res = results.lower_res
            mean_res = results.mean_res
            upper_res = results.upper_res

            # Define min and max values for colorbar
            min_val = np.percentile([lower_res, mean_res, upper_res], 10)
            max_val = np.percentile([lower_res, mean_res, upper_res], 90)

            fin_alpha = len(self.params_median) * 0.75

            y = 12
            r = 0.8
            alpha = (1 - r) * y

            delta_alpha = fin_alpha - alpha

            y_prime = y + delta_alpha
            r_prime = r * y / y_prime

            r_prime_frac = Fraction(r_prime)

            fig = plt.figure(constrained_layout=True, figsize=(24, y_prime))
            spec = GridSpec(2, 6, figure=fig, height_ratios=[r_prime_frac.numerator, r_prime_frac.denominator - r_prime_frac.numerator])

            ax1 = fig.add_subplot(spec[0, :2])
            ax2 = fig.add_subplot(spec[0, 2:4])
            ax3 = fig.add_subplot(spec[0, 4:])
            ax4 = fig.add_subplot(spec[1, :3])
            ax5 = fig.add_subplot(spec[1, 3:])

            # Defines the lower bound subplot
            plot_1 = ax1.tricontourf(X, Y, lower_res, vmin=min_val, vmax=max_val, levels=100)
            ax1.set_title('Generated by the lower bound parameters', fontsize=20)
            ax1.set_xlabel(xlab)
            ax1.set_ylabel(ylab)
            ax1.set_xlim(np.min(X), np.max(X))
            ax1.set_ylim(np.min(Y), np.max(Y))

            # Defines the mean subplot
            ax2.tricontourf(X, Y, mean_res, vmin=min_val, vmax=max_val, levels=100)
            ax2.set_title('Generated by the mean parameters', fontsize=20)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel(ylab)
            ax2.set_xlim(np.min(X), np.max(X))
            ax2.set_ylim(np.min(Y), np.max(Y))

            # Defines the upper bound subplot
            ax3.tricontourf(X, Y, upper_res, vmin=min_val, vmax=max_val, levels=100)
            ax3.set_title('Generated by the upper bound parameters', fontsize=20)
            ax3.set_xlabel(xlab)
            ax3.set_ylabel(ylab)
            ax3.set_xlim(np.min(X), np.max(X))
            ax3.set_ylim(np.min(Y), np.max(Y))

            # Generates the test point data on each graph
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


            # Creates an axis to display the parameter values
            param_string = self._get_param_string()
            ax4.text(0.5, 0.5, param_string, fontsize=30, va="center", ha='center')
            ax4.set_xticks([])
            ax4.set_yticks([])

            param_accuracy_string = self._get_param_accuracy_string()

            # Creates an axis to display the sampling information
            ax5.text(0.5, 0.5, RMSE_string + ',   ' + AIC_string + ',   ' + BIC_string + '\n' + param_accuracy_string, fontsize=30, va="center", ha='center')
            ax5.set_xticks([])
            ax5.set_yticks([])

            # Defines the two colorbars
            cbar1 = plt.colorbar(plot_1, ax=ax4, location='top', shrink=2)

            cbar1.ax.set_title('Predicted ' + self.dependent_variables[0], fontsize=10)

            # Defines the overall title, including the range of values for each plot
            fig.suptitle(final_title, fontsize=32)

            # Saves the figures if required
            plt.close()

    def get_autocorrelations(self):
        """
        Generate and save autocorrelation plots for each parameter in the MCMC samples.
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

        Parameters:
            D (int, optional): The number of samples to consider for autocorrelation calculation. 
                               If not provided, it defaults to the total number of samples.

        Returns:
            dict: A dictionary containing autocorrelation values for each chain and overall.

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

    def get_summary(self):
        """
        Generates and saves the summary of the inference results.

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


            with open(self.results_path + '/instance_' + str(self.instance) + '/summary.json', "w") as fp:
                json.dump(summary, fp, cls=NumpyEncoder, separators=(', ', ': '), indent=4)

        return summary

    def plot_posterior(self, param_name: str, param_range: list, references: dict = None):
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