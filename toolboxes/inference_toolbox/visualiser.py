import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
from numpyencoder import NumpyEncoder
import imageio.v2 as imageio
from matplotlib.gridspec import GridSpec
from fractions import Fraction
from scipy import stats
from labellines import labelLines
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm

# Visualiser class - Used for processing and visualising the results from the sampler
class Visualiser:
    # Initialises the Visualiser class saving all relevant variables and performing some initialising tasks
    def __init__(self, 
                 test_data,
                 sampler, 
                 model, 
                 previous_instance = -1, 
                 data_path = 'results/inference_results/simulated_data/general_instances', 
                 include_test_points = True, 
                 suppress_prints = False,
                 actual_values = []):
        
        # Save variables
        self.suppress_prints = suppress_prints
        self.test_data = test_data
        self.model = model
        self.model_func = model.get_model()
        self.hyperparams = sampler.hyperparams
        self.data_path = data_path
        self.instance = self.get_instance(previous_instance)
        self.num_chains = 1
        self.include_test_points = include_test_points
        self.fields = sampler.fields
        self.sampler = sampler

        # Save the hyperparameter object
        self.save_hyperparams()

        # Defines some variables for saving and loading the samples
        self.sample_data_generated = True
        self.chain_data_generated = True
        if type(sampler.samples) == list and sampler.samples == []:
            self.sample_data_generated = False

        if type(sampler.chain_samples) == list and sampler.chain_samples == []:
            self.chain_data_generated = False
        
        # Decides whether to load the chain samples or not
        if not self.chain_data_generated:
            self.chain_samples = self.load_samples(chain=True)
        else:
            self.chain_samples = sampler.chain_samples
        self.num_chains = int(np.max(self.chain_samples['chain'].unique()))

        # Decides whether to load the samples or not
        if not self.sample_data_generated:
            self.samples = self.load_samples()
        else:
            self.samples = sampler.samples

        # If parameters' actual values are inputted, then they are saved 
        self.actual_values = {}
        for i, param in enumerate(self.samples.columns):
            if actual_values == []:
                self.actual_values[param] = 'NaN'
            else:
                self.actual_values[param] = actual_values[i]

        # Calculates the lower, median and upper bound parameters from the samples
        self.params_min = self.samples.min()
        self.params_lower = self.get_ag_samples(self.samples, 0.05)
        self.params_mean = self.get_ag_samples(self.samples, 0.5)
        self.params_upper = self.get_ag_samples(self.samples, 0.95)
        self.params_max = self.samples.max()

        # Calculates the Root Mean Squared Error
        self.RMSE = 'n/a'
        if self.include_test_points:
            mean_test_pred_C = self.model_func(self.params_mean, self.test_data['x'], self.test_data['y'], self.test_data['z'])
            mean_test_actual_C = self.test_data['Concentration']
            self.RMSE = np.sqrt(np.mean((mean_test_pred_C-mean_test_actual_C)**2))

        # Calculates other success metrics
        self.log_likelihood = self.get_log_likelihood(self.test_data, self.params_mean)
        self.AIC = 2*self.params_mean.size - self.log_likelihood
        self.BIC = self.params_mean.size*np.log(self.test_data.shape[0]) - 2*self.log_likelihood

        # Calculates the autocorrelations
        self.calculate_autocorrs()

        # Saves the samples
        self.save_samples()

    # Checks whether traceplots exist, if they don't they are generated and saved
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

    # Calculates the log likelihood based on inputted data and parameters and selected likelihood and model functions
    def get_log_likelihood(self, data, params):
        likelihood_func = self.sampler.likelihood_func
        model_func = self.model_func
        mu = model_func(params, data.x, data.y, data.z)
        
        test_vals = data.Concentration

        log_likelihoods = likelihood_func(mu, params).log_prob(test_vals)
        return np.sum(log_likelihoods)

    # Generates a traceplot based on inputted samples
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

    # Outputs the plots for visualising the modelled system based on the concluded lower, median and upper bound parameters and an inputted domain
    # There are multiple ways of visualising these results
    def visualise_results(self, domain, name, plot_type = '3D', log_results = True, title = 'Concentration of Droplets'):

        # Output plots are 3D
        if plot_type == '3D':

            if domain.n_dims != 3:
                raise Exception('Domain does not have the correct number of spatial dimensions!')
            
            # Generates domain plots
            points = domain.create_3D_domain()

            X, Y, Z = points[:,0], points[:,1], points[:,2]
            C_lower = self.model_func(self.params_lower, X,Y,Z)
            C_mean = self.model_func(self.params_mean, X,Y,Z)
            C_upper = self.model_func(self.params_upper, X,Y,Z)

            results_df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten(), 'C_lower': C_lower, 'C_mean': C_mean,'C_upper': C_upper})

            self.threeD_plots(results_df, name, log_results=log_results, title=title)

        elif plot_type == '2D_slice':
            count = 0
            for slice_name in ['x_slice', 'y_slice', 'z_slice']:
                if slice_name in domain.domain_params:
                    count+=1
                    points = domain.create_2D_slice_domain(slice_name)

                    X, Y, Z = points[:,0], points[:,1], points[:,2]

                    C_lower = self.model_func(self.params_lower, X,Y,Z)
                    C_mean = self.model_func(self.params_mean, X,Y,Z)
                    C_upper = self.model_func(self.params_upper, X,Y,Z)

                    results_df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten(), 'C_lower': C_lower, 'C_mean': C_mean,'C_upper': C_upper})

                    self.twoD_slice_plots(results_df, name,  slice_name = slice_name, log_results=log_results, title=title)

            if count == 0:
                raise Exception('No slice parameter inputted')

    def gamma_setup(self, mu, sigma):
        alpha =  mu**2/sigma**2
        beta = mu/sigma**2
        return stats.gamma(a = alpha, scale = 1/beta).pdf

    def log_norm_setup(self, mu, sigma):
        alpha = np.log(mu) - 0.5*np.log(1+sigma**2/mu)
        beta = np.sqrt(np.log(1+sigma**2/mu))
        def log_norm(x):
            p = 1/(x*beta*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-alpha)**2/(2*beta**2))
            return p
        return log_norm
    
    def multi_var_log_norm_setup(self, mu, cov):
        mu = np.array(mu)
        cov = np.array(cov)
        n_dim = len(mu)
        alpha = np.zeros(n_dim)
        beta = np.zeros((n_dim, n_dim))

        for i in range(n_dim):
            alpha[i] = np.log(mu[i]) - 1/2*np.log(cov[i,i]/mu[i]**2 +1)
            for j in range(len(mu)):
                beta[i,j] = np.log(cov[i,j]/(mu[i]*mu[j])+1)
        def multi_var_log_norm(x):
            n_points = x.shape[0]
            p = np.ones(n_points)
            for i in range(n_points):
                vals = x[i]
                for j in range(n_dim):
                    p[i] *= (2*np.pi)**(-n_dim/2)*np.linalg.det(beta)**(-1/2)*1/vals[j]*np.exp(-1.2*(np.log(vals)-alpha).T@np.linalg.inv(beta)@(np.log(vals)-alpha))
                    p[i] *= 1/vals[j]
            return p
        return multi_var_log_norm
    
    def multi_mode_log_norm_setup(self, mus, sigmas):
        mus = np.array(mus)
        sigmas = np.array(sigmas)
        mixture_size = mus.size
        alphas = np.log(mus) - 0.5*np.log(1+sigmas**2/mus)
        betas = np.sqrt(np.log(1+sigmas**2/mus))
        def multi_mode_log_norm(x):
            p = []
            for i in range(mixture_size):
                p.append(1/(x*betas[i]*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-alphas[i])**2/(2*betas[i]**2)))
            return np.sum(p,axis = 0)/mixture_size
        return multi_mode_log_norm
    
    def multi_mode_multi_var_log_norm_setup(self, mus, covs):
        mus = np.array(mus)
        covs = np.array(covs)
        mixture_size = mus.shape[0]
        n_dim = mus.shape[1]
        alphas = np.zeros((mixture_size, n_dim))
        betas = np.zeros((mixture_size, n_dim, n_dim))

        for mode in range(mixture_size):
            for i in range(n_dim):
                alphas[mode, i] = np.log(mus[mode, i]) - 1/2*np.log(covs[mode, i,i]/mus[mode, i]**2 +1)
                for j in range(n_dim):
                    betas[mode, i,j] = np.log(covs[mode,i,j]/(mus[mode,i]*mus[mode,j])+1)


        def multi_mode_multi_var_log_norm(x):
            n_points = x.shape[0]
            p = np.ones((mixture_size, n_points))
            for mode in range(mixture_size):
                for i in range(n_points):
                    vals = x[i]
                    for j in range(n_dim):

                        p[mode,i] *= (2*np.pi)**(-n_dim/2)*np.linalg.det(betas[mode])**(-1/2)*1/vals[j]*np.exp(-1.2*(np.log(vals)-alphas[mode]).T@np.linalg.inv(betas[mode])@(np.log(vals)-alphas[mode]))
                        p[mode,i] *= 1/vals[j]
            return np.sum(p, axis = 0)/mixture_size
                
        return multi_mode_multi_var_log_norm    
    
    def get_prior_plots(self, prior_plots):
        for i in range(len(prior_plots)):
            prior_plot_info = prior_plots[i]
            keys = prior_plot_info.keys()
            params = [x for x in keys if x != 'references']
            references = None
            if 'references' in prior_plot_info:
                references = prior_plot_info['references']
            if len(params) == 2:
                self.plot_two_priors(params[0], params[1], prior_plot_info[params[0]], prior_plot_info[params[1]], references)
            elif len(params) == 1:
                self.plot_one_prior(params[0], prior_plot_info[params[0]], references)

    def plot_two_priors(self, param_1, param_2, param_1_range, param_2_range, references = None):
        if param_1_range[0] > self.params_mean[param_1]:
            param_1_range[0] = self.params_mean[param_1]
        elif param_1_range[1] < self.params_mean[param_1]:
            param_1_range[1] = self.params_mean[param_1]

        if param_2_range[0] > self.params_mean[param_2]:
            param_2_range[0] = self.params_mean[param_2]
        elif param_2_range[1] < self.params_mean[param_2]:
            param_2_range[1] = self.params_mean[param_2]

        param_1_linspace = np.linspace(param_1_range[0], param_1_range[1], 100)
        param_2_linspace = np.linspace(param_2_range[0], param_2_range[1], 100)
        param_1_mesh, param_2_mesh = np.meshgrid(param_1_linspace, param_2_linspace)
        shape = param_1_mesh.shape

        P = np.zeros(shape)

        multi_var_param = param_1+'_and_'+param_2
        
        if self.hyperparams['params'][multi_var_param]['prior_func'] == 'log_norm':
            prior_dist = self.multi_var_log_norm_setup(self.hyperparams['params'][multi_var_param]['prior_params']['mu'], self.hyperparams['params'][multi_var_param]['prior_params']['cov'])

        # elif self.hyperparams['params'][param_1]['prior_func'] == 'gamma':
        #     prior_dist_1 = self.gamma_setup(self.hyperparams['params'][param_1]['prior_params']['mu'], self.hyperparams['params'][param_1]['prior_params']['sigma'])
        #     prior_dist_2 = self.gamma_setup(self.hyperparams['params'][param_2]['prior_params']['mu'], self.hyperparams['params'][param_2]['prior_params']['sigma'])

        elif self.hyperparams['params'][multi_var_param]['prior_func'] == 'multi_mode_log_norm':
            prior_dist = self.multi_mode_multi_var_log_norm_setup(self.hyperparams['params'][multi_var_param]['prior_params']['mus'], self.hyperparams['params'][multi_var_param]['prior_params']['covs'])
        else:
            raise Exception('Prior distribution not listed!')

        P = prior_dist(np.array([param_1_mesh.flatten(), param_2_mesh.flatten()]).T)
        P = np.reshape(P, shape)

        plt.figure(figsize=(8, 8))
        plt.contourf(param_1_mesh, param_2_mesh, P, levels=30, cmap='plasma', vmin = np.percentile(P,5))
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.title('Product of the distribution of ' + param_1 + ' and ' + param_2)

        handles = []

        if references:
            reference_labels = references['labels']
            reference_x = references[param_1]
            reference_y = references[param_2]

            refs = plt.plot(reference_x,reference_y,'.r', label = 'References')
            handles.append(refs[0])

            for i, (reference_x_point,reference_y_point) in enumerate(zip(reference_x,reference_y)):
                label = reference_labels[i]
                plt.annotate(label, (reference_x_point,reference_y_point), textcoords="offset points", 
                xytext=(0,5), ha='center', color = 'r')

        est_val = plt.scatter([self.params_mean[param_1]], [self.params_mean[param_2]], color = 'k', marker='s', edgecolors='w', label = 'Estimated Value')
        handles.append(est_val)
        if self.actual_values[param_1] !='NaN' and self.actual_values[param_2] !='NaN':
            act_val = plt.scatter([self.actual_values[param_1]], [self.actual_values[param_2]], color = 'orange', marker='*', edgecolors='black', label = "Actual Value")
            handles.append(act_val)

        plt.legend(handles = handles)

        filename = 'prior_dist_' + param_1 + '_' + param_2 +'.png'
        folder_name = self.data_path + '/instance_' + str(self.instance) + '/prior_dists'
        full_path = folder_name + '/' + filename
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig(full_path)
        plt.close()

    def plot_one_prior(self, param, param_range, references = None):
        
        # Correction to fit estimated value into distribution
        if param_range[0] > self.params_mean[param]:
            param_range[0] = self.params_mean[param]
        elif param_range[1] < self.params_mean[param]:
            param_range[1] = self.params_mean[param]
        
        param_linspace = np.linspace(param_range[0], param_range[1], 100)
        
        if self.hyperparams['params'][param]['prior_func'] == 'log_norm':
            prior_dist = self.log_norm_setup(self.hyperparams['params'][param]['prior_params']['mu'], self.hyperparams['params'][param]['prior_params']['sigma'])
        elif self.hyperparams['params'][param]['prior_func'] == 'gamma':
            prior_dist = self.gamma_setup(self.hyperparams['params'][param]['prior_params']['mu'], self.hyperparams['params'][param]['prior_params']['sigma'])
        elif self.hyperparams['params'][param]['prior_func'] == 'multi_mode_log_norm':
            prior_dist = self.multi_mode_log_norm_setup(self.hyperparams['params'][param]['prior_params']['mus'], self.hyperparams['params'][param]['prior_params']['sigmas'])
        else:
            raise Exception('Prior distribution not listed!')
        
        P = prior_dist(param_linspace)
        
        plt.figure(figsize=(8, 8))
        dist_plot = plt.plot(param_linspace, P, color = 'k', label = 'Prior Distribution')
        plt.xlabel(param)
        plt.ylabel('P')
        plt.title('Prior distribution of ' + param)
        plt.yticks([])

        handles = [dist_plot[0]]

        if references:
            reference_labels = references['labels']
            reference_x = references[param]
            refs = []

            for  i, reference_x_point in enumerate(reference_x):
                label = reference_labels[i]

                ref = plt.axvline(reference_x_point, color ='r', linestyle='dotted', label=label)
                refs.append(ref)

            offset = np.linspace(-np.percentile(P,99)/2,np.percentile(P,99)/2,len(reference_labels))
            labelLines(refs, zorder=2.5, align=True, yoffsets=offset)#, yoffsets=offset)
            
            ref.set_label('References')
            handles.append(ref)
        
        est_val = plt.axvline(self.params_mean[param], color = 'b', linestyle = 'dashed', label = 'Estimated Value')
        handles.append(est_val)

        if self.actual_values[param] != 'NaN':
            act_val = plt.axvline(self.actual_values[param], color = 'g', linestyle = 'dashed', label = 'Actual Value')
            handles.append(act_val)

        plt.legend(handles = handles)


        filename = 'prior_dist_' + param + '.png'
        folder_name = self.data_path + '/instance_' + str(self.instance) + '/prior_dists'
        full_path = folder_name + '/' + filename
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig(full_path)
        plt.close()

    def twoD_slice_plots(self, results, name, slice_name = None, log_results=False, title=None):
        full_path = self.data_path + '/instance_' + str(self.instance) + '/' + name + '_2D_' + slice_name + '.png'
        if os.path.exists(full_path):
            print('2D slice plot already exists!')
        else:        
            if slice_name == 'x_slice':
                X = results.y
                Y = results.z
                final_title = title + '\nslice at x = ' + str(results.x[0])
                xlab = 'y'
                ylab = 'z'
            elif slice_name == 'y_slice':
                X = results.x
                Y = results.z
                final_title = title + '\nslice at y = ' + str(results.y[0])
                xlab = 'x'
                ylab = 'z'
            elif slice_name == 'z_slice':
                X = results.x
                Y = results.y
                final_title = title + '\nslice at z = ' + str(results.z[0])
                xlab = 'x'
                ylab = 'y'
            else:
                raise Exception('No slice inputted!')

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

            # Define min and max values for colorbar
            min_val = np.percentile([C_lower, C_mean, C_upper],10)
            max_val = np.percentile([C_lower, C_mean, C_upper],90)

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

            ax1 = fig.add_subplot(spec[0,:2])
            ax2 = fig.add_subplot(spec[0,2:4])
            ax3 = fig.add_subplot(spec[0,4:])
            ax4 = fig.add_subplot(spec[1,:3])
            ax5 = fig.add_subplot(spec[1,3:])
            
            # Defines the lower bound subplot
            plot_1 = ax1.tricontourf(X, Y, results.C_lower, vmin=min_val, vmax=max_val, levels = 100)
            ax1.set_title('Generated by the lower bound parameters', fontsize = 20)
            ax1.set_xlabel(xlab)
            ax1.set_ylabel(ylab)
            ax1.set_xlim(np.min(X), np.max(X))
            ax1.set_ylim(np.min(Y), np.max(Y))

            # Defines the mean subplot
            ax2.tricontourf(X, Y, results.C_mean, vmin=min_val, vmax=max_val, levels = 100)
            ax2.set_title('Generated by the mean parameters', fontsize = 20)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel(ylab)
            ax2.set_xlim(np.min(X), np.max(X))
            ax2.set_ylim(np.min(Y), np.max(Y))

            # Defines the upper bound subplot
            ax3.tricontourf(X, Y, results.C_upper, vmin=min_val, vmax=max_val, levels = 100)
            ax3.set_title('Generated by the upper bound parameters', fontsize = 20)
            ax3.set_xlabel(xlab)
            ax3.set_ylabel(ylab)
            ax3.set_xlim(np.min(X), np.max(X))
            ax3.set_ylim(np.min(Y), np.max(Y))

            # Generates the test point data on each graph
            if self.include_test_points:
                formatter = "{:.2e}" 
                if  np.floor(np.log10(self.RMSE)) < 2: formatter = "{:.2f}"
                RMSE_string = 'RMSE = ' + formatter.format(self.RMSE)
                AIC_string = 'AIC = ' + formatter.format(self.AIC)
                BIC_string = 'BIC = ' + formatter.format(self.BIC)

            else:
                RMSE_string = 'RMSE = n/a'
                AIC_string = 'AIC = n/a'
                BIC_string = 'BIC = n/a'

            # Creates an axis to display the parameter values
            param_string = self.get_param_string()
            ax4.text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
            ax4.set_xticks([])
            ax4.set_yticks([])

            param_accuracy_string = self.get_param_accuracy_string()

            # Creates an axis to display the sampling information
            ax5.text(0.5,0.5, RMSE_string + ',   ' + AIC_string + ',   ' + BIC_string + '\n' + param_accuracy_string, fontsize = 30, va = "center", ha = 'center')
            ax5.set_xticks([])
            ax5.set_yticks([])

            # Defines the two colorbars
            cbar1 = plt.colorbar(plot_1, ax = ax4, location = 'top', shrink = 2)

            cbar1.ax.set_title('Predicted Concentration', fontsize = 10)

            # Defines the overall title, including the range of values for each plot
            fig.suptitle(final_title, fontsize = 32)

            # Saves the figures if required

            fig.savefig(full_path)
            plt.close()

    # Plotting function for 3D plots
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
        if not os.path.exists(self.data_path + '/instance_' + str(self.instance) + '/' + str(name) + '_3D_scatter/figures'):
            os.makedirs(self.data_path + '/instance_' + str(self.instance) + '/' + str(name) + '_3D_scatter/figures')
        
        # Loops through each bin number and generates a figure with the bin data 
        for bin_num in np.sort(np.unique(mean_bin_nums)):
            fig_name = 'fig_' + str(bin_num + 1) + '_of_' + str(np.max(mean_bin_nums + 1)) + '.png'
            full_path = self.data_path + '/instance_' + str(self.instance) + '/' + name + '_3D_scatter' + '/figures/' + fig_name
            if not os.path.exists(full_path):
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
                ax1.set_title('Generated by the lower bound parameters', fontsize = 20)
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('z')
                ax1.set_xlim(np.min(X), np.max(X))
                ax1.set_ylim(np.min(Y), np.max(Y))
                ax1.set_zlim(np.min(Z), np.max(Z))

                # Defines the mean subplot
                ax2.scatter(mean_bin_data.x, mean_bin_data.y, mean_bin_data.z, c = mean_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
                ax2.set_title('Generated by the mean parameters', fontsize = 20)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.set_zlabel('z')
                ax2.set_xlim(np.min(X), np.max(X))
                ax2.set_ylim(np.min(Y), np.max(Y))
                ax2.set_zlim(np.min(Z), np.max(Z))

                # Defines the upper bound subplot
                ax3.scatter(upper_bin_data.x, upper_bin_data.y, upper_bin_data.z, c = upper_bin_data.conc, cmap='jet', vmin=min_val, vmax=max_val, alpha = 0.3, s=1)
                ax3.set_title('Generated by the upper bound parameters', fontsize = 20)
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

                    plot_2 = ax1.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 20, c = lower_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                    ax2.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 20, c = mean_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                    ax3.scatter(self.test_data['x'],self.test_data['y'],self.test_data['z'], s = 20, c = upper_percentage_difference, cmap='jet', vmin = pd_min, vmax = pd_max)
                    formatter = "{:.2e}" 
                    if  np.floor(np.log10(self.RMSE)) < 2: formatter = "{:.2f}"
                    RMSE_string = 'RMSE = ' + formatter.format(self.RMSE)
                    AIC_string = 'AIC = ' + formatter.format(self.AIC)
                    BIC_string = 'BIC = ' + formatter.format(self.BIC)

                    
                else:
                    RMSE_string = 'RMSE = n/a'
                    AIC_string = 'AIC = n/a'
                    BIC_string = 'BIC = n/a'

                # Creates an axis to display the parameter values
                param_string = self.get_param_string()
                ax4.text(0.5,0.5,param_string, fontsize = 30, va = "center", ha = 'center')
                ax4.set_xticks([])
                ax4.set_yticks([])

                param_accuracy_string = self.get_param_accuracy_string()

                # Creates an axis to display the sampling information
                ax5.text(0.5,0.5, RMSE_string + ',   ' + AIC_string + ',   ' + BIC_string + '\n' + param_accuracy_string, fontsize = 30, va = "center", ha = 'center')
                ax5.set_xticks([])
                ax5.set_yticks([])

                # Defines the two colorbars
                cbar1 = plt.colorbar(plot_1, ax = ax4, location = 'top', shrink = 2)
                cbar2 = plt.colorbar(plot_2, ax = ax5, location = 'top', shrink = 2)

                cbar1.ax.set_title('Predicted Concentration', fontsize = 10)
                cbar2.ax.set_title('Percemtage Difference in Test Data', fontsize = 10)


                # Defines the overall title, including the range of values for each plot
                if mean_bin_labs[bin_num].left < 0:
                    left_bound = 0
                else:
                    left_bound = mean_bin_labs[bin_num].left
                fig.suptitle(title + '\nValues for mean plot greater than ' + "{:.2f}".format(left_bound) + '\n', fontsize = 32)

                # Saves the figures if required
                fig_name = 'fig_' + str(bin_num + 1) + '_of_' + str(np.max(mean_bin_nums + 1)) + '.png'
                full_path = self.data_path + '/instance_' + str(self.instance) + '/' + name + '_3D_scatter' + '/figures/' + fig_name
                if not os.path.exists(full_path):
                    fig.savefig(full_path)
                plt.close()
        
        self.animate(name = name + '_3D_scatter')

    # Generates a string of all parameter accuracy results
    def get_param_accuracy_string(self):
        param_accuracy_string_array = []
        for param in self.samples.columns:
            if not self.actual_values[param] == 'NaN':
                percentage_error = 100*np.abs(self.params_mean[param]-self.actual_values[param])/(self.params_max[param] - self.params_min[param])
                param_accuracy_string_array.append(param + ' error = ' + f'{percentage_error:.3}' + '%')
        return ('\n').join(param_accuracy_string_array)

    # Generates a string of all of the parameter results
    def get_param_string(self):
        param_string_array = []
        for param in self.params_mean.index:
            if np.floor(np.log10(self.params_mean[param])) < 2:
                lower_string =  "{:.3f}".format(self.params_lower[param])
                mean_string = "{:.3f}".format(self.params_mean[param])
                upper_string = "{:.3f}".format(self.params_upper[param])
            else:
                lower_string =  "{:.3e}".format(self.params_lower[param])
                mean_string = "{:.3e}".format(self.params_mean[param])
                upper_string = "{:.3e}".format(self.params_upper[param])
            param_string_array.append(param + ' = [' + lower_string + ', ' + mean_string + ', ' + upper_string + ']')

        return ('\n').join(param_string_array)
        
    # Outputs the next available instance number and generates a folder for that instance
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

    # Saves the hyperparameter object
    def save_hyperparams(self):
        with open(self.data_path + '/instance_' + str(self.instance) + '/hyperparams.json', "w") as fp:
            json.dump(self.hyperparams,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)
        
    # Gathers all of the figures under the inputted name, creates an animation of them and saves that animation
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
    
    # Saves the samples and chain samples objects
    def save_samples(self):
        full_path = self.data_path + '/instance_' + str(self.instance) + '/samples.csv'
        if type(self.samples) == list and self.samples == []:
            raise Exception('Samples data is empty!')    
        pd.DataFrame(self.samples).to_csv(full_path, index=False)

        chain_full_path = self.data_path + '/instance_' + str(self.instance) + '/chain_samples.csv'
        if type(self.chain_samples) == list and self.chain_samples == []:
            raise Exception('Samples data is empty!')    
        pd.DataFrame(self.chain_samples).to_csv(chain_full_path, index=False)
        
    # Loads either the chain samples of samples object
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

    # Checks whether autocorrelation plots exist, if they don't they are generated and saved
    def get_autocorrelations(self):
        autocorr_folder = self.data_path + '/instance_' + str(self.instance) + '/autocorrelations'
        if not os.path.exists(autocorr_folder):
            os.mkdir(autocorr_folder)

        for chain in range(self.num_chains):
            for param in self.samples.columns:
                full_path = autocorr_folder + '/autocorrelations_' + param + '_chain_' + str(chain + 1) + '.png'
                
                if self.num_chains == 1:
                    title = 'MCMC autocorrelations for ' + param
                else:
                    title = 'MCMC autocorrelations for ' + param + ', chain ' + str(chain + 1)

                if os.path.exists(full_path):
                    if not self.suppress_prints:
                        print('Autocorrelations plot ' + param + ', chain ' + str(chain + 1) + ' already exists')
                else:
                    ac = self.autocorr_fig(param, chain, title = title)
                    ac.savefig(full_path)
    
    # Generates a autocorrelation plots based on the samples
    def autocorr_fig(self, param, chain_num = 1, title = ''):
        fig = plt.figure(figsize=(6,4))
        autocorrelations = self.autocorrs['chain_' + str(chain_num + 1)][param]['Ct']
        tau = self.autocorrs['chain_' + str(chain_num + 1)][param]['tau']
        ci = self.autocorrs['chain_' + str(chain_num + 1)][param]['ci']
        formatted_tau = "{:.2f}".format(tau)
        plt.bar(range(autocorrelations.size), autocorrelations, label = param + ', tau = ' + formatted_tau)
        plt.axhline(y = ci, color = 'r', linestyle = '--')
        plt.axhline(y = -ci, color = 'r', linestyle = '--')
        
        plt.legend()
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title(title + '\nDiscrete Autocorrelation')
        plt.tight_layout()
        plt.close()

        return fig
    
    # Calculates the autocorrelations based on the samples and saves them to an object
    def calculate_autocorrs(self, D=-1):
        self.autocorrs = {}
        for chain_num in range(self.num_chains):
            if self.num_chains > 1:
                samples = self.chain_samples[self.chain_samples['chain'] == chain_num + 1].drop(columns = ['chain', 'sample_index'])
            else:
                samples = self.samples

            if D == -1:
                D = int(samples.shape[0])
            
            self.autocorrs['chain_' + str(chain_num+1)] = {}
            for col in samples.columns:
                x = samples[col]
                acf = sm.tsa.acf(x)
                ci = np.sqrt(1/x.size*(1+2*np.sum(acf)))
                tau = 1 + 2*sum(acf)
                self.autocorrs['chain_' + str(chain_num+1)][col] = {}
                self.autocorrs['chain_' + str(chain_num+1)][col]['tau'] = tau
                self.autocorrs['chain_' + str(chain_num+1)][col]['Ct'] = acf
                self.autocorrs['chain_' + str(chain_num+1)][col]['ci'] = ci
        
        self.autocorrs['overall'] = {}
        for param in self.samples.columns:
            tau_overall = np.mean([self.autocorrs['chain_' + str(x + 1)][param]['tau'] for x in range(self.num_chains)])
            self.autocorrs['overall'][param] = {}
            self.autocorrs['overall'][param]['tau'] = tau_overall
    
    # Outputs the specified quantile of the parameter samples
    def get_ag_samples(self,samples, q_val):
        ags = pd.Series({},dtype='float64')
        for col in samples.columns:
            param_samples = samples[col]
            ag = np.quantile(param_samples, q_val)
            ags[col] = ag
        return ags
    
    # Outputs the fields object from the sampler
    def get_fields(self, chain_num):
        output = {}
        if self.fields.keys() != []:
            for key in self.fields.keys():
                field_output = self.fields[key][chain_num]
                if key == 'diverging':
                    field_output = sum(field_output.tolist())

                output[key] = field_output
        return output

    # Generates an object which summarises the results of the sampler and saves it
    def get_summary(self):
        summary = {}
        full_path = self.data_path + '/instance_' + str(self.instance) + '/summary.json'
        if os.path.exists(full_path):
            f = open(full_path)
            summary = json.load(f)
            f.close()
        else:
            summary['RMSE'] = self.RMSE
            summary['AIC'] = self.AIC
            summary['BIC'] = self.BIC
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
                        summary['chain_' + str(chain_num + 1)][param]['param_accuracy'] = np.abs(100*np.abs(proposed-actual)/(self.params_max[param] - self.params_min[param]))

                
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
                summary['overall'][param]['tau'] = self.autocorrs['overall'][param]['tau']

                if self.actual_values[param] !='NaN':
                    summary['overall'][param]['param_accuracy'] = np.abs(np.abs(overall_params_mean[param]-self.actual_values[param])/(self.params_max[param] - self.params_min[param])*100)


            with open(self.data_path + '/instance_' + str(self.instance) + '/summary.json', "w") as fp:
                json.dump(summary,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

        return summary