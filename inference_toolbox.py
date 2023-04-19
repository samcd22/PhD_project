import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import os
import json
from numpyencoder import NumpyEncoder
import imageio
from matplotlib.gridspec import GridSpec

class Parameter:
    prior_params = pd.Series({},dtype='float64')
    val = 0
    def __init__(self, init_val, step_select = "" ,step_size = 1, prior_select = ""):
        self.init_val = init_val
        self.val = init_val
        self.step_select = step_select
        self.step_size = step_size
        self.prior_select = prior_select
        
    def add_prior_param(self, name, val):
        self.prior_params[name] = val
        
    # Step Function
    def get_step_function(self):
        # Probability of step
        def log_p_step_multivariate_gaussian(val, mu):
            return stats.multivariate_normal.logpdf(val, mean=mu, cov=self.step_size**2)
        
        def log_p_step_gamma(val, mu):
            beta = mu/self.step_size**2
            a = mu**2/self.step_size**2
            return stats.gamma.logpdf(val, a, scale=1/beta)
        
        # The step itself
        def step_multivariate_positive_gaussian(mu):
            stepped_val = -1
            while stepped_val <= 0:
                stepped_val = stats.multivariate_normal.rvs(mean=mu,cov=self.step_size**2)
            return stepped_val
        
        def step_multivariate_gaussian(mu):
            return stats.multivariate_normal.rvs(mean=mu,cov=self.step_size**2)
        
        def step_gamma(mu):
            beta = mu/self.step_size**2
            a = mu**2/self.step_size**2
            return stats.gamma.rvs(a,scale=1/beta)
        
        
        if self.step_select == "positive gaussian":
            return log_p_step_multivariate_gaussian, step_multivariate_positive_gaussian
        
        elif self.step_select == 'gamma':
            return log_p_step_gamma, step_gamma
        
        elif self.step_select == 'gaussian':
            return log_p_step_multivariate_gaussian, step_multivariate_gaussian
            

    # Priors
    def get_log_prior(self):
        def log_gaussian_prior(val):
            return -(val-self.prior_params.mu)**2/(2*self.prior_params.sigma**2)
        def log_gamma_prior(val):
            return (self.prior_params.k - 1)*np.log(val)-val/self.prior_params.theta
        def no_prior(val):
            return 0

        if self.prior_select == "gaussian":
            return log_gaussian_prior
        elif self.prior_select == "gamma":
            return log_gamma_prior
        elif self.prior_select == "no prior":
            return no_prior
        
    def copy(self):
        return Parameter(self.val, self.step_select, self.step_size, self.prior_select)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        
class Model:
    def __init__(self, model_select):
        self.model_params = pd.Series({},dtype='float64')
        self.model_select = model_select

    def add_model_param(self,name,val):
        self.model_params[name] = val
    
    # Model Function
    def get_model(self):
        def GPM(params, x, y, z):
            a = params.a.val
            b = params.b.val
            Q = params.Q.val
            H = self.model_params.H
            u = self.model_params.u
            tmp = 2*a*x**b
            
            return Q / (tmp*np.pi*u)*(np.exp(-(y**2)/tmp))*(np.exp(-(z-H)**2/tmp)+np.exp(-(z+H)**2/tmp))

        if self.model_select == "GPM":
            return GPM
        
        def GPM_norm(params, x, y, z):
            a = params.a.val
            b = params.b.val
            Q = params.Q.val
            H = self.model_params.H
            tmp = 2*a*x**b
            
            return Q / (tmp*np.pi)*(np.exp(-(y**2)/tmp))*(np.exp(-(z-H)**2/tmp)+np.exp(-(z+H)**2/tmp))

        if self.model_select == "GPM_norm":
            return GPM_norm
        
        def GPM_alt_norm(params, x, y, z):
            I_y = params.I_y.val
            I_z = params.I_z.val
            Q = params.Q.val
            H = self.model_params.H
            return Q/(np.pi*I_y*I_z*x**2)*np.exp(-y**2/(2*I_y**2*x**2))*(np.exp(-(z-H)**2/(2*I_z**2*x**2))+np.exp(-(z+H)**2/(2*I_z**2*x**2)))
        
        if self.model_select == "GPM_alt_norm":
            return GPM_alt_norm

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



# Likelihood Function
class Likelihood:
    def __init__(self, likelihood_select):
        self.likelihood_params = pd.Series({},dtype='float64')
        self.likelihood_select = likelihood_select

    def add_likelihood_param(self,name,val):
        self.likelihood_params[name] = val
    
    def get_log_likelihood_func(self):
        def gaussian_log_likelihood_fixed_sigma(modeled_vals, measured_vals):
            return -np.sum((modeled_vals-measured_vals)**2/(2*self.likelihood_params.sigma**2)) - modeled_vals.size*np.log(np.sqrt(2*np.pi)*self.likelihood_params.sigma)

        def gaussian_log_likelihood_hetroscedastic_fixed_sigma(modeled_vals, measured_vals):
            res = abs(modeled_vals-measured_vals)
            trans_res = ((res+self.likelihood_params.lambda_2)**self.likelihood_params.lambda_1-1)/self.likelihood_params.lambda_1
            return -sum(trans_res**2)/(2*self.likelihood_params.lambda_1**2*self.likelihood_params.sigma**2)
        
        if self.likelihood_select == "gaussian_fixed_sigma":
            return gaussian_log_likelihood_fixed_sigma
        
        if self.likelihood_select == "gaussian_hetroscedastic_fixed_sigma":
            return gaussian_log_likelihood_hetroscedastic_fixed_sigma
        
        def log_gaussian_log_likelihood_fixed_sigma(model_vals,measured_vals):
            return 0
    
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

class Sampler:
    def __init__(self, params, model, likelihood, data, joint_pdf = True):
        self.current_params = params
        self.proposed_params = self.copy_params(params)
        self.model = model
        self.model_func = model.get_model()
        self.likelihood = likelihood
        self.likelihood_func = likelihood.get_log_likelihood_func()
        self.data = data
        self.joint_pdf = joint_pdf

    def get_hyperparams(self):
        self.hyperparams = {}
        # Adding Parameter related hyperparameters
        self.hyperparams['params'] = {}
        for param_ind in self.current_params.index:
            self.hyperparams['params'][param_ind] = {}
            self.hyperparams['params'][param_ind]['init_val'] = self.current_params[param_ind].init_val
            self.hyperparams['params'][param_ind]['step_func'] = self.current_params[param_ind].step_select
            self.hyperparams['params'][param_ind]['step_size'] = self.current_params[param_ind].step_size
            self.hyperparams['params'][param_ind]['prior_func'] = self.current_params[param_ind].prior_select
            self.hyperparams['params'][param_ind]['prior_params'] = {}
            for prior_param_ind in self.current_params[param_ind].prior_params.index:
                self.hyperparams['params'][param_ind]['prior_params'][prior_param_ind] = self.current_params[param_ind].prior_params[prior_param_ind]

        # Adding Model related hyperparameters
        self.hyperparams['model'] = {}
        self.hyperparams['model']['model_func'] = self.model.model_select
        self.hyperparams['model']['model_params'] = {}
        for model_param_ind in self.model.model_params.index:
            self.hyperparams['model']['model_params'][model_param_ind] = self.model.model_params[model_param_ind]
        
        # Adding Likelihood related hyperparameters
        self.hyperparams['likelihood'] = {}
        self.hyperparams['likelihood']['likelihood_func'] = self.likelihood.likelihood_select
        self.hyperparams['likelihood']['likelihood_params'] = {}
        for likelihood_param_ind in self.likelihood.likelihood_params.index:
            self.hyperparams['likelihood']['likelihood_params'][likelihood_param_ind] = self.likelihood.likelihood_params[likelihood_param_ind]
        
        return self.hyperparams
        
    def copy_params(self,params):
        new_params = params.copy()
        for ind in params.index:
            new_params[ind] = new_params[ind].copy()
        return new_params
    
    def accept_params(self, current_log_priors, proposed_log_priors, step_forward_log_prob, step_backward_log_prob):
        # Calculate the log posterior of the current parameters
        curr_modelled_vals = self.model_func(self.current_params,self.data['x'],self.data['y'],self.data['z'])
        curr_log_lhood = self.likelihood_func(curr_modelled_vals, self.data['Concentration'])
        curr_log_posterior = curr_log_lhood + current_log_priors
        
         # Calculate the log posterior of the proposed parameters
        prop_modelled_vals = self.model_func(self.proposed_params,self.data['x'],self.data['y'],self.data['z'])
        prop_log_lhood = self.likelihood_func(prop_modelled_vals, self.data['Concentration'])
        prop_log_posterior = prop_log_lhood + proposed_log_priors
        

        # Acceptance criteria
        alpha = np.exp(prop_log_posterior - curr_log_posterior + step_backward_log_prob - step_forward_log_prob)

        # Acceptance criteria.
        if np.random.uniform(low = 0, high = 1) < np.min([1,alpha]):
            self.current_params = self.copy_params(self.proposed_params)
            return self.copy_params(self.proposed_params), 1
        else:
            # print(prop_log_lhood, prop_log_posterior, step_backward_log_prob, step_forward_log_prob)
            # print([x.val for x in self.proposed_params])
            # print([x.val for x in self.current_params])
            # print('\n')

            return self.copy_params(self.current_params), 0
    
    def sample_one(self):
        current_log_priors = []
        proposed_log_priors = []
        step_forward_log_probs = []
        step_backward_log_probs = []

        for i in range(self.current_params.size):
            # Define current parameter
            current_param = self.current_params[i]
            proposed_param = current_param.copy()
            
            # Get functions
            step_log_prob, step_function = current_param.get_step_function()
            log_prior_func = current_param.get_log_prior()
            
            # Step to proposed parameter
            proposed_param.val = step_function(current_param.val)
            step_forward_log_probs.append(step_log_prob(proposed_param.val, current_param.val))
            step_backward_log_probs.append(step_log_prob(current_param.val, proposed_param.val))

            # Add to series of proposed parameters
            self.proposed_params[i] = proposed_param
                        
            # Create a list of log prior probabilities from each current and proposed parameter
            current_log_priors.append(log_prior_func(current_param.val))
            proposed_log_priors.append(log_prior_func(proposed_param.val))

        # print(step_forward_log_probs)
        # print(step_backward_log_probs)

            # if proposed_param.val <0:
            #     print(current_param.val)
            #     print(proposed_param.val)
            #     raise Exception("Number below 0")
            
            # Can include non joint PDF here
            
        if self.joint_pdf:
            return self.accept_params(sum(current_log_priors), sum(proposed_log_priors), sum(step_forward_log_probs), sum(step_backward_log_probs))
            
    def check_data_exists(self):
        data_path = 'results/inference'
        data_exists = False
        for instance_folder in os.listdir(data_path):
            folder_path = data_path + '/' + instance_folder
            f = open(folder_path + '/hyperparams.json')
            instance_hyperparams = json.load(f)
            f.close()
            if self.hyperparams == instance_hyperparams:
                data_exists = True
                print(instance_folder)
            return data_exists
            
    def sample_all(self, n_samples):
        data_exists = self.check_data_exists()
        acceptance_rate = 0
        samples = []
        accepted = []
        if not data_exists:
            for i in range(1,n_samples+1):
                if (i % 1000 == 0):
                    print('Running sample ' + str(i) + '...')    # Print progress every 1000th sample.
                sample, accept = self.sample_one()
                accepted.append(accept)
                samples.append(sample)
            acceptance_rate = sum(accepted)/len(accepted)*100
            samples = pd.DataFrame(samples)

        return samples, acceptance_rate
    
    def get_mean_samples(self,samples):
        if type(samples) == list and samples == []:
            means = []
        else: 
            means = pd.Series({},dtype='float64')
            for col in samples.columns:
                mean = samples[col].apply(lambda x: x.val).mean()
                means[col] = Parameter(mean)
        return means
    
    def get_lower_samples(self,samples):
        if type(samples) == list and samples == []:
            lowers = []
        else: 
            lowers = pd.Series({},dtype='float64')
            for col in samples.columns:
                formatted_samples = samples[col].apply(lambda x: x.val)
                lower = np.quantile(formatted_samples, 0.05)
                lowers[col] = Parameter(lower)
        return lowers
        
    def get_upper_samples(self,samples):
        if type(samples) == list and samples == []:
            uppers = []
        else: 
            uppers = pd.Series({},dtype='float64')
            for col in samples.columns:
                formatted_samples = samples[col].apply(lambda x: x.val)
                upper = np.quantile(formatted_samples, 0.95)
                uppers[col] = Parameter(upper)
        return uppers
    
                
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# Visualiser
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
        min_val = np.min([C_lower, C_mean, C_upper])
        max_val = np.max([C_lower, C_mean, C_upper])

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

            lower_bin_data = lower_conc_and_bins[lower_conc_and_bins['bin'] == bin_num]
            mean_bin_data = mean_conc_and_bins[mean_conc_and_bins['bin'] == bin_num]
            upper_bin_data = upper_conc_and_bins[upper_conc_and_bins['bin'] == bin_num]

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
            fig.suptitle('Values for mean plot between ' + "{:.2f}".format(mean_bin_labs[bin_num].right) + ' and ' + "{:.2f}".format(left_bound), fontsize = 32)

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
