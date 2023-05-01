# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from inference_toolbox.parameter import Parameter
from inference_toolbox.model import Model
from inference_toolbox.likelihood import Likelihood
from inference_toolbox.sampler import Sampler
from inference_toolbox.visualiser import Visualiser
from inference_toolbox.domain import Domain

from data_processing.normaliser import Normaliser
from data_processing.box_gridder import BoxGridder
from gaussian_processor import GaussianProcessor

# %%
# # Import and select data.
# all_data = pd.read_csv('data/total_data.csv')

# # Import and select metadata.
# metadata = pd.read_csv('data/data_summary.csv',
#     usecols = ['Experiment', 'Wind_Dir', 'WindSpeed', 'boat.lat', 'boat.lon']
# )

# %%

# normaliser = Normaliser(all_data, metadata)

# all_experiments = normaliser.get_experiments_list()

# selected_experiments = np.delete(all_experiments, np.where(all_experiments == 'Control'))

# normalised_data = normaliser.normalise_data(selected_experiments)


# %%
# box_gridder = BoxGridder(normalised_data)

# averaged_df = box_gridder.get_averages([200,200,50],False)

# # box_gridder.visualise_average_data(averaged_df)

# # averaged_df

# %%
# gp = GaussianProcessor(averaged_df, kernel = 'matern_white', data_norm='sqrt')
# training_data, test_data = gp.train_test_split()
# print(training_data)
# gp.train_gp(training_data)
# gp.test(test_data)
# grid = box_gridder.get_grid([10,10,10])
# gp.predict_from_gp(grid, threeD=True, save=True, log_results=True)
# gp.animate() 

# %%
# Get dummy data
dummy_data = pd.read_csv('data/dummy_data.csv')

# %%
# training_data, testing_data = train_test_split(averaged_df, test_size=0.2)
training_data, testing_data = train_test_split(dummy_data, test_size=0.2)

# %%
# Initialize parameter series
params = pd.Series({},dtype='float64')

# Parameter Assignment
I_y = Parameter(init_val = 0.1, step_select = "gamma", step_size = 0.01, prior_select = "no prior")
I_y.add_prior_param("mu",0.1)
I_y.add_prior_param("sigma",0.1)
params['I_y'] = I_y

I_z = Parameter(init_val = 0.1, step_select = "gamma", step_size = 0.01, prior_select = "no prior")
I_z.add_prior_param("mu",0.1)
I_z.add_prior_param("sigma",0.1)
params['I_z'] = I_z

Q = Parameter(init_val = 3e13, step_select = "gamma", step_size = 1e12, prior_select = "no prior")
Q.add_prior_param("mu",3e13)
Q.add_prior_param("sigma",1e13)
params['Q'] = Q

# Model Assignment
model = Model('GPM_alt_norm')
model.add_model_param("H",10)

# Likelihood function assigmnent
likelihood = Likelihood("gamma_fixed_sigma")
likelihood.add_likelihood_param("sigma",1e8)
# likelihood.add_likelihood_param("lambda_1",1)
# likelihood.add_likelihood_param("lambda_2",0.05)

# Initialize the sampler
sampler = Sampler(params, model, likelihood, training_data, show_sample_info = True)
hyperparams = sampler.get_hyperparams()

# Sample the parameters
params_samples, acceptance_rate = sampler.sample_all(1000)

params_mean = sampler.get_mean_samples(params_samples)
params_lower = sampler.get_lower_samples(params_samples)
params_upper = sampler.get_upper_samples(params_samples)

# %%
domain = Domain('cone_from_source_z_limited')
domain.add_domain_param('r', 1000)
domain.add_domain_param('theta', np.pi/8)
domain.add_domain_param('source', [0,0,10])

points = domain.create_domain(resolution = 200)

params_all = [params_lower, params_mean, params_upper]

visualiser = Visualiser(testing_data, params_all, model, hyperparams, acceptance_rate)
instance = visualiser.visualise_results(points, save = True)
visualiser.get_traceplot(params_samples, instance, save = True)
visualiser.animate(instance)


