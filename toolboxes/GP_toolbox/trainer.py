import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

class Trainer:
    def __init__(self, training_data, kernel_type, num_epochs = 100):
        self.training_data = training_data
        self.kernel = kernel_type
        self.num_epochs = num_epochs
        
        # x, y, and z are the independent variables (scalars)
        x = self.training_data.x.values
        y = self.training_data.y.values
        z = self.training_data.z.values

        self.X = np.column_stack((x, y, z))

        # "Concentration" is the dependent variable
        self.concentration = self.training_data.Concentration.values

        # Create the kernel
        if self.kernel == 'matern_white':
            self.kernel = Matern(length_scale=1, nu=0.5) + WhiteKernel(noise_level=1.0)
        elif self.kernel == 'rbf':
            self.kernel = RBF(length_scale=1)
        elif self.kernel == 'matern':
            self.kernel = Matern(length_scale=1, nu=0.5)   

        self.model = None

    def train(self):
        # Create the Gaussian Process Regression model
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.num_epochs, normalize_y=True)

        # Fit the model with the data
        self.model.fit(self.X, self.concentration)
        
        self.params = self.model.kernel_.get_params()
        
        return self.model