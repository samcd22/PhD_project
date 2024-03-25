import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

# Trainer class = trains the gaussian processor
class Trainer:
    # Initialises the Trainer class saving all relevant variables and performing some initialising tasks
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
        self.params = None

    # Trains the gaussian processor based on the selected kernel
    def train(self):
        # Create the Gaussian Process Regression model
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.num_epochs, normalize_y=True)

        # Fit the model with the data
        self.model.fit(self.X, self.concentration)
        
        # Saves the fitted gaussian processor parameters
        self.params = self.model.kernel_.get_params()
        
        return self.model