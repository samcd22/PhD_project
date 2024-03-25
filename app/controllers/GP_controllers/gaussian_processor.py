import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import train_test_split
import os
import imageio
import json
from numpyencoder import NumpyEncoder

from toolboxes.data_processing_toolbox.get_data import get_data
from controllers.controller import Controller
from toolboxes.GP_toolbox.trainer import Trainer
from toolboxes.GP_toolbox.visualiser import Visualiser

# GaussianProcessor class - runs an instance of the gaussian processor, saving plots of the results
class GaussianProcessor(Controller):
    # Initialises the GaussianProcessor class saving all relevant variables and performing some initialising tasks
    def __init__(self,results_name = 'name_placeholder',
                 data_params = None,
                 results_path = 'results/GP_results'):
        
        # Inherits methods and attributes from parent Controller class
        super().__init__(results_name, data_params, None, results_path)

        # Generates results folder
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Generates the construction object
        construction = self.get_constriction()

        # Initialises the construction
        self.init_construction(construction)

        # Generates the data based on the data_params object and splits it into train and test data
        data = get_data(data_params)
        self.training_data, self.testing_data = train_test_split(data, test_size=0.2, random_state = 1)
        
    # Initialises the construction using the construction object, checking and creating all relevant files and folders
    def init_construction(self, construction):
        self.construction_results_path = self.results_path + '/' + self.results_name
        self.full_results_path = self.construction_results_path

        if not os.path.exists(self.construction_results_path):
            os.makedirs(self.construction_results_path)

        if not os.path.exists(self.full_results_path):
            os.makedirs(self.full_results_path)

        if os.path.exists(self.construction_results_path + '/construction.json'):
            f = open(self.construction_results_path + '/construction.json')
            saved_construction = json.load(f)
            f.close()

            if saved_construction != construction:
                raise Exception('Default generator parameters do not match for this folder name!')
        else:
            with open(self.full_results_path + '/construction.json', "w") as fp:
                json.dump(construction,fp, cls=NumpyEncoder, separators=(', ',': '), indent=4)

    # Runs an instance of the Gaussian Processor
    def run(self, kernel_type, domain, name, num_epochs = 20):
        trainer = Trainer(self.training_data, kernel_type, num_epochs)
        trainer.train()
        visualiser = Visualiser(self.testing_data, trainer, kernel_type, data_path = self.results_path + '/' + self.results_name)
        visualiser.visualise_results(domain, name)
        visualiser.animate(name)