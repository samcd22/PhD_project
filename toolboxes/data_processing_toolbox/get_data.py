import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from toolboxes.data_processing_toolbox.data_simulator import DataSimulator
from toolboxes.data_processing_toolbox.data_normaliser import DataNormaliser


def get_data(data_params):
    if data_params['data_type'] == 'simulated_data':
        data_simulator = DataSimulator(data_params, suppress_prints = True)
        data = data_simulator.generate_data()
    elif data_params['data_type'] == 'normalised_data':
        data_normaliser = DataNormaliser(data_params, suppress_prints = True)
        data = data_normaliser.normalise_data()
    return data