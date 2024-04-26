import jax.numpy as jnp
import pandas as pd

from typing import Union
ModelInput = Union[str, int, float]

class Model:
    """
    A class representing the Model used for predicting a system's behaviour. The parameters of this model are optimised using Bayesian Inference

    Attributes:
    - model_params (pd.Series): A pandas Series object to store model parameters.
    - model_select (str): The selected model.

    Methods:
    - __init__(self, model_select: str): Initializes the Model class.
    - add_model_param(self, name: str, val: ModelInput) -> 'Model': Adds a named parameter to the Model class.
    - get_model(self) ->: callable Generates the selected model function using the model parameters.
    """

    def __init__(self, model_select: str):
        """
        Initializes the Model class.

        Args:
        - model_select (str): The selected model. Options are:
            - "gpm_norm": Wind speed weighted Gaussian Plume Model.
            - "gpm_norm_log_Q": Wind speed weighted Gaussian Plume Model, with logarithmised Q.
            - "log_gpm_norm": Wind speed weighted, logarithmised Gaussian Plume Model.
            - "log_gpm_norm_log_Q": Wind speed weighted, logarithmised Gaussian Plume Model, with logarithmised Q.
            - "log_gpm_norm_source": Wind speed weighted, logarithmised Gaussian Plume Model, with source.
        """
        self.model_params = pd.Series({}, dtype='float64')
        self.model_select = model_select
    
    def add_model_param(self, name: str, val: ModelInput) -> 'Model':
        """
        Adds a named parameter to the Model class.

        Args:
        - name (str): The name of the parameter.
        - val (ModelInput): The value of the parameter.

        Returns:
        - self: The Model object.
        """
        self.model_params[name] = val
        return self
    
    def get_model(self):
        
        """
        Generates the selected model function using the model parameters.

        Returns:
        - function: The selected model function.
        """
        def _GPM_norm(params, x, y, z):
            """
            Wind speed weighted Gaussian Plume Model.

            Args:
            - params (dict): A dictionary containing the model parameters.
            - x, y, z: The coordinates.

            Returns:
            - jnp.ndarray: The model output.
            """
            I_y = params['I_y']
            I_z = params['I_z']
            Q = params['Q']
            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            H = self.model_params.H
            
            return Q/(2*jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2)))
        
        def _GPM_norm_log_Q(params, x, y, z):
            """
            Wind speed weighted Gaussian Plume Model, with logarithmised Q.

            Args:
            - params (dict): A dictionary containing the model parameters.
            - x, y, z: The coordinates.

            Returns:
            - jnp.ndarray: The model output.
            """
            I_y = params['I_y']
            I_z = params['I_z']
            log_10_Q = params['log_10_Q']
            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)
            H = self.model_params.H
            return 10**log_10_Q/(2*jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2)))
        
        def _log_GPM_norm(params, x, y, z):
            """
            Wind speed weighted, logarithmised Gaussian Plume Model.

            Args:
            - params (dict): A dictionary containing the model parameters.
            - x, y, z: The coordinates.

            Returns:
            - jnp.ndarray: The model output.
            """
            I_y = params['I_y']
            I_z = params['I_z']
            Q = params['Q']
            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)
            H = self.model_params.H
            return jnp.log10(Q/(2*jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2))))
        
        def _log_GPM_norm_log_Q(params, x, y, z):
            """
            Wind speed weighted, logarithmised Gaussian Plume Model, with logarithmised Q.

            Args:
            - params (dict): A dictionary containing the model parameters.
            - x, y, z: The coordinates.

            Returns:
            - jnp.ndarray: The model output.
            """
            I_y = params['I_y']
            I_z = params['I_z']
            Q = params['Q']
            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)
            H = self.model_params.H
            return jnp.log10(jnp.log(Q)/(2*jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2))))
        
        def _log_gpm_norm_source(params, x, y, z):
            """
            Wind speed weighted, logarithmised Gaussian Plume Model.

            Args:
            - params (dict): A dictionary containing the model parameters.
            - x, y, z: The coordinates.

            Returns:
            - jnp.ndarray: The model output.
            """
            I_y = params['I_y']
            I_z = params['I_z']
            Q = params['Q']
            x_0 = params['x_0']
            y_0 = params['y_0']
            z_0 = params['z_0']
            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)
            u = self.model_params.u
            output = jnp.log10(Q/(2*u*jnp.pi*I_y*I_z*(x-x_0)**2)*jnp.exp(-(y-y_0)**2/(2*I_y**2*(x-x_0)**2))*(jnp.exp(-(z-z_0)**2/(2*I_z**2*(x-x_0)**2))+jnp.exp(-(z+z_0)**2/(2*I_z**2*(x-x_0)**2))))
            return output
        
        if self.model_select == "gpm_norm":
            return _GPM_norm
        elif self.model_select == "gpm_norm_log_Q":
            return _GPM_norm_log_Q
        elif self.model_select == "log_gpm_norm":
            return _log_GPM_norm
        elif self.model_select == "log_gpm_norm_log_Q":
            return _log_GPM_norm_log_Q
        elif self.model_select == "log_gpm_norm_source":
            return _log_gpm_norm_source
        else:
            raise Exception('Invalid model selected!')