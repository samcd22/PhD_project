import jax.numpy as jnp
import pandas as pd

from typing import Union
ModelInput = Union[str, int, float]

class Model:
    """
    A class representing the Model used for predicting a system's behaviour. The parameters of this model are optimised using Bayesian Inference

    Attributes:
    - fixed_model_params (pd.Series): A pandas Series object to store model parameters.
    - model_select (str): The selected model.
    - model_type (str): The type of model.

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
            - Spatial 3D models:
                - "gpm_norm": Wind speed weighted Gaussian Plume Model.
                - "log_gpm_norm": Wind speed weighted, logarithmised Gaussian Plume Model.
                - "log_gpm_source": Logarithmised Gaussian Plume Model, with source location.
        """
        self.fixed_model_params = pd.Series({}, dtype='float64')
        self.model_select = model_select
        
        if self.model_select == "gpm_norm":
            self.model_type = 'scalar_spatial_3D'
            self.dependent_variables = ['Concentration']
            self.independent_variables = ['x', 'y', 'z']
            self.all_param_names = ['I_y', 'I_z', 'Q', 'H']

        elif self.model_select == "log_gpm_norm":
            self.model_type = 'scalar_spatial_3D'
            self.dependent_variables = ['Concentration']
            self.independent_variables = ['x', 'y', 'z']
            self.all_param_names = ['I_y', 'I_z', 'Q', 'H']
        
        elif self.model_select == "log_gpm_source":
            self.model_type = 'scalar_spatial_3D'
            self.dependent_variables = ['Concentration']
            self.independent_variables = ['x', 'y', 'z']
            self.all_param_names = ['I_y', 'I_z', 'Q', 'H', 'x_0', 'y_0', 'z_0', 'u']

        # IMPLEMENT MORE MODELS HERE
        else:
            raise Exception('Invalid model selected!')    
        

    def add_fixed_model_param(self, name: str, val) -> 'Model':
        """
        Adds a named parameter to the Model class.

        Args:
        - name (str): The name of the parameter.
        - val: The value of the parameter.

        Returns:
        - self: The Model object.
        """
        self.fixed_model_params[name] = val
        return self
        
    def _set_params(self, inference_model_params):
        """
        Set the parameters of the model.

        Args:
            inference_model_params (dict): A dictionary containing the inference model parameters.

        Raises:
            Exception: If any required parameter is missing for the model.

        """
        self.inference_params = inference_model_params
        for param_name in self.all_param_names:
            if param_name in inference_model_params:
                setattr(self, param_name, inference_model_params[param_name])
            elif param_name in self.fixed_model_params:
                setattr(self, param_name, self.fixed_model_params[param_name]) 
            else:
                raise Exception('Model - missing parameters for the model!')
            

    def get_model(self):
        
        """
        Generates the selected model function using the model parameters.

        Returns:
        - function: The selected model function.
        """

        def _GPM_norm(inferance_model_params: pd.Series, x, y, z):
            """
            Wind speed weighted Gaussian Plume Model.

            Args:
            - inferance_model_params (pd.Series): A pandas Series object containing the inferred model parameters.
            - x (float or array-like): The x-coordinate.
            - y (float or array-like): The y-coordinate.
            - z (float or array-like): The z-coordinate.

            Parameters:
            - I_y (float): The y-component of the plume dispersion.
            - I_z (float): The z-component of the plume dispersion.
            - Q (float): The input flux rate.
            - H (float): The height of the source.

            Independent Variables:
            - x (float or array-like): The x-coordinate.
            - y (float or array-like): The y-coordinate.
            - z (float or array-like): The z-coordinate.

            Dependent Variables:
            - Concentration (float or array-like): The concentration of the plume.
            
            Returns:
            - jnp.ndarray: The model output.
            """
            self._set_params(inferance_model_params)
                
            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            func = self.Q/(2*jnp.pi*self.I_y*self.I_z*x**2)*jnp.exp(-y**2/(2*self.I_y**2*x**2))*(jnp.exp(-(z-self.H)**2/(2*self.I_z**2*x**2))+jnp.exp(-(z+self.H)**2/(2*self.I_z**2*x**2)))
            
            return func
                        
        def _log_GPM_norm(inference_model_params, x, y, z):
            """
            Wind speed weighted, logarithmised Gaussian Plume Model.

            Args:
            - inference_model_params (dict): A dictionary containing the model parameters.
            - x (float or array-like): The x-coordinate.
            - y (float or array-like): The y-coordinate.
            - z (float or array-like): The z-coordinate.

            Parameters:
            - I_y (float): The y-component of the plume dispersion.
            - I_z (float): The z-component of the plume dispersion.
            - Q (float): The input flux rate.
            - H (float): The height of the source.

            Independent Variables:
            - x (float or array-like): The x-coordinate.
            - y (float or array-like): The y-coordinate.
            - z (float or array-like): The z-coordinate.

            Dependent Variables:
            - Concentration (float or array-like): The concentration of the plume.

            Returns:
            - jnp.ndarray: The model output.
            """

            self._set_params(self.all_param_names, inference_model_params, self.fixed_model_params)

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            return jnp.log10(self.Q/(2*jnp.pi*self.I_y*self.I_z*x**2)*jnp.exp(-y**2/(2*self.I_y**2*x**2))*(jnp.exp(-(z-self.H)**2/(2*self.I_z**2*x**2))+jnp.exp(-(z+self.H)**2/(2*self.I_z**2*x**2))))
                
        def _log_gpm_source(params, x, y, z):
            """
            Wind speed weighted, logarithmised Gaussian Plume Model.

            Args:
            - params (dict): A dictionary containing the model parameters.
            - x (float or array-like): The x-coordinate.
            - y (float or array-like): The y-coordinate.
            - z (float or array-like): The z-coordinate.

            Parameters:
            - I_y (float): The y-component of the plume dispersion.
            - I_z (float): The z-component of the plume dispersion.
            - Q (float): The input flux rate.
            - x_0 (float): The x-coordinate of the source.
            - y_0 (float): The y-coordinate of the source.
            - z_0 (float): The z-coordinate of the source.
            - u (float): The wind speed.
            - H (float): The height of the source.

            Independent Variables:
            - x (float or array-like): The x-coordinate.
            - y (float or array-like): The y-coordinate.
            - z (float or array-like): The z-coordinate.

            Dependent Variables:
            - Concentration (float or array-like): The concentration of the plume.

            Returns:
            - jnp.ndarray: The model output.

            """
            self._set_params(self.all_param_names, params, self.fixed_model_params)

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)
            func = jnp.log10(self.Q/(2*self.u*jnp.pi*self.I_y*self.I_z*(x-self.x_0)**2)*jnp.exp(-(y-self.y_0)**2/(2*self.I_y**2*(x-self.x_0)**2))*(jnp.exp(-(z-self.z_0)**2/(2*self.I_z**2*(x-self.x_0)**2))+jnp.exp(-(z+self.z_0)**2/(2*self.I_z**2*(x-self.x_0)**2))))
            return func
        
        # IMPLEMENT MORE MODELS HERE

        if self.model_select == "gpm_norm":
            return _GPM_norm
        elif self.model_select == "log_gpm_norm":
            return _log_GPM_norm
        elif self.model_select == "log_gpm_source":
            return _log_gpm_source
        else:
            raise Exception('Model - invalid model selected!')