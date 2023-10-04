import jax.numpy as jnp
import pandas as pd

# Model class - used for generating the model function
class Model:
    # Initialises the Model class saving all relevant variables
    def __init__(self, model_select):
        self.model_params = pd.Series({},dtype='float64')
        self.model_select = model_select

    # Saves a named parameter to the Model class before generating the model function
    def add_model_param(self,name,val):
        self.model_params[name] = val
        return self
    
    # Generates the selected model function using the model parameters
    def get_model(self):
        # Wind speed normalised Gaussian Plume Model, alternative form
        def GPM_norm(params, x, y, z):
            I_y = params['I_y']
            I_z = params['I_z']
            Q = params['Q']

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            H = self.model_params.H
            
            return Q/(jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2)))
        

        
        # Wind speed normalised Gaussian Plume Model, alternative form, with logged Q
        def GPM_norm_log_Q(params, x, y, z):
            I_y = params['I_y']
            I_z = params['I_z']
            log_10_Q = params['log_10_Q']

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            H = self.model_params.H
            return 10**log_10_Q/(jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2)))
        

        
        #Wind speed normalised, logged Gaussian Plume Model, alternative form
        def log_GPM_norm(params, x, y, z):
            I_y = params['I_y']
            I_z = params['I_z']
            Q = params['Q']

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            H = self.model_params.H
            return jnp.log10(Q/(jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2))))
        

        
    
        #Wind speed normalised, logged Gaussian Plume Model, alternative form
        def sarah_model(params, x, y, z):
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
        

        #Logged Gaussian Plume Model, using pasquil gifford for sigma_y and sigma_z
        def sarah_model_PG(params, x, y, z):
            Q = params['Q']
            x_0 = params['x_0']
            y_0 = params['y_0']
            z_0 = params['z_0']

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            u = self.model_params.u

            output = jnp.log10(Q/(2*u*jnp.pi*0.195*(x-x_0)**0.9*0.112*(x-x_0)**0.91)*jnp.exp(-(y-y_0)**2/(2*(0.195*(x-x_0)**0.9)**2))*(jnp.exp(-(z-z_0)**2/(2*(0.112*(x-x_0)**0.91)**2))+jnp.exp(-(z+z_0)**2/(2*(0.112*(x-x_0)**0.91)**2))))
            return output
        
        #Logged Gaussian Plume Model, using Martin scheme for sigma_y and sigma_
        def sarah_model_Martin(params, x, y, z):
            Q = params['Q']
            x_0 = params['x_0']
            y_0 = params['y_0']
            z_0 = params['z_0']

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            u = self.model_params.u

            output = jnp.log10(Q/(2*u*jnp.pi*104*(x-x_0)**0.894*61*(x-x_0)**0.911)*jnp.exp(-(y-y_0)**2/(2*(104*(x-x_0)**0.894)**2))*(jnp.exp(-(z-z_0)**2/(2*(61*(x-x_0)**0.911)**2))+jnp.exp(-(z+z_0)**2/(2*(61*(x-x_0)**0.911)**2))))
            return output

        if self.model_select == "gpm_norm":
            return GPM_norm
        
        elif self.model_select == "gpm_norm_log_Q":
            return GPM_norm_log_Q
        
        elif self.model_select == "log_gpm_norm":
            return log_GPM_norm

        elif self.model_select == "sarah_model":
            return sarah_model
        
        elif self.model_select == "sarah_model_PG":
            return sarah_model_PG
        
        elif self.model_select == "sarah_model_Martin":
            return sarah_model_Martin
        
        else:
            raise Exception('Model does not exist!')