import jax.numpy as jnp

import pandas as pd

class Model:
    def __init__(self, model_select):
        self.model_params = pd.Series({},dtype='float64')
        self.model_select = model_select

    def add_model_param(self,name,val):
        self.model_params[name] = val
    
    # Model Function
    def get_model(self, sample = False):

        def GPM(params, x, y, z):
            
            a = params.a
            b = params.b
            Q = params.Q
            H = self.model_params.H
            u = self.model_params.u
            tmp = 2*a*x**b
            
            return Q / (tmp*jnp.pi*u)*(jnp.exp(-(y**2)/tmp))*(jnp.exp(-(z-H)**2/tmp)+jnp.exp(-(z+H)**2/tmp))

        if self.model_select == "GPM":
            return GPM
        
        def GPM_norm(params, x, y, z):
            a = params.a
            b = params.b
            Q = params.Q

            H = self.model_params.H
            tmp = 2*a*x**b
            
            return Q / (tmp*jnp.pi)*(jnp.exp(-(y**2)/tmp))*(jnp.exp(-(z-H)**2/tmp)+jnp.exp(-(z+H)**2/tmp))

        if self.model_select == "GPM_norm":
            return GPM_norm
        
        def GPM_alt_norm(params, x, y, z):
            I_y = params.I_y
            I_z = params.I_z
            Q = params.Q

            x = jnp.array(x)
            y = jnp.array(y)
            z = jnp.array(z)

            H = self.model_params.H
            
            return Q/(jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2)))
        
        if self.model_select == "GPM_alt_norm":
            return GPM_alt_norm
        
        def GPM_alt_norm_log_Q(params, x, y, z):
            I_y = params.I_y
            I_z = params.I_z
            log_10_Q = params.log_10_Q
            H = self.model_params.H
            return 10**log_10_Q/(jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2)))
        
        if self.model_select == "GPM_alt_norm_log_Q":
            return GPM_alt_norm_log_Q
        
        def log_GPM_alt_norm(params, x, y, z):
            I_y = params.I_y
            I_z = params.I_z
            Q = params.Q
            H = self.model_params.H
            return jnp.log10(Q/(jnp.pi*I_y*I_z*x**2)*jnp.exp(-y**2/(2*I_y**2*x**2))*(jnp.exp(-(z-H)**2/(2*I_z**2*x**2))+jnp.exp(-(z+H)**2/(2*I_z**2*x**2))))
        
        if self.model_select == "log_GPM_alt_norm":
            return log_GPM_alt_norm