import numpy as np
import pandas as pd

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