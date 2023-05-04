import numpy as np
import pandas as pd

class Domain:
    def __init__(self, domain_select = 'cuboid_from_source', resolution = 100):
        self.domain_params = pd.Series({},dtype='float64')
        self.domain_select = domain_select
        self.resolution  = resolution

    def add_domain_param(self,name,val):
        self.domain_params[name] = val

    def create_domain(self):
        if self.domain_select == 'cone_from_source':
            return self.create_cone()
        
        if self.domain_select == 'cuboid_from_source':
            x_values = np.linspace(self.domain_params.source[0], self.domain_params.x, self.resolution)
            y_values = np.linspace(self.domain_params.source[1], self.domain_params.y, self.resolution)
            z_values = np.linspace(self.domain_params.source[2], self.domain_params.z, self.resolution)

            X, Y, Z = np.meshgrid(x_values, y_values, z_values)

            points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
            return points
        
        if self.domain_select == 'cone_from_source_z_limited':
            points = self.create_cone()
            return points[points[:,2]>=0]


    def create_cone(self):
        r_values = np.linspace(1, self.domain_params.r, self.resolution)
        x, y, z = [], [], []

        # Create a meshgrid of r and theta values
        for R in r_values:
            theta_values = np.linspace(-self.domain_params.theta, self.domain_params.theta, int(np.ceil(self.resolution*R/(self.domain_params.r))))
            for Theta in theta_values:
                phi_values = np.linspace(0,np.pi,int(np.ceil(self.resolution*abs(Theta)/(2*self.domain_params.theta))))
                for Phi in phi_values:
                # Compute the x, y, and z values for each point in the meshgrid
                    x.append(self.domain_params.source[0] + R*(np.cos(Theta)))
                    y.append(self.domain_params.source[1] + R*np.sin(Theta)*np.cos(Phi))
                    z.append(self.domain_params.source[2] + R*np.sin(Theta)*np.sin(Phi))

        # Flatten the arrays to create a list of (x, y, z) points
        points = np.column_stack([np.array(x), np.array(y), np.array(z)])

        return points
