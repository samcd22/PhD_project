import numpy as np
import pandas as pd

# Domain class - used for creating different domains, used for plotting resulta
class Domain:
    # Initialises the Domain class saving all relevant variables
    def __init__(self, domain_select = 'cuboid_from_source', resolution = None):
        self.domain_params = pd.Series({},dtype='float64')
        self.domain_select = domain_select
        self.resolution  = resolution

    # Saves a named parameter to the Domain class before generating the domain
    def add_domain_param(self,name,val):
        self.domain_params[name] = val

    # Generates the selected domain using the domain parameters
    def create_domain(self):
        if self.domain_select == 'cone_from_source':
            return self.create_cone()
        
        if self.domain_select == 'cuboid_from_source':
            return self.create_cuboid()
        
        if self.domain_select == 'cone_from_source_z_limited':
            return self.create_z_limited_cone()

        if self.domain_select == 'discrete_data_points':
            return self.create_discrete_data_points()


    # Creates a cuboid of points
    def create_cuboid(self):
        x_values = np.linspace(self.domain_params.source[0], self.domain_params.x, self.resolution)
        y_values = np.linspace(self.domain_params.source[1], self.domain_params.y, self.resolution)
        z_values = np.linspace(self.domain_params.source[2], self.domain_params.z, self.resolution)

        X, Y, Z = np.meshgrid(x_values, y_values, z_values)

        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        return points

    # Creates a cone of points from the source, for a radius of r away from it
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

    # Deletes all points of the cone which are below the x-y plane (z=0)
    def create_z_limited_cone(self):
        points = self.create_cone()
        return points[points[:,2]>=0]

    #Create a grid of point
    def create_discrete_data_points(self):
        x = self.domain_params.x
        y = self.domain_params.y
        z = self.domain_params.z
        points = np.column_stack([np.array(x), np.array(y), np.array(z)])
        if not (len(x) == len(y) and len(y) == len(z)):
            raise Exception('Coordinates with different length')
        return points
