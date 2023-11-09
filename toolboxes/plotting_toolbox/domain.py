import numpy as np
import pandas as pd

# Domain class - used for creating different domains, used for plotting resulta
class Domain:
    # Initialises the Domain class saving all relevant variables
    def __init__(self, domain_select = None, resolution = None):
        self.domain_params = pd.Series({},dtype='float64')
        self.domain_select = domain_select
        self.n_dims = None
        if domain_select in ['cone_from_source', 'cuboid_from_cource', 'cone_from_source_z_limited']:
            self.n_dims = 3
        self.resolution  = resolution

    # Saves a named parameter to the Domain class before generating the domain
    def add_domain_param(self,name,val):
        self.domain_params[name] = val

    # Generates the selected domain using the domain parameters
    def create_3D_domain(self):
        if self.domain_select == 'cone_from_source':
            return self.create_cone()
        
        if self.domain_select == 'cuboid_from_source':
            return self.create_cuboid()
        
        if self.domain_select == 'cone_from_source_z_limited':
            return self.create_z_limited_cone()
        
    def create_2D_slice_domain(self, slice_name):
        if self.domain_select == 'cone_from_source_z_limited':
            return self.create_z_limited_cone_slice(slice_name)
        
    def create_cone_slice(self, slice_name):
        if 'x_slice' in slice_name:
            x = self.domain_params.x_slice
            r = x*np.tan(self.domain_params.theta)
            
            # Generate evenly spaced points on the surface of the circular area
            y_coordinates = np.linspace(self.domain_params.source[1] - r, self.domain_params.source[1] + r, self.resolution)
            z_coordinates = np.linspace(self.domain_params.source[2] - r, self.domain_params.source[2] + r, self.resolution)

            # Create a grid of points
            points = [(y, z) for y in y_coordinates for z in z_coordinates if (y - self.domain_params.source[1]) ** 2 + (z - self.domain_params.source[2]) ** 2 <= r ** 2]

            # Extract x and y coordinates for plotting
            Y, Z = zip(*points)
            Y = np.array(Y)
            Z = np.array(Z)

            X = x*np.ones(Y.shape)

            points = np.column_stack([X, Y, Z])
            return points

        elif 'y_slice' in slice_name:
            x_coordinates = np.linspace(np.tan(self.domain_params.theta)*(self.domain_params.y_slice-self.domain_params.source[1])+self.domain_params.source[0]+1,self.domain_params.r, self.resolution)
            z_coordinates = np.linspace(self.domain_params.source[2]-np.tan(self.domain_params.theta)*self.domain_params.r, self.domain_params.source[2]+np.tan(self.domain_params.theta)*self.domain_params.r)

            points = [(x, z) for x in x_coordinates for z in z_coordinates if (z-self.domain_params.source[2])**2<=(x-self.domain_params.source[0])**2*np.tan(self.domain_params.theta)**2-(self.domain_params.y_slice-self.domain_params.source[1])**2]
            # Create a meshgrid of r and theta values

            X, Z = zip(*points)
            X = np.array(X)
            Z = np.array(Z)

            Y = self.domain_params.y_slice*np.ones(X.shape)

            points = np.column_stack([X, Y, Z])
            return points
        
        elif 'z_slice' in slice_name:
            x_coordinates = np.linspace(np.tan(self.domain_params.theta)*(self.domain_params.z_slice-self.domain_params.source[2])+self.domain_params.source[0]+1,self.domain_params.r, self.resolution)
            y_coordinates = np.linspace(self.domain_params.source[1]-np.tan(self.domain_params.theta)*self.domain_params.r, self.domain_params.source[1]+np.tan(self.domain_params.theta)*self.domain_params.r)

            points = [(x, y) for x in x_coordinates for y in y_coordinates if (y-self.domain_params.source[1])**2<=(x-self.domain_params.source[0])**2*np.tan(self.domain_params.theta)**2-(self.domain_params.z_slice-self.domain_params.source[2])**2]
            # Create a meshgrid of r and theta values

            X, Y = zip(*points)
            X = np.array(X)
            Y = np.array(Y)

            Z = self.domain_params.z_slice*np.ones(X.shape)

            points = np.column_stack([X, Y, Z])
            return points
        else:
            raise Exception('No slice inputted!')

    # Deletes all points of the cone which are below the x-y plane (z=0)
    def create_z_limited_cone_slice(self, slice_name):
        points = self.create_cone_slice(slice_name)
        return points[points[:,2]>=0]

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