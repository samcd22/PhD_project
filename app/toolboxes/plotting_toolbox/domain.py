import numpy as np
import pandas as pd

class Domain:
    """
    A class used for creating different domains, used for plotting results.

    Attributes:
    - domain_params (pd.Series): A pandas Series object to store the domain parameters.
    - domain_select (str): The selected domain type. Options are:
        - 'cone_from_source': A cone domain from the source.
        - 'cuboid_from_source': A cuboid domain from the source.
        - 'cone_from_source_z_limited': A cone domain from the source with z limited to >= 0.
    - n_dims (int): The number of dimensions of the domain.
    - resolution (int): The resolution of the domain.

    Methods:
    - __init__(self, domain_select: str): Initialises the Domain class.
    - add_domain_param(self, name: str, val: float) -> 'Domain': Saves a named parameter to the Domain class.
    - create_domain(self) -> np.ndarray: Generates the selected domain using the domain parameters.
    - create_domain_slice(self, slice_name: str) -> np.ndarray: Creates a slice of the selected domain.
    - get_construction(self) -> dict: Get the construction parameters of the domain.
    """

    def __init__(self, domain_select: str):
        """
        Initialises the Domain class saving all relevant variables.

        Args:
        - domain_select (str): The selected domain type.

        """
        self.domain_params = pd.Series({}, dtype='float64')
        self.domain_select = domain_select
        self.n_dims = None

        if domain_select in ['cone_from_source', 'cuboid_from_cource', 'cone_from_source_z_limited']:
            self.n_dims = 3

    def get_construction(self):
        """
        Get the construction parameters.

        Returns:
        - dict: The construction parameters.
        """
        construction = {
            'domain_select': self.domain_select,
            'domain_params': self.domain_params.to_dict(),
            'n_dims': self.n_dims
        }
        return construction

    def _check_required_params(self, required_params):
        """
        Checks if all required parameters are present in the domain_params.

        Args:
        - required_params (list): A list of required parameters.

        Raises:
        - Exception: If a required parameter is missing.
        """
        for param in required_params:
            if param not in self.domain_params:
                raise Exception(f'Domain - {param} is a required parameter for the {self.domain_select} domain! Please add this parameter.')

    def add_domain_param(self, name, val):
        """
        Saves a named parameter to the Domain class before generating the domain.

        Args:
        - name (str): The name of the parameter.
        - val: The value of the parameter.

        Returns:
        - self: The Domain object.
        """
        self.domain_params[name] = val
        return self

    def create_domain(self):
        """
        Generates the selected domain using the domain parameters.

        Returns:
        - points (numpy.ndarray): The generated domain points.
        """
        if self.domain_select == 'cone_from_source':
            return self._create_cone()

        elif self.domain_select == 'cuboid_from_source':
            return self._create_cuboid()

        elif self.domain_select == 'cone_from_source_z_limited':
            return self._create_z_limited_cone()
        else:
            raise Exception('Domain - Invalid domain selected!')

    def create_domain_slice(self, slice_name):
        """
        Creates a slice of the selected domain.

        Args:
        - slice_name (str): The name of the slice. Options are:
            - 'x_slice': A slice along the x-axis.
            - 'y_slice': A slice along the y-axis.
            - 'z_slice': A slice along the z-axis.

        Returns:
        - points (numpy.ndarray): The generated slice points.
        """
        if self.domain_select == 'cone_from_source_z_limited':
            return self._create_z_limited_cone_slice(slice_name)
        elif self.domain_select == 'cone_from_source':
            return self._create_cone_slice(slice_name)
        else:
            raise Exception('Domain - Invalid domain selected!')

    def _create_cone_slice(self, slice_name):
        """
        Creates a slice of the cone domain.

        Args:
        - slice_name (str): The name of the slice.

        Required parameters
        - source (list): The position of the tip of the cone.
        - r (float): The length of the cone away from the source.
        - theta (float): The angle the cone sweeps out.
        - resolution (int): The resolution of the domain.

        Returns:
            points (numpy.ndarray): The generated slice points.
        """

        required_params = ['source', 'r', 'theta', 'resolution']
        self._check_required_params(required_params)

        if 'x_slice' in slice_name:
            x = self.domain_params.x_slice
            r = x*np.tan(self.domain_params.theta)
            
            # Generate evenly spaced points on the surface of the circular area
            y_coordinates = np.linspace(self.domain_params.source[1] - r, self.domain_params.source[1] + r, self.domain_params.resolution)
            z_coordinates = np.linspace(self.domain_params.source[2] - r, self.domain_params.source[2] + r, self.domain_params.resolution)

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
            x_coordinates = np.linspace(np.tan(self.domain_params.theta)*(self.domain_params.y_slice-self.domain_params.source[1])+self.domain_params.source[0]+1,self.domain_params.r, self.domain_params.resolution)
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
            x_coordinates = np.linspace(np.tan(self.domain_params.theta)*(self.domain_params.z_slice-self.domain_params.source[2])+self.domain_params.source[0]+1,self.domain_params.r, self.domain_params.resolution)
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

    def _create_z_limited_cone_slice(self, slice_name):
        """
        Deletes all points of the cone which are below the x-y plane (z=0).

        Args:
        - slice_name (str): The name of the slice.

        Required parameters
        - source (list): The position of the tip of the cone.
        - r (float): The length of the cone away from the source.
        - theta (float): The angle the cone sweeps out.
        - resolution (int): The resolution of the domain.

        Returns:
        - points (numpy.ndarray): The generated slice points.
        """

        required_params = ['source', 'r', 'theta', 'resolution']
        self._check_required_params(required_params)

        points = self._create_cone_slice(slice_name)
        return points[points[:, 2] >= 0]

    def _create_cuboid(self):
        """
        Creates a cuboid of points.

        Required parameters
        - source (list): The source of the cuboid.
        - x (float): The x length of the cuboid.
        - y (float): The y length of the cuboid.
        - z (float): The z length of the cuboid.
        - resolution (int): The resolution of the domain.

        Returns:
        - points (numpy.ndarray): The generated cuboid points.
        """

        required_params = ['source', 'x', 'y', 'z', 'resolution']
        self._check_required_params(required_params)

        x_values = np.linspace(self.domain_params.source[0], self.domain_params.x, self.domain_params.resolution)
        y_values = np.linspace(self.domain_params.source[1], self.domain_params.y, self.domain_params.resolution)
        z_values = np.linspace(self.domain_params.source[2], self.domain_params.z, self.domain_params.resolution)

        X, Y, Z = np.meshgrid(x_values, y_values, z_values)

        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        return points


    def _create_cone(self):
        """
        Creates a cone of points from the source.

        Required parameters
        - source (list): The position of the tip of the cone.
        - r (float): The length of the cone away from the source.
        - theta (float): The angle the cone sweeps out.
        - resolution (int): The resolution of the domain.

        Returns:
        - points (numpy.ndarray): The generated cone points.
        """

        required_params = ['source', 'r', 'theta', 'resolution']
        self._check_required_params(required_params)

        r_values = np.linspace(1, self.domain_params.r, self.domain_params.resolution)
        x, y, z = [], [], []

        # Create a meshgrid of r and theta values
        for R in r_values:
            theta_values = np.linspace(-self.domain_params.theta, self.domain_params.theta, int(np.ceil(self.domain_params.resolution*R/(self.domain_params.r))))
            for Theta in theta_values:
                phi_values = np.linspace(0,np.pi,int(np.ceil(self.domain_params.resolution*abs(Theta)/(2*self.domain_params.theta))))
                for Phi in phi_values:
                # Compute the x, y, and z values for each point in the meshgrid
                    x.append(self.domain_params.source[0] + R*(np.cos(Theta)))
                    y.append(self.domain_params.source[1] + R*np.sin(Theta)*np.cos(Phi))
                    z.append(self.domain_params.source[2] + R*np.sin(Theta)*np.sin(Phi))

        # Flatten the arrays to create a list of (x, y, z) points
        points = np.column_stack([np.array(x), np.array(y), np.array(z)])

        return points

    def _create_z_limited_cone(self):
        """
        Deletes all points of the cone which are below the x-y plane (z=0).
        
        Required parameters
        - source (list): The position of the tip of the cone.
        - r (float): The length of the cone away from the source.
        - theta (float): The angle the cone sweeps out.
        - resolution (int): The resolution of the domain.
        
        Returns:
        - points (numpy.ndarray): The generated cone points.
        """
        points = self._create_cone()
        return points[points[:, 2] >= 0]