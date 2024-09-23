import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Domain:
    """
    A class for creating different domains, used for plotting results and simulating data. parameters are added to the domain_params attribute before creating the domain.

    Args:
        - domain_select (str): The selected domain type. Options are:
            - 'cone_from_source': A cone domain from the source.
                Required parameters:
                    - source (list): The position of the tip of the cone.
                    - r (float): The length of the cone away from the source.
                    - theta (float): The angle the cone sweeps out.
                    - resolution (int): The resolution of the domain. The higher the resolution, the more points are generated.
            - 'cuboid_from_source': A cuboid domain from the source.
                Required parameters:
                    - source (list): The source of the cuboid. The source is the bottom left corner of the cuboid.
                    - x (float): The x length of the cuboid.
                    - y (float): The y length of the cuboid.
                    - z (float): The z length of the cuboid.
                    - resolution (int): The resolution of the domain.
            - 'cone_from_source_z_limited': A cone domain from the source with z limited to >= 0.
                Required parameters:
                    - source (list): The position of the tip of the cone.
                    - r (float): The length of the cone away from the source.
                    - theta (float): The angle the cone sweeps out.
                    - resolution (int): The resolution of the domain. The higher the resolution, the more points are generated.
            - 'one_D': A one dimensional domain.
                Required parameters:
                    - x_min (float): The minimum x value.
                    - x_max (float): The maximum x value.
                    - resolution (int): The resolution of the domain. The higher the resolution, the more points are generated
            - 'two_D': A two dimensional domain.
                Required parameters:
                    - x_min (float): The minimum x value.
                    - x_max (float): The maximum x value.
                    - y_min (float): The minimum y value.
                    - y_max (float): The maximum y value.
                    - resolution (int): The resolution of the domain. The higher the resolution, the more points are generated
            - 'sphere': A sphere domain.
                Required parameters:
                    - source (list): The centre of the sphere.
                    - r (float): The radius of the sphere.
                    - n_points (int): The number of points to generate on the sphere.
            - 'cylinder': A cylinder domain.
                Required parameters:
                    - source (list): The centre of the cylinder.
                    - r (float): The radius of the cylinder.
                    - h (float): The height of the cylinder.
                    - resolution (int): The resolution of the domain.
                Optional parameters:
                    - orientation (list): The orientation of the cylinder. The default orientation is [0, 0, 1].
            - Additional domain types can be added as required.

    Attributes:
        - domain_params (pd.Series): A pandas Series object to store the domain parameters.
        - domain_select (str): The selected domain type.
        - n_dims (int): The number of dimensions of the domain.
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

        if domain_select in ['cone_from_source', 'cuboid_from_cource', 'cone_from_source_z_limited', 'sphere', 'cylinder']:
            self.n_dims = 3
        if domain_select in ['one_D']:
            self.n_dims = 1

    def get_construction(self) -> dict:
        """
        Get the construction of the domain. The conctruction parameters includes all of the config information used to construct the domain object. It includes:
            - domain_select: The selected domain type.
            - domain_params: The domain parameters.
            - n_dims: The number of dimensions of the domain.
        """

        domain_params = {}

        for key, value in self.domain_params.items():
            if isinstance(value, np.ndarray):
                domain_params[key] = value.tolist()
            else:
                domain_params[key] = value

        construction = {
            'domain_select': self.domain_select,
            'domain_params': domain_params,
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

    def add_domain_param(self, name, val) -> 'Domain':
        """
        Saves a named parameter to the Domain class before generating the domain. Note that all domain parameters bust be added to the class before the domain is created.

        Args:
            - name (str): The name of the parameter.
            - val: The value of the parameter.

        """
        self.domain_params[name] = val
        return self

    def create_domain(self) -> np.ndarray:
        """
        Generates the selected domain using the domain parameters.

        Returns:
        - numpy.ndarray: The generated domain points.
        """
        if self.domain_select == 'cone_from_source':
            return self._create_cone()

        elif self.domain_select == 'cuboid_from_source':
            return self._create_cuboid()

        elif self.domain_select == 'cone_from_source_z_limited':
            return self._create_z_limited_cone()
        elif self.domain_select == 'one_D':
            return self._create_one_D()
        elif self.domain_select == 'sphere':
            return self._create_sphere()
        elif self.domain_select == 'cylinder':
            return self._create_cylinder()
        else:
            raise Exception('Domain - Invalid domain selected!')

    def create_domain_slice(self, slice_name) -> np.ndarray:
        """
        Creates a slice of the selected domain along the specified axis.
        (This currently only works for the cone domains)

        Args:
            - slice_name (str): The name of the slice. Options are:
                - 'x_slice': A slice along the x-axis.
                - 'y_slice': A slice along the y-axis.
                - 'z_slice': A slice along the z-axis.

        Returns:
            - numpy.ndarray: The generated slice points.
        """
        if self.n_dims == 3:
            if self.n_dims == 3:
                if slice_name == 'x_slice':
                    return self._create_slice('x', self.domain_params.x_slice)
                elif slice_name == 'y_slice':
                    return self._create_slice('y', self.domain_params.y_slice)
                elif slice_name == 'z_slice':
                    return self._create_slice('z', self.domain_params.z_slice)
                else:
                    raise Exception('Invalid slice name!')
            else:
                raise Exception('Domain - Invalid domain selected!')

    def _create_slice(self, axis, value) -> np.ndarray:
        """
        Creates a slice of the selected domain along the specified axis.

        Args:
            - axis (str): The axis along which to create the slice.
            - value (float): The value of the coordinate along the specified axis.

        Returns:
            - numpy.ndarray: The generated slice points.
        """
        points = self.create_domain()
        if axis == 'x':
            return points[np.isclose(points[:, 0], value)]
        elif axis == 'y':
            return points[np.isclose(points[:, 1], value)]
        elif axis == 'z':
            return points[np.isclose(points[:, 2], value)]
        else:
            raise Exception('Domain - Invalid axis!')
        

    def _create_one_D(self) -> np.ndarray:
        if 'x_min' in self.domain_params and 'x_max' in self.domain_params:
            x_min = self.domain_params.x_min
            x_max = self.domain_params.x_max
            resolution = self.domain_params.resolution
            return np.linspace(x_min, x_max, resolution)
        elif 'points' in self.domain_params:
            self.domain_params.points = np.sort(np.array(self.domain_params.points))
            return self.domain_params.points

    def _create_cone_slice(self, slice_name) -> np.ndarray:
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
            - points (numpy.ndarray): The generated slice points.
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

    def _create_z_limited_cone_slice(self, slice_name) -> np.ndarray:
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

    def _create_cuboid(self) -> np.ndarray:
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

    def _create_sphere(self) -> np.ndarray:
        """
        Creates a sphere of points.

        Required parameters
            - source (list): The centre of the sphere.
            - r (float): The radius of the sphere.
            - resolution (int): The resolution of the domain.

        Returns:
            - points (numpy.ndarray): The generated sphere points.
        """

        required_params = ['source', 'r', 'n_points']
        self._check_required_params(required_params)

        radius = self.domain_params.r
        n_points = self.domain_params.n_points
        points = []
        for i in range(n_points):
            # Step 1: Map point onto a unit sphere
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)  # polar angle
            theta = np.pi * (1 + 5 ** 0.5) * i  # azimuthal angle (Fibonacci-based)

            # Step 2: Generate random radius for the point (scaled for volume uniformity)
            r = radius * np.cbrt(np.random.uniform(0, 1))  # Cube root for volume distribution

            # Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
            x = self.domain_params.source[0] + r * np.sin(phi) * np.cos(theta)
            y = self.domain_params.source[1] + r * np.sin(phi) * np.sin(theta)
            z = self.domain_params.source[2] + r * np.cos(phi)

            points.append([x, y, z])
        
        return np.array(points)
    
    def _create_cylinder(self) -> np.ndarray:
        """
        Creates a cylinder of points.

        Required parameters
            - source (list): The centre of the cylinder.
            - r (float): The radius of the cylinder.
            - h (float): The height of the cylinder.
            - resolution (int): The resolution of the domain.
        Optional parameters
            - orientation (list): The orientation of the cylinder.

        Returns:
            - points (numpy.ndarray): The generated cylinder points.
        """

        orientation = self.domain_params.orientation if 'orientation' in self.domain_params else [0, 0, 1]

        required_params = ['source', 'r', 'h', 'resolution']
        self._check_required_params(required_params)



        r_values = np.linspace(0, self.domain_params.r, self.domain_params.resolution)
        theta_values = np.linspace(0, 2*np.pi, self.domain_params.resolution)
        h_values = np.linspace(-self.domain_params.h/2,self.domain_params.h/2, self.domain_params.resolution)
        x, y, z = [], [], []

        default_orientation = [0, 0, 1]
        default_orientation = default_orientation / np.linalg.norm(default_orientation)
        new_orientation = orientation / np.linalg.norm(orientation)
        axis = np.cross(default_orientation, new_orientation)
        Rot = np.eye(3)

        if not np.allclose(axis, [0, 0, 0]):
            angle = np.arccos(np.clip(np.dot(default_orientation, new_orientation), -1, 1))

            K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]])
            
            Rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            print(Rot)

        # Create a meshgrid of r and theta values
        for h in h_values:
            for R in r_values:
                for Theta in theta_values:
                    # Compute the x, y, and z values for each point in the meshgrid
                    x_coord = R*np.cos(Theta)
                    y_coord = R*np.sin(Theta)
                    z_coord = h

                    # # Rotate the cylinder to the correct orientation
                    vec = np.array([x_coord, y_coord, z_coord])
                    vec = np.dot(Rot, vec)
                    x.append(vec[0] + self.domain_params.source[0])
                    y.append(vec[1] + self.domain_params.source[1])
                    z.append(vec[2] + self.domain_params.source[2])

        # Flatten the arrays to create a list of (x, y, z) points
        points = np.column_stack([np.array(x), np.array(y), np.array(z)])

        return points

    def _create_cone(self) -> np.ndarray:
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

        r_values = np.linspace(200, self.domain_params.r, self.domain_params.resolution)
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

    def _create_z_limited_cone(self) -> np.ndarray:

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
    
    def _create_two_D(self) -> np.ndarray:
        """
        Creates a 2D domain of points.

        Required parameters
            - x_min (float): The minimum x value.
            - x_max (float): The maximum x value.
            - y_min (float): The minimum y value.
            - y_max (float): The maximum y value.
            - resolution (int): The resolution of the domain.

        Returns:
            - points (numpy.ndarray): The generated 2D points.
        """

        required_params = ['x_min', 'x_max', 'y_min', 'y_max', 'resolution']
        self._check_required_params(required_params)

        x_values = np.linspace(self.domain_params.x_min, self.domain_params.x_max, self.domain_params.resolution)
        y_values = np.linspace(self.domain_params.y_min, self.domain_params.y_max, self.domain_params.resolution)

        X, Y = np.meshgrid(x_values, y_values)

        points = np.column_stack([X.flatten(), Y.flatten()])
        return points
    
    def visualise_points(self, points):
        """
        Visualises the domain points.

        Args:
            - points (numpy.ndarray): The domain points to visualise.
        """

        if self.n_dims == 1:
            plt.plot(points, np.zeros(points.shape), 'o')
            plt.show()
        elif self.n_dims == 2:
            plt.plot(points[:, 0], points[:, 1], 'o')
            plt.show()
        elif self.n_dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], '.', s=5)
            plt.show()