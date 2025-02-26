import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.path import Path


class Domain:
    """
    Domain class for generating and manipulating points in 1D, 2D, and 3D domains.
    
    Attributes:
        - n_dims: Number of dimensions (1, 2, or 3)
        - domain_select: Type of domain to generate (linear, log, rectangular, circular, triangular, polygonal, cuboidal, cylindrical, spherical)
        - dim_names: Names of dimensions (default: ['x', 'y', 'z'])
        - transformations: List of transformations applied to the domain
        - cuts: List of cuts made to the domain
        - domain_dict: Dictionary of valid domain types for each dimension
        - domain_params: Dictionary of domain parameters
        - points: DataFrame of generated points
        - cross_sections: List of cross sections made to the domain

    Methods:
        - add_domain_param: Add a parameter to the domain for generating points
        - build_domain: Build the domain based on the specified domain type and parameters
        - cut_domain: Cut the domain based on a point and direction
        - transform_domain: Transform the domain using a transformation matrix
        - apply_cross_section: Generate a cross section of the domain based on a point, normal direction, and width
        - plot_cross_sections: Plot the cross sections of the domain
        - plot_domain: Plot the generated points in the domain
        - get_construction: Get the construction parameters for the domain

    Domain Types:
        - 1D:
            - linear: Linear domain with min, max, and number of points
            - log: Logarithmic domain with min, max, and number of points
        - 2D:
            - rectangular: Rectangular domain with min/max x/y, and number of points in x/y
            - circular: Circular domain with radius, mass, and optional center/angle
            - triangular: Triangular domain with 3 vertices and mass
            - polygonal: Polygonal domain with vertices and mass
        - 3D:
            - cuboidal: Cuboidal domain with min/max x/y/z, and number of points in x/y/z
            - cylindrical: Cylindrical domain with radius, height, mass, and optional center/angle
            - spherical: Spherical domain with radius, mass, and optional center/theta/phi
    """
    def __init__(self, n_dims, domain_select, dim_names = ['x', 'y', 'z'], time_array = None):
        """
        Initialize the Domain class with the specified number of dimensions and domain type.

        Parameters:
            - n_dims (int): Number of dimensions (1, 2, or 3)
            - domain_select (str): Type of domain to generate
            - dim_names (list): Names of dimensions (default: ['x', 'y', 'z'])
            - time_array (list): Time array for time-varying domains (default: None)
        """
        self.n_dims = n_dims
        self.domain_select = domain_select
        self.dim_names = dim_names
        self.transformations = []
        self.cuts = []
        self.cross_sections = []

        self.domain_dict = {
            1: ['linear', 'log'],
            2: ['rectangular', 'circular', 'triangular', 'polygonal'],
            3: ['cuboidal', 'cylindrical', 'spherical']
        }

        self.domain_params = {}
        self.points = None

        self.time_array = time_array
            

    def add_domain_param(self, param_name, param_value):
        """
        Add a parameter to the domain for generating points.

        Parameters:
            - param_name (str): Name of the parameter
            - param_value: Value of the parameter
        """
        self.domain_params[param_name] = param_value
        return self

    def _check_domain(self):
        if self.domain_select not in self.domain_dict[self.n_dims]:
            raise ValueError('Domain - Invalid domain selected')

    def _check_domain_params(self, required_params):
        for param in required_params:
            if param not in self.domain_params:
                raise ValueError(f'Domain - Missing parameter: {param}')

    def build_domain(self):
        """
        Build the domain based on the specified domain type and parameters.

        Returns:
            - points (DataFrame): DataFrame of generated points
        """
        self._check_domain()

        if self.domain_select == 'linear':
            points = self._linear_domain()
        elif self.domain_select == 'log':
            points = self._log_domain()
        elif self.domain_select == 'rectangular':
            points = self._rectanglular_domain()
        elif self.domain_select == 'circular':
            points = self._circular_domain()
        elif self.domain_select == 'triangular':
            points = self._triangular_domain()
        elif self.domain_select == 'polygonal':
            points = self._polygonal_domain()
        elif self.domain_select == 'cuboidal':
            points = self._cuboidal_domain()
        elif self.domain_select == 'cylindrical':
            points = self._cylindrical_domain()
        elif self.domain_select == 'spherical':
            points = self._spherical_domain()
        self.points = points

        return points

    """
    One dimensional domains
    """

    def _linear_domain(self):
        required_params = ['min', 'max', 'n_points']
        self._check_domain_params(required_params)
        x_points = np.linspace(self.domain_params['min'], self.domain_params['max'], self.domain_params['n_points'])
        return pd.DataFrame(x_points, columns = [self.dim_names[0]])
    
    def _log_domain(self):
        required_params = ['min', 'max', 'n_points']
        self._check_domain_params(required_params)
        x_points = np.logspace(self.domain_params['min'], self.domain_params['max'], self.domain_params['n_points']) 
        return pd.DataFrame(x_points, columns = [self.dim_names[0]])
    
    """
    Two dimensional domains
    """

    def _rectanglular_domain(self):
        required_params = ['min_x', 'max_x', 'n_points_x', 'min_y', 'max_y', 'n_points_y']
        self._check_domain_params(required_params)
        x_points = np.linspace(self.domain_params['min_x'], self.domain_params['max_x'], self.domain_params['n_points_x'])
        y_points = np.linspace(self.domain_params['min_y'], self.domain_params['max_y'], self.domain_params['n_points_y'])
        return pd.DataFrame(np.array(np.meshgrid(x_points, y_points)).T.reshape(-1, 2), columns = self.dim_names[:2])

    def _circular_domain(self):
        required_params = ['radius', 'mass']
        self._check_domain_params(required_params)
        
        # Get parameters with default values
        center = np.array(self.domain_params.get('center', [0, 0]))
        angle = np.array(self.domain_params.get('angle', [0, 2 * np.pi]))  # Full circle by default
        radius = self.domain_params['radius']
        
        # Define proper hexagonal grid spacing
        total_area = np.pi * radius**2 * ((angle[1] - angle[0]) / (2 * np.pi))  # Adjust for circular segments
        point_density = self.domain_params['mass'] / total_area  # Mass acts as the density parameter
        hex_spacing = np.sqrt(1 / point_density)  # Uniform spacing based on density
        hex_height = hex_spacing * np.sqrt(3) / 2  # Maintain hexagonal proportions

        # Create hexagonal grid
        x_coords = []
        y_coords = []
        
        for i in np.arange(-radius, radius + hex_spacing, hex_spacing):  # Extend slightly for coverage
            for j in np.arange(-radius, radius + hex_height, hex_height):
                x = i + (hex_spacing / 2 if int(j / hex_height) % 2 == 1 else 0)
                y = j

                # Convert to polar coordinates
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)

                # Map theta to [0, 2π]
                if theta < 0:
                    theta += 2 * np.pi

                # Ensure full-circle coverage
                if r <= radius and angle[0] <= theta <= angle[1]:  
                    x_coords.append(center[0] + x)
                    y_coords.append(center[1] + y)

        # Convert to DataFrame
        self.points = pd.DataFrame({
            self.dim_names[0]: x_coords,
            self.dim_names[1]: y_coords
        })

        return self.points

    def _triangular_domain(self):
        required_params = ['coord1', 'coord2', 'coord3', 'mass']
        self._check_domain_params(required_params)

        # Extract triangle vertices
        p1 = np.array(self.domain_params['coord1'])
        p2 = np.array(self.domain_params['coord2'])
        p3 = np.array(self.domain_params['coord3'])

        # Compute triangle bounding box
        min_x, max_x = min(p1[0], p2[0], p3[0]), max(p1[0], p2[0], p3[0])
        min_y, max_y = min(p1[1], p2[1], p3[1]), max(p1[1], p2[1], p3[1])

        # Compute triangle area and estimate spacing for hexagonal packing
        triangle_area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))  # Triangle area formula
        point_density = self.domain_params['mass'] / triangle_area  # Points per unit area
        hex_spacing = np.sqrt(1 / point_density)  # Uniform spacing
        hex_height = hex_spacing * np.sqrt(3) / 2  # Vertical spacing

        # Generate hexagonal grid within bounding box
        x_coords = []
        y_coords = []

        for y in np.arange(min_y, max_y + hex_height, hex_height):
            row_offset = 0 if int((y - min_y) / hex_height) % 2 == 0 else hex_spacing / 2
            for x in np.arange(min_x, max_x + hex_spacing, hex_spacing):
                x_adj = x + row_offset  # Adjust x for staggered rows

                # Convert to barycentric coordinates
                v0, v1, v2 = p3 - p1, p2 - p1, np.array([x_adj, y]) - p1
                dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
                dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)

                inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom

                # Check if point is inside the triangle (allow slight floating point errors)
                if u >= -1e-8 and v >= -1e-8 and (u + v) <= 1 + 1e-8:
                    x_coords.append(x_adj)
                    y_coords.append(y)

        # Convert to DataFrame
        self.points = pd.DataFrame({
            self.dim_names[0]: x_coords,
            self.dim_names[1]: y_coords
        })

        return self.points

    def _polygonal_domain(self):
        required_params = ['vertices', 'mass']
        self._check_domain_params(required_params)

        # Extract polygon vertices
        vertices = np.array(self.domain_params['vertices'])  # Shape: (N, 2)
        if len(vertices) < 3:
            raise ValueError("Invalid polygon: A polygon must have at least 3 vertices.")

        # Compute bounding box
        min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])

        # Compute area estimate and spacing for hexagonal packing
        area_estimate = 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))  # Shoelace formula
        point_density = max(10, self.domain_params['mass'] / area_estimate)  # Prevents too small hex spacing
        hex_spacing = max(0.01, np.sqrt(1 / point_density))  # Hexagonal grid spacing
        hex_height = hex_spacing * np.sqrt(3) / 2  # Maintain hex proportions

        # Create hexagonal grid within bounding box
        x_coords = []
        y_coords = []
        polygon_path = Path(vertices)  # Create a Path object for point containment checks

        for y in np.arange(min_y, max_y + hex_height, hex_height):
            row_offset = 0 if int((y - min_y) / hex_height) % 2 == 0 else hex_spacing / 2
            for x in np.arange(min_x, max_x + hex_spacing, hex_spacing):
                x_adj = x + row_offset  # Adjust x for staggered rows
                
                # Check if point is inside the polygon
                if polygon_path.contains_point([x_adj, y]):
                    x_coords.append(x_adj)
                    y_coords.append(y)

        # Convert to DataFrame
        self.points = pd.DataFrame({
            self.dim_names[0]: x_coords,
            self.dim_names[1]: y_coords
        })

        return self.points

    """
    Three dimensional domains
    """

    def _cuboidal_domain(self):
        required_params = ['min_x', 'max_x', 'n_points_x', 'min_y', 'max_y', 'n_points_y', 'min_z', 'max_z', 'n_points_z']
        self._check_domain_params(required_params)
        x_points = np.linspace(self.domain_params['min_x'], self.domain_params['max_x'], self.domain_params['n_points_x'])
        y_points = np.linspace(self.domain_params['min_y'], self.domain_params['max_y'], self.domain_params['n_points_y'])
        z_points = np.linspace(self.domain_params['min_z'], self.domain_params['max_z'], self.domain_params['n_points_z'])
        return pd.DataFrame(np.array(np.meshgrid(x_points, y_points, z_points)).T.reshape(-1, 3), columns = self.dim_names[:3])

    def _cylindrical_domain(self):
        required_params = ['radius', 'height', 'mass']
        optional_params = ['center', 'angle']
        self._check_domain_params(required_params)

        # Extract parameters
        radius = self.domain_params['radius']
        height = self.domain_params['height']
        mass = self.domain_params['mass']
        center = np.array(self.domain_params.get('center', [0, 0, 0]))
        angle = np.array(self.domain_params.get('angle', [0, 2 * np.pi]))  # Default: full cylinder

        # Compute total volume and estimate spacing for hexagonal packing
        volume_fraction = (angle[1] - angle[0]) / (2 * np.pi)  # Fraction of full cylinder
        total_volume = np.pi * radius**2 * height * volume_fraction  # Adjusted volume
        point_density = mass / total_volume  # Approximate density per unit volume
        hex_spacing = np.cbrt(1 / point_density)  # 3D hexagonal spacing
        hex_height = hex_spacing * np.sqrt(3) / 2  # Vertical offset for hexagonal packing

        # Initialize point lists
        x_coords = []
        y_coords = []
        z_coords = []

        ### 1️⃣ Generate hexagonal grid for the **full volume**
        for z in np.arange(-height / 2, height / 2 + hex_height, hex_height):  # Move up in height
            for y in np.arange(-radius, radius + hex_height, hex_height):
                for x in np.arange(-radius, radius + hex_spacing, hex_spacing):
                    # Stagger every other row in 3D for better packing
                    if int((y / hex_height) % 2) == 1:
                        x += hex_spacing / 2

                    if int((z / hex_height) % 2) == 1:
                        y += hex_height / 2

                    # Convert (x, y) to polar coordinates
                    r = np.sqrt(x**2 + y**2)
                    theta = np.arctan2(y, x)
                    if theta < 0:  # Ensure theta is in [0, 2π]
                        theta += 2 * np.pi

                    # Check if the point is inside the cylinder and within the angle range
                    if r <= radius and angle[0] <= theta <= angle[1]:
                        x_coords.append(center[0] + x)
                        y_coords.append(center[1] + y)
                        z_coords.append(center[2] + z)

        # Convert to DataFrame
        self.points = pd.DataFrame({
            self.dim_names[0]: x_coords,
            self.dim_names[1]: y_coords,
            self.dim_names[2]: z_coords
        })

        return self.points

    def _spherical_domain(self):
        required_params = ['radius', 'mass']
        optional_params = ['center', 'theta', 'phi']
        self._check_domain_params(required_params)

        # Extract parameters
        radius = self.domain_params['radius']
        mass = self.domain_params['mass']
        center = np.array(self.domain_params.get('center', [0, 0, 0]))

        # Default to full sphere if theta and phi aren't provided
        theta_range = np.array(self.domain_params.get('theta', [-np.pi / 2, np.pi / 2]))  # Elevation (-π/2 to π/2)
        phi_range = np.array(self.domain_params.get('phi', [-np.pi, np.pi]))  # Azimuth (-π to π)

        # 🚨 Raise an error if theta or phi are outside valid ranges
        if np.any(theta_range < -np.pi / 2) or np.any(theta_range > np.pi / 2):
            raise ValueError(f"Invalid theta range: {theta_range}. Must be between [-π/2, π/2].")

        if np.any(phi_range < -np.pi) or np.any(phi_range > np.pi):
            raise ValueError(f"Invalid phi range: {phi_range}. Must be between [-π, π].")

        # Compute volume-based spacing for hexagonal packing
        total_volume = (4 / 3) * np.pi * radius**3  # Sphere volume
        point_density = mass / total_volume  # Density per unit volume
        hex_spacing = np.cbrt(1 / point_density)  # 3D hexagonal spacing

        # Initialize lists for storing points
        x_coords = []
        y_coords = []
        z_coords = []

        theta_vec = []
        phi_vec = []

        ### 1️⃣ Generate tessellated hexagonal points inside sphere
        for z in np.arange(-radius, radius + hex_spacing, hex_spacing):  # Height levels
            for y in np.arange(-radius, radius + hex_spacing, hex_spacing):
                for x in np.arange(-radius, radius + hex_spacing, hex_spacing):

                    # Convert Cartesian coordinates to spherical
                    r = np.sqrt(x**2 + y**2 + z**2)
                    if r == 0 or r > radius:
                        continue  # Avoid division by zero and points outside the sphere
                    
                    theta = np.arcsin(z / r)  # Corrected elevation angle (-π/2 to π/2)

                    phi = np.arctan2(y, x)  # Azimuth angle (-π to π)


                    theta_vec.append(theta)
                    phi_vec.append(phi)

                    if (
                        theta_range[0] <= theta <= theta_range[1] and
                        phi_range[0] <= phi <= phi_range[1]
                    ):
                        x_coords.append(center[0] + x)
                        y_coords.append(center[1] + y)
                        z_coords.append(center[2] + z)

        # Convert to DataFrame
        self.points = pd.DataFrame({
            self.dim_names[0]: x_coords,
            self.dim_names[1]: y_coords,
            self.dim_names[2]: z_coords
        })

        return self.points

    """
    Cut the domain
    """

    def cut_domain(self, point, direction):
        """
        Cut the domain based on a point and direction.

        Parameters:
            - point (list): Point to cut the domain
            - direction (list): Direction to cut the domain

        Returns:
            - points (DataFrame): DataFrame of cut points

        Raises:
            - ValueError: If the domain has no points, invalid dimensions, or invalid point/direction
        """
        direction = np.array(direction)
        point = np.array(point)
        direction = direction / np.linalg.norm(direction)
        cut = {'point': point, 'direction': direction}
        self.cuts.append(cut)
        if self.points is None:
            raise ValueError('Domain - No points to cut')

        if self.n_dims == 1:
            raise ValueError('Domain - Cannot cut 1D domain')
        elif self.n_dims == 2:
            points = self._cut_2d(point, direction)
        elif self.n_dims == 3:
            points = self._cut_3d(point, direction)
        else:
            raise ValueError('Domain - Invalid number of dimensions')
        return points

    def _cut_2d(self, point, direction):
        if point.shape[0] != 2 or direction.shape[0] != 2:
            raise ValueError("Domain - Invalid point or direction vector")
        signed_distance = (self.points[self.dim_names[0]] - point[0]) * direction[1] - (self.points[self.dim_names[1]] - point[1]) * direction[0]
        self.points = self.points[signed_distance <= 0]

        return self.points
    
    def _cut_3d(self, point, direction):
        if point.shape[0] != 3 or direction.shape[0] != 3:
            raise ValueError("Domain - Invalid point or direction vector")
        signed_distance = (self.points[self.dim_names[0]] - point[0]) * direction[0] + (self.points[self.dim_names[1]] - point[1]) * direction[1] + (self.points[self.dim_names[2]] - point[2]) * direction[2]
        self.points = self.points[signed_distance <= 0]

        return self.points

    """
    Transform the domain
    """

    def transform_domain(self, transformation):
        """
        Transform the domain using a transformation matrix.

        Parameters:
            - transformation: Transformation matrix

        Returns:
            - points (DataFrame): DataFrame of transformed points

        Raises:
            - ValueError: If the domain has no points, invalid dimensions, or invalid transformation matrix
        """

        if self.points is None:
            raise ValueError('Domain - No points to transform')

        transformation = np.array(transformation)
        self.transformations.append(transformation)

        if self.n_dims == 1:
            raise ValueError('Domain - Cannot transform 1D domain')
        elif self.n_dims == 2:
            points = self._transform_2d(transformation)
        elif self.n_dims == 3:
            points = self._transform_3d(transformation)
        else:
            raise ValueError('Domain - Invalid number of dimensions')
        return points

    def _transform_2d(self, transformation):
        # Check if the transformation matrix is valid (2x2)
        if transformation.shape != (2, 2):
            raise ValueError("Domain - Transformation matrix must be 2x2")

        # Convert points into a NumPy array (N x 2)
        points_array = self.points[[self.dim_names[0], self.dim_names[1]]].values.T  # Shape (2, N)

        # Apply matrix multiplication
        transformed_points = transformation @ points_array  # Shape (2, N)

        # Update the DataFrame with transformed values
        self.points[self.dim_names[0]], self.points[self.dim_names[1]] = transformed_points[0], transformed_points[1]

        return self.points
        
    def _transform_3d(self, transformation):

        # Check if the transformation matrix is valid (3x3)
        if transformation.shape != (3, 3):
            raise ValueError("Domain - Transformation matrix must be 3x3")

        # Convert points into a NumPy array (N x 3)
        points_array = self.points[[self.dim_names[0], self.dim_names[1], self.dim_names[2]]].values.T

        # Apply matrix multiplication
        transformed_points = transformation @ points_array

        # Update the DataFrame with transformed values
        self.points[self.dim_names[0]], self.points[self.dim_names[1]], self.points[self.dim_names[2]] = transformed_points[0], transformed_points[1], transformed_points[2]

        return self.points

    """
    Cross section
    """
    
    def apply_cross_section(self, point, normal, width):
        """
        Generate a cross section of the domain based on a point, normal direction, and width.

        Parameters:
            - point (list): Point on the plane
            - normal (list): Normal direction of the plane
            - width (float): Width of the cross section

        Returns:
            - cross_section_points (DataFrame): DataFrame of cross section points
            - projected_points (DataFrame): DataFrame of projected points
            
        Raises:
            - ValueError: If the domain has no points, invalid dimensions, or invalid point/normal
        """
        normal = np.array(normal)
        point = np.array(point)
        normal = normal / np.linalg.norm(normal)

        if self.points is None:
            raise ValueError('Domain - No points to cross section')
        
        if self.n_dims == 1:
            raise ValueError('Domain - Cannot cross section 1D domain')
        elif self.n_dims == 2:
            raise ValueError('Domain - Cannot cross section 2D domain')
        elif self.n_dims == 3:
            cross_section_points, projected_points = self._cross_section_3d(point, normal, width)
        else:
            raise ValueError('Domain - Invalid number of dimensions')
        
        cross_section = {'point': point, 'normal': normal, 'width': width, 'points':cross_section_points, 'projected_points': projected_points}
        self.cross_sections.append(cross_section)
        return cross_section_points, projected_points
    
    def _cross_section_3d(self, point, normal, width):
        if point.shape[0] != 3 or normal.shape[0] != 3:
            raise ValueError("Domain - Invalid point or normal vector")

        # Calculate the signed distance of each point from the plane
        signed_distance = (self.points[self.dim_names[0]] - point[0]) * normal[0] + (self.points[self.dim_names[1]] - point[1]) * normal[1] + (self.points[self.dim_names[2]] - point[2]) * normal[2]

        # Filter points within the cross section width
        cross_section_points = self.points[np.abs(signed_distance) <= width]

        # Choose an arbitrary vector that is not collinear with the normal
        arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < abs(normal[2]) else np.array([0, 0, 1])

        # Compute two perpendicular vectors (basis vectors for the plane)
        u = np.cross(arbitrary_vector, normal)
        u = u / np.linalg.norm(u)  # Normalize

        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)  # Normalize

        # Project each point onto the (u, v) basis
        projected_points = np.zeros((len(cross_section_points), 2))
        for i, (_, row) in enumerate(cross_section_points.iterrows()):
            p = np.array([row[self.dim_names[0]], row[self.dim_names[1]], row[self.dim_names[2]]])
            projected_points[i, 0] = np.dot(p - point, u)
            projected_points[i, 1] = np.dot(p - point, v)

        u_name = 'u'
        v_name = 'v'

        if np.array_equal(u, [1,0,0]) or np.array_equal(u, [-1,0,0]):
            u_name = self.dim_names[0]
        elif np.array_equal(u, [0,1,0]) or np.array_equal(u, [0,-1,0]):
            u_name = self.dim_names[1]
        elif np.array_equal(u, [0,0,1]) or np.array_equal(u, [0,0,-1]):
            u_name = self.dim_names[2]

        if np.array_equal(v, [1,0,0]) or np.array_equal(v, [-1,0,0]):
            v_name = self.dim_names[0]
        elif np.array_equal(v, [0,1,0]) or np.array_equal(v, [0,-1,0]):
            v_name = self.dim_names[1]
        elif np.array_equal(v, [0,0,1]) or np.array_equal(v, [0,0,-1]):
            v_name = self.dim_names[2]

        projected_points = pd.DataFrame(projected_points, columns=[u_name, v_name])

        return cross_section_points, projected_points

    def plot_cross_sections(self):
        """
        Plot the cross sections of the domain.
        """
        if self.cross_sections is None:
            raise ValueError('Domain - No cross sections to plot')

        for cross_section in self.cross_sections:
            self._plot_cross_section(cross_section)

    def _plot_cross_section(self, cross_section):
        points = cross_section['points']
        projected_points = cross_section['projected_points']
        ref_point = cross_section['point']

        u_name = projected_points.columns[0]
        v_name = projected_points.columns[1]
        axis_val = None
        fig = plt.figure(figsize=(12, 6))

        # Plot the 3D cross section
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(points[self.dim_names[0]], points[self.dim_names[1]], points[self.dim_names[2]], s=2)
        ax.set_xlabel(self.dim_names[0])
        ax.set_ylabel(self.dim_names[1])
        ax.set_zlabel(self.dim_names[2])
        ax.set_title('3D Cross Section')

        # Plot the 2D cross section
        ax = fig.add_subplot(122)
        ax.scatter(projected_points[u_name], projected_points[v_name], s=2)
        ax.set_xlabel(u_name)
        ax.set_ylabel(v_name)
        title = f'2D Cross Section ({u_name}-{v_name})'
        if (u_name == self.dim_names[0] and v_name == self.dim_names[1]) or (u_name == self.dim_names[1] and v_name == self.dim_names[0]):
            axis_val = {'dim': self.dim_names[2], 'val': ref_point[2]}
        elif (u_name == self.dim_names[0] and v_name == self.dim_names[2]) or (u_name == self.dim_names[2] and v_name == self.dim_names[0]):
            axis_val = {'dim': self.dim_names[1], 'val': ref_point[1]}
        elif (u_name == self.dim_names[1] and v_name == self.dim_names[2]) or (u_name == self.dim_names[2] and v_name == self.dim_names[1]):
            axis_val = {'dim':self. dim_names[0], 'val': ref_point[0]}
        if axis_val:
            title += f' \n {axis_val["dim"]} = {axis_val["val"]}'
        ax.set_title(title)

        plt.show()

    """
    Plotting
    """

    def plot_domain(self):
        """
        Plot the generated points in the domain.
        """
        if self.points is None:
            raise ValueError('Domain - No points to plot')
        
        if self.n_dims == 1:
            self._plot_1d()
        elif self.n_dims == 2:
            self._plot_2d()
        elif self.n_dims == 3:
            self._plot_3d()

    def _plot_1d(self):
        plt.scatter(self.points[self.dim_names[0]], np.zeros(self.points.shape[0]))
        plt.xlabel(self.dim_names[0])
        plt.yticks([])
        plt.gca().yaxis.set_ticklabels([])
        plt.title(f'1D Domain - {self.domain_select.capitalize()}')
        plt.show()

    def _plot_2d(self):
        plt.scatter(self.points[self.dim_names[0]], self.points[self.dim_names[1]])
        plt.xlabel(self.dim_names[0])
        plt.ylabel(self.dim_names[1])
        plt.title(f'2D Domain - {self.domain_select.capitalize()}')
        plt.show()

    def _plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[self.dim_names[0]], self.points[self.dim_names[1]], self.points[self.dim_names[2]], s=2)
        ax.set_xlabel(self.dim_names[0])
        ax.set_ylabel(self.dim_names[1])
        ax.set_zlabel(self.dim_names[2])
        ax.set_title(f'3D Domain - {self.domain_select.capitalize()}')
        plt.show()

    """
    Get construction
    """

    def _convert_arrays_to_lists(self, d):
        """Recursively converts all NumPy arrays in a dictionary or list to lists."""
        if isinstance(d, dict):  # If it's a dictionary, loop through keys
            return {key: self._convert_arrays_to_lists(value) for key, value in d.items()}
        elif isinstance(d, list):  # If it's a list, process each element
            return [self._convert_arrays_to_lists(value) for value in d]
        elif isinstance(d, np.ndarray):  # Convert NumPy arrays to lists
            return d.tolist()
        else:  # Return the value unchanged if it's neither
            return d

    def get_construction(self):
        """
        Get the construction parameters for the domain.

        Returns:
            - construction (dict): Dictionary of construction parameters
        """
        construction = {
            'n_dims': self.n_dims,
            'domain_select': self.domain_select,
            'dim_names': self.dim_names,
            'domain_params': self.domain_params,
            'transformations': self.transformations,
            'cuts': self.cuts,
            'time_array': self.time_array
        }

        return self._convert_arrays_to_lists(construction)

    def _add_time_to_points(self):
        """
        Expands the points DataFrame by adding a time dimension.
        Each unique combination of spatial coordinates will be duplicated for each time step.
        """
        if self.time_array is None:
            raise ValueError('Domain - Time range is not defined')
        if 't' not in self.dim_names:
            raise ValueError('Domain - Time varying domains require a time dimension (t)')

        time_points = self.time_array  # Array of time values
        points = self.points.copy()  # Copy original spatial points DataFrame

        # Repeat each row for every time step
        expanded_points = pd.concat([points.assign(t=t) for t in time_points], ignore_index=True)

        self.points = expanded_points  # Update the points attribute

        return expanded_points  # Optional: Return the modified DataFrame
