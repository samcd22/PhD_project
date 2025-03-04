from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared,
    WhiteKernel, ConstantKernel, Product, Sum
)
from sklearn.gaussian_process.kernels import Kernel as GP_Kernel
import numpy as np

class Kernel:
    """
    A class to handle one or multiple kernels across different dimensions.

    Attributes:
        - kernel_config (dict): A dictionary where keys are kernel instances and values are lists of dimensions they apply to.
        - kernel_params (dict): A dictionary where keys are kernel instances and values are dictionaries of their parameters.
        - kernel_concatination (string): How to combine kernels ('+' for addition, '*' for multiplication)
        - dirty (bool): Whether the kernel needs to be rebuilt.
        - kernel (sklearn.gaussian_process.kernels.Kernel): The constructed kernel.
    """

    def __init__(self, kernel_config, kernel_concatination='*'):
        """
        Initialize the Kernel class to handle one or multiple kernels across different dimensions.

        Args:
            - kernel_config (dict): A dictionary where keys are kernel instances and values are lists of dimensions they apply to.
            - kernel_concatination (string): How to combine kernels ('+' for addition, '*' for multiplication)
        """
        self.kernel_config = kernel_config
        self.kernel_params = {}
        self.kernel_concatination = kernel_concatination
        self.dirty = True
        self.kernel = None
        self._validate_kernel_config()

    def _validate_kernel_config(self):
        """Ensures kernels are valid for multivariate cases."""
        multivariate_kernels = {'rbf', 'matern', 'rq'}
        for (kernel_name, identifier), dims in self.kernel_config.items():
            if kernel_name not in multivariate_kernels and len(dims) > 1:
                raise ValueError(f"Kernel '{kernel_name}' does not support multiple dimensions. Assign it to a single variable.")

    def _validate_kernel_params(self):
        """Validates the provided kernel parameters against the expected parameters for each kernel."""
        valid_params = {
            'rbf': ['length_scale'],
            'matern': ['length_scale', 'nu'],
            'rq': ['length_scale', 'alpha'],
            'linear': [],
            'periodic': ['length_scale', 'periodicity'],
            'cosine': ['length_scale', 'periodicity'],
            'white': ['noise_level'],
            'constant': ['constant_value']
        }
        
        for (kernel_name, identifier), params in self.kernel_params.items():
            if kernel_name not in valid_params:
                raise ValueError(f"Invalid kernel type '{kernel_name}' in parameters.")
            
            for param in params.keys():
                # Allow bounds parameters (e.g., "length_scale_bounds", "noise_level_bounds")
                if param not in valid_params[kernel_name] and not param.endswith("_bounds"):
                    raise ValueError(f"Invalid parameter '{param}' for kernel '{kernel_name}'. Expected parameters: {valid_params[kernel_name]}")

    def add_kernel_param(self, kernel_name, identifier, param_name, val):
        """
        Adds a parameter to a specific kernel instance and marks it for rebuilding.

        Args:
            - kernel_name (string): The type of kernel to add the parameter to.
            - identifier (string): A unique identifier for the kernel instance.
            - param_name (string): The name of the parameter to add.
            - val (float, list, or tuple): The value of the parameter. Bounds should be passed as a list or tuple of (min, max).
        
        Returns:
            - self: The Kernel instance with the added parameter
        """
        key = (kernel_name, identifier)
        if key not in self.kernel_params:
            self.kernel_params[key] = {}

        # Ensure bounds are stored as tuples, as required by sklearn
        if "bounds" in param_name and isinstance(val, (list, tuple)) and len(val) == 2:
            val = tuple(val)  # Convert to tuple for sklearn compatibility

        self.kernel_params[key][param_name] = val
        self.dirty = True
        return self

    class _KernelTransformer(GP_Kernel):
        """Wraps a kernel to apply only to specific dimensions of X."""

        def __init__(self, base_kernel, dims):
            self.base_kernel = base_kernel
            self.dims = dims

        def __call__(self, X, Y=None, eval_gradient=False):
            X_subset = X[:, self.dims]
            if Y is not None:
                Y_subset = Y[:, self.dims]
            else:
                Y_subset = None
            return self.base_kernel(X_subset, Y_subset, eval_gradient)

        def diag(self, X):
            return self.base_kernel.diag(X[:, self.dims])

        def is_stationary(self):
            return self.base_kernel.is_stationary()

        @property
        def hyperparameters(self):
            return self.base_kernel.hyperparameters

        @property
        def theta(self):
            return self.base_kernel.theta

        @theta.setter
        def theta(self, theta):
            self.base_kernel.theta = theta

        @property
        def bounds(self):
            return self.base_kernel.bounds

        @property
        def requires_vector_input(self):
            return self.base_kernel.requires_vector_input

        @property
        def n_dims(self):
            return len(self.dims)

        def get_params(self, deep=True):
            return {"base_kernel": self.base_kernel, "dims": self.dims}

        def set_params(self, **params):
            if "base_kernel" in params:
                self.base_kernel = params["base_kernel"]
            if "dims" in params:
                self.dims = params["dims"]

    def _build_kernel(self):
        """Constructs the combined kernel, ensuring separate kernels are applied correctly."""
        self._validate_kernel_params()

        kernel_map = {
            'rbf': lambda params: RBF(**params),
            'matern': lambda params: Matern(**params),
            'rq': lambda params: RationalQuadratic(**params),
            'linear': lambda params: DotProduct(**params),
            'periodic': lambda params: ExpSineSquared(**params),
            'cosine': lambda params: ExpSineSquared(**params),
            'white': lambda params: WhiteKernel(**params),
            'constant': lambda params: ConstantKernel(**params)
        }

        # Group kernels by their assigned dimensions
        grouped_kernels = {}
        for (k, identifier), dims in self.kernel_config.items():
            if k in kernel_map:
                params = self.kernel_params.get((k, identifier), {})
                kernel = kernel_map[k](params)
                transformed_kernel = self._KernelTransformer(kernel, dims)

                dims_tuple = tuple(dims)  # Convert list to tuple for dict key
                if dims_tuple in grouped_kernels:
                    grouped_kernels[dims_tuple] = Sum(grouped_kernels[dims_tuple], transformed_kernel)
                else:
                    grouped_kernels[dims_tuple] = transformed_kernel
            else:
                raise ValueError(f"Invalid kernel type '{k}'.")

        # Combine kernels across different dimensions
        grouped_kernels_list = list(grouped_kernels.values())
        self.kernel = grouped_kernels_list[0]
        for k in grouped_kernels_list[1:]:
            self.kernel = Product(self.kernel, k) if self.kernel_concatination == '*' else Sum(self.kernel, k)

        self.dirty = False

    def get_kernel(self):
        """Returns the constructed kernel, rebuilding if necessary."""
        if self.dirty or self.kernel is None:
            self._build_kernel()
        return self.kernel

    def __repr__(self):
        return f"Kernel({self.kernel})"
    
    def get_construction(self):
        """
        Get the construction of the kernel. The construction parameters includes all of the config information used to construct the kernel object. It includes:
            - kernel_config: A dictionary where keys are kernel instances and values are lists of dimensions they apply to.
            - kernel_params: A dictionary where keys are kernel instances and values are dictionaries of their parameters.
            - kernel_concatination: How to combine kernels ('+' for addition, '*' for multiplication)
        """
        construction = {
            'kernel_config': self.kernel_config,
            'kernel_params': self.kernel_params,
            'kernel_concatination': self.kernel_concatination
        }

        return construction