import numpy as np

class Transformation:
    """
    A class to handle transformations for Gaussian Process Regression, including:
    - Applying transformations to data
    - Adjusting uncertainty (alpha) for Gaussian Processes
    - Performing inverse transformations on predictions and uncertainty

    Supported transformations:
    - 'log'      : Log transformation (with offset to avoid log(≤0))
    - 'log1p'    : Log(x + 1) transformation
    - 'sqrt'     : Square root transformation
    - 'identity' : No transformation

    Attributes:
        transformation_type (str): Type of transformation applied
    """

    def __init__(self, transformation_type="identity"):
        self.transformation_type = transformation_type

        # Ensure valid transformation type
        valid_transforms = ["log", "log1p", "sqrt", "identity"]
        if transformation_type not in valid_transforms:
            raise ValueError(f"Unsupported transformation '{transformation_type}'. Must be one of {valid_transforms}.")

    ## ---------------------- APPLY TRANSFORMATION ---------------------- ##
    def transform(self, y):
        """
        Apply the chosen transformation to y.
        Args:
            y (array-like): Data to transform.
        Returns:
            y_transformed (array-like): Transformed data.
        """
        y = np.array(y)

        if self.transformation_type == "log":
            if np.any(y <= 0):
                raise ValueError("Log transformation error: Data contains non-positive values.")
            return np.log(y)

        elif self.transformation_type == "log1p":
            if np.any(y < 0):
                raise ValueError("Log1p transformation error: Data contains negative values.")
            return np.log1p(y)  # log(y + 1), safe for y = 0

        elif self.transformation_type == "sqrt":
            if np.any(y < 0):
                raise ValueError("Square root transformation error: Data contains negative values.")
            return np.sqrt(y)

        return y  # Identity transformation (no change)

    def inverse_transform(self, y_transformed, std_transformed=None):
        """
        Apply the inverse transformation to get predictions in original space.
        Also adjusts standard deviation if provided.

        Args:
            y_transformed (array-like): Transformed predictions.
            std_transformed (array-like, optional): Transformed standard deviations.

        Returns:
            y_original (array-like): Inverse transformed predictions.
            std_original (array-like, optional): Adjusted standard deviations.
        """
        y_transformed = np.array(y_transformed)
        std_transformed = np.array(std_transformed) if std_transformed is not None else None

        if self.transformation_type == "log":
            # Correct inverse transformation for log
            y_original = np.exp(y_transformed)
            std_original = y_original * std_transformed if std_transformed is not None else None

        elif self.transformation_type == "log1p":
            # Correct inverse transformation for log1p
            y_original = np.expm1(y_transformed)
            std_original = (y_original + 1) * std_transformed if std_transformed is not None else None

        elif self.transformation_type == "sqrt":
            # Correct inverse transformation for sqrt
            y_original = y_transformed**2
            std_original = 2 * y_transformed * std_transformed if std_transformed is not None else None

        else:  # Identity transformation (no change)
            y_original = y_transformed
            std_original = std_transformed

        return (y_original, std_original) if std_transformed is not None else y_original

    ## ---------------------- TRANSFORM UNCERTAINTY (ALPHA) ---------------------- ##
    def transform_alpha(self, y, alpha):
        """
        Adjusts alpha (uncertainty) for the transformed space.

        Args:
            y (array-like): Original y values (before transformation).
            alpha (array-like): Original noise variances.

        Returns:
            alpha_transformed (array-like): Transformed noise variances.
        """
        y = np.array(y)
        alpha = np.array(alpha)

        if self.transformation_type == "log":
            return (1 / (y + 1e-6)) ** 2 * alpha  # Avoid division by zero

        elif self.transformation_type == "log1p":
            return (1 / (y + 1)) ** 2 * alpha  # Avoid issues with log1p

        elif self.transformation_type == "sqrt":
            return (1 / (2 * np.sqrt(y))) ** 2 * alpha

        return alpha  # Identity transformation (no change)
