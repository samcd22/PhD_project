import numpy as np

class Transformation:
    def __init__(self, transformation_type=None):
        self.transformation_type = transformation_type

    def transform(self, X):
        if self.transformation_type == 'log':
            return np.log(X)
        elif self.transformation_type == 'sqrt':
            return np.sqrt(X)
        elif self.transformation_type == 'square':
            return X**2
        elif self.transformation_type == 'cube':
            return X**3
        elif self.transformation_type == 'exp':
            return np.exp(X)
        elif self.transformation_type == 'logp1':
            return np.log1p(X)
        elif self.transformation_type == None:
            return X
        else:
            raise ValueError(f"Invalid transformation type: {self.transformation_type}")
        
    def inverse_transform(self, X):
        if self.transformation_type == 'log':
            return np.exp(X)
        elif self.transformation_type == 'sqrt':
            return X**2
        elif self.transformation_type == 'square':
            return np.sqrt(X)
        elif self.transformation_type == 'cube':
            return X**(1/3)
        elif self.transformation_type == 'exp':
            return np.log(X)
        elif self.transformation_type == 'logp1':
            return np.expm1(X)
        elif self.transformation_type == None:
            return X
        else:
            raise ValueError(f"Invalid transformation type: {self.transformation_type}")
        
    