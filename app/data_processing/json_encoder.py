"""JSON encoder for NumPy and JAX scalars and arrays.

Replaces the external ``numpyencoder`` package, whose 0.3.0 release references
``np.float_`` and ``np.complex_``, both removed in NumPy 2.
"""
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """``json.dumps(obj, cls=NumpyEncoder)`` handles NumPy and JAX types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.complexfloating):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__array__"):  # jax arrays and other array-likes
            return np.asarray(obj).tolist()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)
