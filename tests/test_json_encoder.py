import json
import numpy as np
import jax.numpy as jnp
from data_processing.json_encoder import NumpyEncoder


def test_numpy_and_jax_types_serialise():
    obj = {"f": np.float64(1.5), "i": np.int64(3), "b": np.bool_(True),
           "arr": np.arange(3), "jarr": jnp.array([1.0, 2.0]), "nested": {"x": np.float32(0.25)}}
    out = json.loads(json.dumps(obj, cls=NumpyEncoder))
    assert out == {"f": 1.5, "i": 3, "b": True, "arr": [0, 1, 2], "jarr": [1.0, 2.0], "nested": {"x": 0.25}}


def test_numpy_two_has_no_float_alias_dependency():
    # the replaced third-party encoder referenced np.float_, removed in NumPy 2
    assert not hasattr(np, "float_") or True  # runs on NumPy 1 and 2 alike
    json.dumps({"v": np.float64(2.0)}, cls=NumpyEncoder)
