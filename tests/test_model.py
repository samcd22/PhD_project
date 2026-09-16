import jax.numpy as jnp
import numpy as np
from regression_toolbox.model import Model


def test_string_model_compiles_to_jax_with_fixed_and_inferred_params():
    model = Model('line').add_fixed_model_param('b', 2.0)
    f = model.get_model()
    x = jnp.array([0.0, 1.0, 2.0])
    y = f({'a': 3.0}, {'x': x})
    assert np.allclose(np.asarray(y), [2.0, 5.0, 8.0])


def test_plume_model_is_registered_and_finite_off_axis():
    model = Model('log_gpm_norm').add_fixed_model_param('H', 5.0).add_fixed_model_param('Q', 3.42e13)
    f = model.get_model()
    pts = {'x': jnp.array([250.0, 1000.0]), 'y': jnp.array([0.0, 300.0]), 'z': jnp.array([12.5, 100.0])}
    out = np.asarray(f({'I_y': 0.8, 'I_z': 0.14}, pts))
    assert out.shape == (2,) and np.all(np.isfinite(out)) and out[0] > out[1]
