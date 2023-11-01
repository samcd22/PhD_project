import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_solve, solve_triangular
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution

from jax import lax
from numpyro.distributions.util import (
    cholesky_of_inverse,
    lazy_property,
    promote_shapes,
    validate_sample,
)
from numpyro.util import is_prng_key



class MultivariateLogNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covariance_matrix",
        "precision_matrix",
        "scale_tril",
    ]

    def __init__(
        self,
        loc=0.0,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]
        if covariance_matrix is not None:
            loc, self.covariance_matrix = promote_shapes(loc, covariance_matrix)
            self.scale_tril = jnp.linalg.cholesky(self.covariance_matrix)
        elif precision_matrix is not None:
            loc, self.precision_matrix = promote_shapes(loc, precision_matrix)
            self.scale_tril = cholesky_of_inverse(self.precision_matrix)
        elif scale_tril is not None:
            loc, self.scale_tril = promote_shapes(loc, scale_tril)
        else:
            raise ValueError(
                "One of `covariance_matrix`, `precision_matrix`, `scale_tril`"
                " must be specified."
            )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-2], jnp.shape(self.scale_tril)[:-2]
        )
        event_shape = jnp.shape(self.scale_tril)[-1:]
        self.loc = loc[..., 0]
        super(MultivariateLogNormal, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + jnp.squeeze(
            jnp.matmul(self.scale_tril, eps[..., jnp.newaxis]), axis=-1
        )


    @validate_sample
    def log_prob(self, value):
        M = _batch_mahalanobis(self.scale_tril, value - self.loc)
        half_log_det = jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(
            -1
        )
        normalize_term = half_log_det + 0.5 * self.scale_tril.shape[-1] * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * M - normalize_term

    @lazy_property
    def covariance_matrix(self):
        return jnp.matmul(self.scale_tril, jnp.swapaxes(self.scale_tril, -1, -2))


    @lazy_property
    def precision_matrix(self):
        identity = jnp.broadcast_to(
            jnp.eye(self.scale_tril.shape[-1]), self.scale_tril.shape
        )
        return cho_solve((self.scale_tril, True), identity)


    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.shape())

    @property
    def variance(self):
        return jnp.broadcast_to(
            jnp.sum(self.scale_tril**2, axis=-1), self.batch_shape + self.event_shape
        )

    @staticmethod
    def infer_shapes(
        loc=(), covariance_matrix=None, precision_matrix=None, scale_tril=None
    ):
        batch_shape, event_shape = loc[:-1], loc[-1:]
        for matrix in [covariance_matrix, precision_matrix, scale_tril]:
            if matrix is not None:
                batch_shape = lax.broadcast_shapes(batch_shape, matrix[:-2])
                event_shape = lax.broadcast_shapes(event_shape, matrix[-1:])
        return batch_shape, event_shape

def _batch_mahalanobis(bL, bx):
    if bL.shape[:-1] == bx.shape:
        # no need to use the below optimization procedure
        solve_bL_bx = solve_triangular(bL, bx[..., None], lower=True).squeeze(-1)
        return jnp.sum(jnp.square(solve_bL_bx), -1)

    # NB: The following procedure handles the case: bL.shape = (i, 1, n, n), bx.shape = (i, j, n)
    # because we don't want to broadcast bL to the shape (i, j, n, n).

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tril_solve
    sample_ndim = bx.ndim - bL.ndim + 1  # size of sample_shape
    out_shape = jnp.shape(bx)[:-1]  # shape of output
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = out_shape[:sample_ndim]
    for sL, sx in zip(bL.shape[:-2], out_shape[sample_ndim:]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (-1,)
    bx = jnp.reshape(bx, bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        tuple(range(sample_ndim))
        + tuple(range(sample_ndim, bx.ndim - 1, 2))
        + tuple(range(sample_ndim + 1, bx.ndim - 1, 2))
        + (bx.ndim - 1,)
    )
    bx = jnp.transpose(bx, permute_dims)

    # reshape to (-1, i, 1, n)
    xt = jnp.reshape(bx, (-1,) + bL.shape[:-1])
    # permute to (i, 1, n, -1)
    xt = jnp.moveaxis(xt, 0, -1)
    solve_bL_bx = solve_triangular(bL, xt, lower=True)  # shape: (i, 1, n, -1)
    M = jnp.sum(solve_bL_bx**2, axis=-2)  # shape: (i, 1, -1)
    # permute back to (-1, i, 1)
    M = jnp.moveaxis(M, -1, 0)
    # reshape back to (..., 1, j, i, 1)
    M = jnp.reshape(M, bx.shape[:-1])
    # permute back to (..., 1, i, j, 1)
    permute_inv_dims = tuple(range(sample_ndim))
    for i in range(bL.ndim - 2):
        permute_inv_dims += (sample_ndim + i, len(out_shape) + i)
    M = jnp.transpose(M, permute_inv_dims)
    return jnp.reshape(M, out_shape)