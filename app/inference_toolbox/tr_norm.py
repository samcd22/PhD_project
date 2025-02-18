import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform

class TruncatedNormal(dist.Distribution):
    arg_constraints = {"loc": dist.constraints.real, "scale": dist.constraints.positive}
    support = dist.constraints.positive
    has_rsample = True

    def __init__(self, loc=0.0, scale=1.0, low=0.0, high=jnp.inf, validate_args=None):
        self.base_dist = dist.Normal(loc, scale)
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self._transform = AffineTransform(loc, scale)
        self._low_cdf = self.base_dist.cdf((low - loc) / scale)
        self._high_cdf = self.base_dist.cdf((high - loc) / scale)
        super().__init__(batch_shape=jnp.shape(loc), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = numpyro.distributions.util.clamp_probs(
            jax.random.uniform(key, shape=sample_shape + self.batch_shape)
        )
        p = self._low_cdf + u * (self._high_cdf - self._low_cdf)
        return self._transform(self.base_dist.icdf(p))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value)
        log_prob = log_prob - jnp.log(self._high_cdf - self._low_cdf)
        return jnp.where((value >= self.low) & (value <= self.high), log_prob, -jnp.inf)

    @property
    def mean(self):
        z_low = (self.low - self.base_dist.loc) / self.base_dist.scale
        z_high = (self.high - self.base_dist.loc) / self.base_dist.scale
        normal_mean = self.base_dist.loc + self.base_dist.scale * (
            dist.Normal(0, 1).pdf(z_low) - dist.Normal(0, 1).pdf(z_high)
        ) / (self._high_cdf - self._low_cdf)
        return normal_mean

    @property
    def variance(self):
        raise NotImplementedError("Variance calculation for truncated normal is not yet implemented.")
