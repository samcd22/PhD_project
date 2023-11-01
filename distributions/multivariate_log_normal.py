from numpyro.distributions import Distribution
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

class MultivariateLogNormal(Distribution):
    # class attributes
    arg_constraints = {}
    support = None

    def __init__(self, locs = [1,1], scales = [[1,0],[0,1]], validate_args = None):
        self.locs = np.array(locs)
        self.scales = np.array(scales)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.locs),
            jnp.shape(self.scales),
        )
        self.n_dim = len(locs)
        self.get_alpha_beta()

        super().__init__(batch_shape, validate_args=validate_args)

    def get_alpha_beta(self):
        self.alpha = np.zeros(len(self.locs))
        self.beta = np.zeros((len(self.locs), len(self.locs)))

        for i in range(len(self.locs)):
            self.alpha[i] = np.log(self.locs[i]) - 1/2*np.log(self.scales[i,i]/self.locs[i]**2 +1)
            for j in range(len(self.locs)):
                self.beta[i,j] = np.log(self.scales[i,j]/(self.locs[i]*self.locs[j])+1)
        self.alpha = jnp.array(self.alpha)
        self.beta = jnp.array(self.beta)

    def log_prob(self, value):
        log_prob = 1
        for i in range(self.n_dim):
            log_prob *= (2*jnp.pi)**(-self.n_dim/2)*jnp.linalg.det(self.beta)**(-1/2)*1/value[i]*jnp.exp(-1.2*(jnp.log(value)-self.alpha).T@jnp.linalg.inv(self.beta)@(jnp.log(value)-self.alpha))
            log_prob *= 1/value[i] 
        return np.log(log_prob)
    
    def sample(self, key, sample_shape=()):
        u = np.exp(np.random.multivariate_normal(self.alpha, self.scales))
        return u