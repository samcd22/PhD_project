import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from regression_toolbox.parameter import Parameter

PEAKS = [[0.22, 0.2], [0.16, 0.12], [0.11, 0.08], [0.08, 0.06], [0.06, 0.03], [0.04, 0.016]]


def make_mixture():
    p = Parameter(name=['I_y', 'I_z'], prior_select='log_norm', order=0, multi_mode=True)
    return p.add_prior_param('peak', PEAKS).add_prior_param('overall_scale', 0.01)


def test_equal_peak_mixture_builds_eagerly_with_six_modes():
    p = make_mixture()
    dist = p.get_prior_function()
    assert len(dist.component_distributions) == 6
    modes = [np.exp(np.asarray(c.base_dist.loc) - float(c.base_dist.covariance_matrix[0, 0])) for c in dist.component_distributions]
    assert np.allclose(np.array(modes), np.array(PEAKS), rtol=1e-3)
    dens = [float(jnp.exp(dist.log_prob(jnp.array(pk)))) for pk in PEAKS]
    assert max(dens) / min(dens) < 2.0  # peaks equalised to within the mixture overlap


def test_eagerly_built_mixture_samples_under_jit():
    p = make_mixture()
    p.prior_func = p.get_prior_function()  # what Sampler now does at construction

    @jax.jit
    def draw(key):
        with numpyro.handlers.seed(rng_seed=key):
            return p.sample_param()

    val = np.asarray(draw(jax.random.PRNGKey(0)))
    assert val.shape == (2,) and np.all(val > 0)


def test_sampler_builds_priors_at_construction(workdir):
    from regression_toolbox.model import Model
    from regression_toolbox.likelihood import Likelihood
    from regression_toolbox.sampler import Sampler
    from visualisation_toolbox.domain import Domain
    from data_processing.sim_data_processor import SimDataProcessor
    sim_model = Model('line').add_fixed_model_param('a', 1).add_fixed_model_param('b', 1)
    dom = Domain(1, 'linear').add_domain_param('min', 0).add_domain_param('max', 10).add_domain_param('n_points', 20)
    dom.build_domain()
    sdp = SimDataProcessor('prior_build_test', sim_model, dom, noise_dist='gaussian', noise_level=0.1, plot_data=False)
    a = Parameter(name='a', prior_select='gaussian').add_prior_param('mu', 1).add_prior_param('sigma', 1)
    b = Parameter(name='b', prior_select='gaussian').add_prior_param('mu', 1).add_prior_param('sigma', 1)
    sigma = Parameter(name='sigma', prior_select='uniform').add_prior_param('low', 0.0001).add_prior_param('high', 5)
    Sampler(pd.Series({'a': a, 'b': b, 'sigma': sigma}), Model('line'), Likelihood('gaussian'), sdp, n_samples=10, n_chains=1)
    assert all(p.prior_func is not None for p in (a, b, sigma))
