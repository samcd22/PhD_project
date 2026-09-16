import os
import numpy as np
import pandas as pd
from regression_toolbox.model import Model
from regression_toolbox.likelihood import Likelihood
from regression_toolbox.parameter import Parameter
from regression_toolbox.sampler import Sampler
from visualisation_toolbox.domain import Domain
from data_processing.sim_data_processor import SimDataProcessor


def build(workdir, n_samples=300):
    sim_model = Model('line').add_fixed_model_param('a', 2.0).add_fixed_model_param('b', 1.0)
    dom = Domain(1, 'linear').add_domain_param('min', 0).add_domain_param('max', 10).add_domain_param('n_points', 40)
    dom.build_domain()
    sdp = SimDataProcessor('roundtrip', sim_model, dom, noise_dist='gaussian', noise_level=0.2, plot_data=False)
    a = Parameter(name='a', prior_select='gaussian').add_prior_param('mu', 1).add_prior_param('sigma', 2)
    b = Parameter(name='b', prior_select='gaussian').add_prior_param('mu', 1).add_prior_param('sigma', 2)
    sigma = Parameter(name='sigma', prior_select='uniform').add_prior_param('low', 0.0001).add_prior_param('high', 5)
    return Sampler(pd.Series({'a': a, 'b': b, 'sigma': sigma}), Model('line'), Likelihood('gaussian'), sdp,
                   n_samples=n_samples, n_chains=2, p_warmup=0.5)


def test_sample_save_reload_and_resample(workdir):
    s1 = build(workdir)
    assert not s1.data_exists
    samples, chains, fields = s1.sample_all()
    inst = os.path.join(s1.results_path, f'instance_{s1.instance}')
    for f in ('samples.csv', 'chain_samples.csv', 'fields.json', 'sampler_construction.json'):
        assert os.path.exists(os.path.join(inst, f)), f
    assert len(chains) == 2 * 300 and set(chains['chain'].unique()) == {1, 2}
    assert abs(samples['a'].median() - 2.0) < 0.3 and abs(samples['b'].median() - 1.0) < 0.6
    # results land under the working directory, not at the filesystem root
    assert os.path.abspath(s1.results_path).startswith(str(workdir))

    s2 = build(workdir)  # identical configuration: reloads instead of sampling
    assert s2.data_exists and s2.instance == s1.instance
    samples2, _, _ = s2.sample_all()
    pd.testing.assert_frame_equal(samples2.reset_index(drop=True), samples.reset_index(drop=True))

    s3 = build(workdir, n_samples=200)  # different configuration: new instance
    assert not s3.data_exists and s3.instance != s1.instance

    # explicit resample overrides the cached instance
    s4 = build(workdir)
    samples4, _, _ = s4.sample_all(resample=True)
    assert len(samples4) == len(samples)
