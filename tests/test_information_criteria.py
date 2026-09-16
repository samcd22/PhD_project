import numpy as np


def test_aic_and_bic_use_twice_the_log_likelihood():
    # guards the visualiser formulas: AIC = 2k - 2 lnL, BIC = k ln n - 2 lnL
    src = open(__file__.replace('tests/test_information_criteria.py', 'app/visualisation_toolbox/visualiser.py')).read()
    live = [l for l in src.splitlines() if 'self.AIC =' in l and not l.strip().startswith('#')]
    assert live, 'no live AIC assignment found'
    assert all('- 2*log_likelihood' in l for l in live), live
    bic = [l for l in src.splitlines() if 'self.BIC =' in l and not l.strip().startswith('#')]
    assert all('- 2*log_likelihood' in l for l in bic), bic
