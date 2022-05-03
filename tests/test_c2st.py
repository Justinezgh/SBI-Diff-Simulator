import pytest
import jax
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions
from numpy.testing import assert_allclose, assert_equal

from sbids.metrics.c2st import c2st

# distributions
dist1 = tfd.MultivariateNormalDiag(loc=[3,4,5,6],scale_diag=[4,2,1,2])
sample11 = dist1.sample(10000,jax.random.PRNGKey(11))
sample12 = dist1.sample(10000,jax.random.PRNGKey(12))

dist2 = tfd.MultivariateNormalDiag(loc=[3,3,5,6],scale_diag=[4,2,1,2])
sample21 = dist2.sample(10000,jax.random.PRNGKey(21))

dist7 = tfd.MultivariateNormalDiag(loc=[3,4,5,9],scale_diag=[4,2,1,2])
sample71 = dist7.sample(10000,jax.random.PRNGKey(71))

dist6 = tfd.MultivariateNormalDiag(loc=[16,7,5,23],scale_diag=[4,2,1,2])
sample61 = dist6.sample(10000,jax.random.PRNGKey(61))

mu = [1., 2, 3]
cov = [[ 0.36,  0.12,  0.06],
       [ 0.12,  0.29, -0.13],
       [ 0.06, -0.13,  0.26]]
dist3 = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
sample31 = dist3.sample(10000,jax.random.PRNGKey(31))
sample32 = dist3.sample(10000,jax.random.PRNGKey(32))

dist4 = tfd.Gamma(concentration=[3.0, 4.0], rate=[2.0, 3.0])
sample41 = dist4.sample(10000,jax.random.PRNGKey(41))
sample42 = dist4.sample(10000,jax.random.PRNGKey(42))

dist5 = tfd.Cauchy(loc=[1, 2.], scale=[11, 22.])
sample51 = dist5.sample(10000,jax.random.PRNGKey(51))
sample52 = dist5.sample(10000,jax.random.PRNGKey(52))


_test_params_same_dist = [[sample11,sample12],
                          [sample31,sample32],
                          [sample41,sample42],
                          [sample51,sample52]]

def test_c2st__s1_s2_are_from_the_same_dist():
    """Test if two samples from the same distribution return 0.5 accuracy."""
    for params in _test_params_same_dist: 
        t = c2st(params[0],params[1],0,5)
        print(t)
        assert_allclose(t, 0.5, rtol=0.01, atol=0.01)


_test_params = [[sample11,sample21,sample61],
                [sample11,sample71,sample61],]


def test_c2st__():
    """Test if the accuracy when sample1 is 'close' to sample2 is smaller than
    when sample1 is 'far' from sample2."""
    for params in _test_params:
          t_not_far = c2st(params[0],params[1],0,5)
          t_far = c2st(params[0],params[2],0,5)
          assert_equal(bool(t_not_far < t_far), True)