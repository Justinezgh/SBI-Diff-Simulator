import pytest
import sbibm
from sbids.tasks import lotkavolterra
import numpyro

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import numpy as np
from numpy.testing import assert_allclose

def test_lotkavolterra():
    task = sbibm.get_task("lotka_volterra")
    # Reference sample from the forward model SBIBM
    d0 = jnp.array(task.get_observation(num_observation=1).reshape([2, 10]).T)

    model = numpyro.handlers.condition(lotkavolterra, 
                    {'z':jnp.array([30.0, 1.0]), 
                     'theta':jnp.array([0.6859, 0.1076, 0.8879, 0.1168])})

    model = numpyro.handlers.seed(model, PRNGKey(42))

    # Sample from the model
    d = model()

    # Check against SBIBM
    assert_allclose(d, d0, atol=4.5)

    # Check against precomputed value
    d1 = np.array([[29.567883  ,  1.1902004 ],
       [ 1.0772254 , 31.346981  ],
       [ 0.26328862,  4.0119452 ],
       [ 0.70920855,  0.78482395],
       [ 3.031377  ,  0.17608653],
       [11.053237  ,  0.13315858],
       [38.401436  ,  8.250235  ],
       [ 0.41325566, 15.833204  ],
       [ 0.36986515,  2.6465595 ],
       [ 1.374722  ,  0.47477472]])
    assert_allclose(d, d1)



