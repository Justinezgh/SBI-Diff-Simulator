import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import cond


__all__=["galton_board"]

def sigmoid_(x):
    return 1. / (1. + jnp.exp(-x))

def nail_positions(theta, n_rows, n_nails, level, nail):
    """
    Compute the probability p(zh, zv, θ) of going left. 
    
    Args:
        theta: parameter used for the probability of bouncing to the left
        n_rows: number of rows in the galton board
        n_nails: number of nails in the galton board
        level: current row
        nail: current nail
    """

    level_rel = 1. * level / (n_rows - 1) #zv
    nail_rel = 2. * nail / (n_nails - 1) - 1. #zh

    nail_positions = ((1. - jnp.sin(jnp.pi * level_rel)) * 0.5
                      + jnp.sin(jnp.pi * level_rel) * sigmoid_(10 * theta * nail_rel))
    
    res = cond(level % 2 == 1 and nail == 0, 
                       lambda _: 0.0, 
                       lambda _: cond(level % 2 == 1 and nail == n_nails, 
                                                 lambda _: 1.0, 
                                                 lambda _: nail_positions, 
                                                 0), 
                       0)

    return res

def galton_board(y = None,  n_rows = 20, n_nails = 31):
  """
  Probabilistic model for the Generalized Galton Board example as described
  in https://github.com/johannbrehmer/simulator-mining-example.

  Args:
    y: bin in which the ball ends up 
    n_rows: number of rows in the galton board
    n_nails: number of nails in the galton board
  """

  theta = numpyro.sample('theta', dist.Uniform(-1, 0))
  pos = n_nails // 2

  for level in range(n_rows):
    pos = numpyro.sample('z%d' %level,  
                      dist.TransformedDistribution(dist.Bernoulli(1 - nail_positions(theta, n_rows, n_nails, level, pos)), 
                                              dist.transforms.AffineTransform( - (level % 2) + pos, 1)))
  y = numpyro.sample("y", 
                      dist.TransformedDistribution(dist.Bernoulli(1 - nail_positions(theta, n_rows, n_nails, level, pos)), 
                                              dist.transforms.AffineTransform( - (level % 2) + pos, 1)), 
                      obs=y)

  return y