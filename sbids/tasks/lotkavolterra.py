import jax.numpy as jnp
from jax.experimental.ode import odeint
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import numpyro
import numpyro.distributions as dist

__all__=["lotka_volterra", "lotka_volterra_y_bijector", "lotka_volterra_theta_bijector"]

def _dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """
    u = z[0]
    v = z[1]
    
    alpha, beta, gamma, delta = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
    )
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def lotka_volterra(y=None, ts=jnp.linspace(0,18.9,10)):
    """
    Probabilistic model for the Lotka–Volterra system.
    :param int N: number of measurement times
    :param numpy.ndarray y: measured populations with shape (N, 2)
    """
    # initial population
    z = numpyro.sample("z", dist.LogNormal(jnp.log(3), 0.5).expand([2]))
    
    # parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.LogNormal(
            loc=jnp.array([-0.125,-3,-0.125,-3]),
            scale=jnp.array([0.5, 0.5, 0.5, 0.5]),
        ),
    )
    
    # integrate dz/dt, the result will have shape N x 2
    x = odeint(_dz_dt, z, ts, theta, rtol=1e-6, atol=1e-5, mxstep=1000)
    x = jnp.clip(x, 1e-10, 1e4) # clip to avoid nan

    # measured populations
    return numpyro.sample("y", dist.LogNormal(jnp.log(x), jnp.ones_like(x)*0.1), obs=y)

# Defines some useful bijectors that normalize the output of the model to approximately Gaussian and unconstrained.
lotka_volterra_y_bijector = tfb.Chain([ 
                tfb.Scale(0.38), 
                tfb.Invert(tfb.Softplus()),
                tfb.Scale(0.021)
                ])
lotka_volterra_theta_bijector = tfb.Chain([
                tfb.Scale(jnp.array([2.,2.,2.,2.])),
                tfb.Shift(jnp.array([0.125,3,0.125,3])),
                tfb.Log()
                ])