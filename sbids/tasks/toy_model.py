import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

__all__ = ["toy_model"]


def sigmoid_(x):
    return 1. / (1. + jnp.exp(-x))


def toy_model(nb_latent_variables):
    """Toy simulator to study the impact of gradients stochasticty.

    Parameters
    ----------
    nb_latent_variables : int
        Number of latent variables

    Returns
    -------
        Numpyro model
    """

    theta = numpyro.sample('theta', dist.Normal(0., 0.1))
    z = 0.

    for i in range(nb_latent_variables - 10):
        z = numpyro.sample('z%d' % i, dist.Normal(z, 0.1))

    for i in range(nb_latent_variables - 10, nb_latent_variables):
        z = numpyro.sample('z%d' % i, dist.Normal(z, sigmoid_(theta * z)))

    y = numpyro.sample(
       'y',
       dist.Normal((z / jnp.log(nb_latent_variables)) + 5 * theta, 0.5)
    )

    return y
