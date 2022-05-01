import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class Lotka_Volterra:

    def __init__(self,time=10,nb_statistics=10):
        """Lotka Voleterra 
        Parameters
        ----------
        time: Number of measurement times.
        nb_statistics: Number of statistics for the observation vector. It must be divisible by 2.
        """
        self.time = time
        self.nb_statistics = nb_statistics



        def equation(y,t, theta):
            """
            Evaluate the Lotka-Volterra time derivative of the solution `y` at time
            `t` as `func(y, t, *args)` to feed odeint.
            Parameters
            ----------
            y: float
            The two populations.
            t : float
            The time.
            theta: float
            Model parameters.
            Returns
            -------
            Lotka-Volterra equations.
            """
            X = y[0]
            Y = y[1]

            alpha, beta, gamma, delta = (
                theta[..., 0],
                theta[..., 1],
                theta[..., 2],
                theta[..., 3],)

            dX_dt = alpha * X - beta * X * Y
            dY_dt = -gamma * Y + delta * X * Y
            return jnp.stack([dX_dt, dY_dt],axis=0)
        self.equation = equation

        @jax.jit
        def solve_equation(init,theta):
            nb_statistics = self.nb_statistics / 2 
            step = self.time / nb_statistics
            ts = jnp.arange(0.,time, step)
            z = odeint(equation, init, ts, theta, rtol=1e-6, atol=1e-5)
            z = z.T

            return z
        self.solve_equation = solve_equation


    def simulator(self, seed, batch_size=100, which_score=0):
        """
        Generate dataset (theta, observation, score) with stochastic initial 
        conditions sampled from LogNormal(log(10),0.8).
        Parameters
        ----------
        key: PRNGKeyArray
        batch_size: int
            Size of the batch.
        which_score: int 
            0 -> Grad_theta(log p(theta|x,z))
            1 -> Grad_theta(log p(x,z|theta))
        Returns
        -------
            Dataset (theta, observation, score).
            'Theta': the model parameters.
            'Observation': the simulations from p(x|theta). 
            'Score': the joint score.
        """

        key1, key2, key3, key4 = jax.random.split(seed,4)

        prior_parameters = tfd.Independent(tfd.LogNormal(jnp.array([-0.125,-3,-0.125,-3]), 0.5*jnp.ones(4)),1)
        prior_latent_variable = tfd.LogNormal(jnp.log(10*jnp.ones(2)), 0.8*jnp.ones(2))

        theta = prior_parameters.sample(batch_size, key1)
        latent_variable = prior_latent_variable.sample(batch_size, key2)

        def get_log_prob(theta,latent_variable,seed):

          z = self.solve_equation(latent_variable,theta).reshape(1, -1)

          prior = tfd.Independent(tfd.LogNormal(jnp.array([-0.125,-3,-0.125,-3]), 0.5*jnp.ones(4)),1) 
          likelihood = tfd.Independent(tfd.LogNormal(jnp.log(z),0.1),1)

          proportion = likelihood.sample(seed=seed)
          posterior = likelihood.log_prob(jax.lax.stop_gradient(proportion))

          if which_score==0:
            posterior += prior.log_prob(theta)

          return posterior.reshape(), proportion    

        score, x = jax.vmap(jax.grad(lambda p, z, key : get_log_prob(p.reshape(4,), 
                                                                     z.reshape(2,), 
                                                                     key), 
                                     has_aux=True))(theta, 
                                                    latent_variable, 
                                                    jax.random.split(key4,batch_size))
        x = x.reshape(batch_size,-1)

        return theta, x, score



    def get_truth(self, seed):
        """
        Return observed data and the parameters and latent variables that generated this observation.
        """
        thetas, observations, _ = self.simulator(seed,batch_size=1, which_score=0)
        return thetas, observations



    def get_reference_posterior(self, seed, batch_size=10000):
        """
        Return the reference posterior for a given observation.
        """
        _, observation = self.get_truth(seed)

        @jax.jit
        def get_log_prob_mcmc(x,observation):
  
            theta = x[...,:4]
            latent_z = x[...,4:]

            prior_params = tfd.TransformedDistribution(tfd.Independent(tfd.LogNormal(jnp.array([-0.125,-3,-0.125,-3]), 0.5*jnp.ones(4)),1),
                                                        tfb.Invert(tfb.Softplus()))
            prior_latent_variables = tfd.TransformedDistribution(tfd.Independent(tfd.LogNormal(jnp.log(jnp.ones(2)*3), 0.5*jnp.ones(2)),1),
                                                        tfb.Invert(tfb.Softplus()))

            z = self.solve_equation(tfb.Softplus()(latent_z),tfb.Softplus()(theta)).reshape(-1,)
            likelihood = tfd.Independent(tfd.LogNormal(jnp.log(z),0.1),1) 

            posterior = likelihood.log_prob(observation.reshape(-1,))
            posterior += prior_params.log_prob(theta)
            posterior += prior_latent_variables.log_prob(latent_z)

            return posterior
        
        def unnormalized_log_prob(x):
          return jax.vmap(lambda x: get_log_prob_mcmc(x,observation))(x)

        # Initialize the HMC transition kernel.
        adaptive_hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=3,
            step_size=5e-3)
        
        key1, key2 = jax.random.split(seed,2) 

        # To run 100 chains in parallel
        init_state = 0.1*jax.random.normal(key1, [100, 6])

        @jax.jit
        def run_chain():
            samples,is_accepted = tfp.mcmc.sample_chain(
                num_results=int(3*batch_size),
                num_burnin_steps=int(2000),
                current_state=init_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.is_accepted,
                seed=key2)
            return samples,is_accepted
  
        samples_hmc,is_accepted = run_chain() 

        size = len(samples_hmc[is_accepted])
        step = size // batch_size
        sample = samples_hmc[is_accepted][::step]

        return tfb.Softplus()(sample[...,:4])