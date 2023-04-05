# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1Fw_H-gvHwGCwMOSIrc9i5SNCH-ueJAYI
"""
import os
import argparse
import pickle
from functools import partial
import csv

import optax
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
from numpyro.handlers import condition

import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

try:
  from azureml.core import Run
  run = Run.get_context()
  ON_AZURE = True
except ImportError:
  ON_AZURE = False

from sbids.metrics.c2st import c2st
from sbids.tasks import (
  lotka_volterra, lotka_volterra_y_bijector, 
  lotka_volterra_theta_bijector, get_samples_and_scores
)
from sbids.models import AffineSigmoidCoupling, ConditionalRealNVP

os.makedirs("./outputs", exist_ok=True)

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--thetas", type=str)
parser.add_argument("--n_simulations", type=int, default=500_000)
parser.add_argument("--rounds", type=int)
parser.add_argument("--model_seed", type=int, default=42)
args = parser.parse_args()

if ON_AZURE:
  run.log('input_file', args.input_file)
  run.log('thetas', args.thetas)
  run.log('n_simulations', args.n_simulations)
  run.log('rounds', args.rounds)
  run.parent.log('model_seed', args.model_seed)
  


from numpyro.handlers import seed, trace, condition
import jax


# waiting for azure environement to work correctly --' 
def get_samples_and_scores2(model, key, batch_size=64, score_type='density', thetas=None):
    """
    Handling function sampling and computing the score from the model.

    model: a numpyro model
    key: jax random seed
    batch_size: size of the batch to sample
    score_type: 'density' for nabla_theta p(theta | y, z) or 
        'conditional' for nabla_theta p(y | z, theta), default is 'density'.
        
    returns: (log_prob, sample), score
    """
    def log_prob_fn(theta, key):
        cond_model = condition(model, {'theta': theta})
        cond_model = seed(cond_model, key)
        model_trace = trace(cond_model).get_trace()

        theta_prob = model_trace['theta']['fn'].log_prob(model_trace['theta']['value']) 
        z_prob = model_trace['z']['fn'].log_prob(model_trace['z']['value'])
        y_prob = model_trace['y']['fn'].log_prob(jax.lax.stop_gradient(model_trace['y']['value'])) # 

        sample = {'theta': model_trace['theta']['value'],
                  'y': model_trace['y']['value'],
                  'z': model_trace['z']['value']}
        
        if score_type == 'density':
            return theta_prob.sum() + z_prob.sum() + y_prob.sum(), sample
        elif score_type == 'conditional':
            return y_prob.sum() + z_prob.sum(), sample
    
    # Split the key by batch
    keys = jax.random.split(key, batch_size)

    # Sample theta from the model
    if thetas == None: 
      thetas = jax.vmap(lambda k: trace(seed(model, k)).get_trace()['theta']['value'])(keys)

    return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)


## create model compressor 
class Compressor(hk.Module):

  def __call__(self, x):
    
    x = x/1000
    x = x[..., jnp.newaxis]

    net_x = hk.Conv1D(32, 3, 1)(x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Conv1D(64, 4, 2)(net_x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Conv1D(128, 3, 1)(net_x)
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Flatten()(net_x) 
    
    net_x = hk.Linear(32)(net_x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Linear(16)(net_x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Linear(4)(net_x) 

    return net_x.squeeze()

bijector_layers_compressor = [128] * 2

bijector_compressor = partial(
  AffineSigmoidCoupling, 
  layers=bijector_layers_compressor, 
  n_components=16, 
  activation=jax.nn.silu
)

NF_compressor = partial(
  ConditionalRealNVP, 
  n_layers=3, 
  bijector_fn=bijector_compressor)


class Flow_nd_Compressor(hk.Module):
    def __call__(self, y):
        nvp = NF_compressor(4)(y)
        return tfd.TransformedDistribution(nvp,
                                           tfb.Chain([tfb.Invert(lotka_volterra_theta_bijector),
                                                      tfb.Scale(30.),
                                                      tfb.Shift(-0.5)]))

# compressor
compressor = hk.without_apply_rng(hk.transform(lambda x : Compressor()(x)))
# nf
nf = hk.without_apply_rng(hk.transform(lambda p,x : Flow_nd_Compressor()(x).log_prob(p).squeeze()))

a_file = open("params_compressor4.pkl", "rb")
parameters_compressor = pickle.load(a_file)

# reg = compressor.apply(parameters_compressor,batch)
scale_reg = jnp.array([ 6.233213, 13.966835, 13.656771,  7.450791]) #(jnp.std(reg, axis =0)/0.04)
shift_reg = jnp.array([-0.4551523,  -0.48648357, -0.50399256, -0.4354317 ])#jnp.mean(reg/scale_reg, axis = 0)-0.5



## create model 
from functools import partial
bijector_layers = [128] * 2

bijector_npe = partial(
  AffineSigmoidCoupling, 
  layers=bijector_layers, 
  n_components=16, 
  activation=jax.nn.silu
)

NF_npe = partial(
  ConditionalRealNVP, 
  n_layers=4, 
  bijector_fn=bijector_npe)


class SmoothNLE(hk.Module):
    def __call__(self, theta):
        transfo = tfb.Chain([tfb.Invert(lotka_volterra_theta_bijector),tfb.Scale(30.),tfb.Shift(-0.5)])
        net = transfo.inverse(theta)
        nvp = NF_npe(4)(net)
        return tfd.TransformedDistribution(nvp,
                                           tfb.Chain([tfb.Scale(scale_reg), tfb.Shift(shift_reg)]))

nvp_nd = hk.without_apply_rng(hk.transform(lambda theta,y : SmoothNLE()(theta).log_prob(y).squeeze()))


import arviz as az
import matplotlib.pyplot as plt
az.style.use("arviz-darkgrid")

a_file2 = open(args.input_file, "rb") 
params_nd = pickle.load(a_file2)
a_file2.close()

nvp_sample_nd = hk.transform(lambda x : SmoothNLE()(x).sample(10000, seed=hk.next_rng_key()))
# rng_seq = hk.PRNGSequence(19)

import tensorflow as tf
with tf.device('/CPU:0'):
  c2st_save = []
  for seed_for_truth in range(1):
      # create truth
      model = lotka_volterra
      (log_probs, samples), scores = get_samples_and_scores2(model, jax.random.PRNGKey(seed_for_truth), batch_size=1, thetas=None)
      truth, observation = samples['theta'], samples['y'].reshape([-1,20],order='F')
      reference_samples = jnp.load('reference_sample_seed%d.npy' % seed_for_truth)

      obs = compressor.apply(parameters_compressor, observation.reshape([1,20]))

      def unnormalized_log_prob(x):
          likelihood = nvp_nd.apply(params_nd, x.reshape([-1, 4]), obs.reshape([-1, 4])).squeeze()
          prior = tfd.LogNormal(loc=jnp.array([-0.125,-3,-0.125,-3]),
                                scale=jnp.array([0.5, 0.5, 0.5, 0.5])).log_prob(x).sum()
          return likelihood + prior

      
      sample_nd = []
      j = 0

      nb_simulations_needed = 10000
      if args.n_simulations > 10000: 
        nb_simulations_needed = args.n_simulations

      count = 0
      while len(sample_nd) < nb_simulations_needed :
        count+=1
        num_results_size = 5 + j
        # Initialize the HMC transition kernel.
        num_results = int(nb_simulations_needed * 5) # jsp pq il ressort jamais suffisament de sample, ah je suis giga con
        num_burnin_steps = int(4e3)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_prob,
                num_leapfrog_steps=3,
                step_size=0.01),
            num_adaptation_steps=int(num_burnin_steps * 0.8))

        # Run the chain (with burn-in).
        @jax.jit
        def run_chain():
          samples, is_accepted = tfp.mcmc.sample_chain(
              num_results=num_results,
              num_burnin_steps=num_burnin_steps,
              current_state=jnp.array(truth),
              kernel=adaptive_hmc,
              trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
              seed = jax.random.PRNGKey(j))
            
          return samples, is_accepted


        samples_hmc, is_accepted_hmc = run_chain()
        plt.clf()
        plt.plot(samples_hmc[:,0,1])
        plt.savefig('./outputs/nf_%d_chain_round_%d_sim_%d.png' %(args.model_seed, args.rounds, args.n_simulations))
          # run.log_image(name='chain', path='./outputs/nf_%d_chain_round_%d_sim_%d.png' %(args.model_seed, args.rounds, args.n_simulations), description='chain mcmc')
          
        sample_nd = samples_hmc[is_accepted_hmc].reshape([-1,4])
        # sample_nd = sample_nd[::2,...]
        run.log('sample size mcmc', len(sample_nd))

        if count > 5:
          run.cancel()

        j += 5

        # j += 6
      
      # to be sure we have the correct amount of thetas
      # inds = np.random.randint(0, len(sample_nd),  args.n_simulations) 
      # sample_nd = sample_nd[inds,...]

      print(sample_nd)

      


      # compute metric c2st
      if len(sample_nd) > 10000 : 
        inds = np.random.randint(0, len(sample_nd),  10000) 
        sample_nd_for_c2st = sample_nd[inds,...]
      
      if len(sample_nd) > args.n_simulations : 
          inds = np.random.randint(0, len(sample_nd),  args.n_simulations) 
          sample_nd = sample_nd[inds,...]

      c2st_metric = c2st(reference_samples, sample_nd_for_c2st, seed=0, n_folds=5)

      if ON_AZURE:
          run.log('c2st_metric', float(c2st_metric))
      else:
          print(c2st_metric)

      c2st_save.append(c2st_metric)

  c2st_mean = jnp.mean(jnp.array(c2st_save))
  if ON_AZURE:
      run.parent.log('c2st_metric_mean', float(c2st_mean))
  else:
      print(c2st_mean)

  jnp.save(args.thetas, sample_nd)
  jnp.save('./outputs/c2st_save.npy', c2st_save)




# plot results
if ON_AZURE:
  import arviz as az
  az.style.use("arviz-darkgrid")

  parameters = ["alpha", "beta", "gamma", "delta"]

  plt.figure(figsize=(10, 10))
  ax = az.plot_pair(
      data={
        k: reference_samples[:,i]
        for i,k in enumerate(parameters)
      },
      kind="kde",
      var_names=parameters,
      kde_kwargs={
          "hdi_probs": [0.9],  # Plot 30%, 60% and 90% HDI contours
          "contourf_kwargs": {"cmap": "Greens"},
      },
      marginals=True,
      marginal_kwargs={'color': 'g', 'label': 'truth'},

  )
  az.plot_pair(
      data={
        k: sample_nd[:,i] 
        for i,k in enumerate(parameters)
      },
      kind="kde",
      var_names=parameters,
      kde_kwargs={
          "hdi_probs": [0.9],  # Plot 30%, 60% and 90% HDI contours
          "contourf_kwargs": {"cmap": "Blues", "alpha": 0.7},
      },
      marginals=True,
      marginal_kwargs={'color': 'b', 'label': 'predict'}, 
      reference_values=dict(zip(parameters, np.array(truth))),
      reference_values_kwargs={'markersize': 10, 'color': 'r', 'label': 'truth'}, 
      ax=ax
  )
  plt.savefig('./outputs/nf_%d_contour_plot_%d_sim_%d.png' %(args.model_seed, args.rounds, args.n_simulations))
  # run.log_image(name='contour_plot', path='./outputs/nf_%d_contour_plot_%d_sim_%d.png' %(args.model_seed, args.rounds, args.n_simulations), description='contour plot of the predicted posterior vs true posterior')
else:
  try:
    from chainconsumer import ChainConsumer

    parameters = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']
    c = ChainConsumer()
    c.add_chain(sample_nd, parameters=parameters, name="prediction")
    c.add_chain(reference_samples, parameters=parameters, name="truth")
    c.plotter.plot(
      filename="./outputs/contour_plot.png", 
      figsize=[10, 10], 
      truth=np.array(truth), 
      extents=[[t - 5 * np.std(reference_samples[:,i]), 
                t + 5 * np.std(reference_samples[:,i])] for i,t in enumerate(truth)]
    )
  except ImportError:
    pass
