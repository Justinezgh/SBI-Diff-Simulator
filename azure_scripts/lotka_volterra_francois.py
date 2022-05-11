# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1Fw_H-gvHwGCwMOSIrc9i5SNCH-ueJAYI
"""
import os
import argparse
import pickle
from functools import partial

import optax
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
import sbibm
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
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_simulations", type=int, default=500_000)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--dimension", type=int, default=4)
parser.add_argument("--bijector_layers_size", type=int, default=64)
parser.add_argument("--bijector_layers_shape", type=int, default=2)
parser.add_argument("--nf_layers", type=int, default=4)
parser.add_argument("--n_components", type=int, default=8)
parser.add_argument("--score_weight", type=float, default=0.0)
parser.add_argument("--model_seed", type=int, default=42)
parser.add_argument("--initial_learning_rate", type=float, default=2e-3)
parser.add_argument("--n_steps", type=int, default=20_000)
args = parser.parse_args()

if ON_AZURE:
  run.log('batch_size', args.batch_size)
  run.log('n_simulations', args.n_simulations)
  run.log('n_epochs', args.n_epochs)
  run.log('n_steps', args.n_steps)
  run.log('dimension', args.dimension)
  run.log('bijector_layers_size', args.bijector_layers_size)
  run.log('bijector_layers_shape', args.bijector_layers_shape)
  run.log('nf_layers', args.nf_layers)
  run.log('n_components', args.n_components)
  run.log('score_weight', args.score_weight)
  run.log('model_seed', args.model_seed)
  run.log('initial_learning_rate', args.initial_learning_rate)
else:
  print(args)

rng_seq = hk.PRNGSequence(args.model_seed)

# create truth
task = sbibm.get_task("lotka_volterra")
reference_samples = jnp.array(task.get_reference_posterior_samples(num_observation=1))
observation = jnp.array(task.get_observation(num_observation=1).reshape([2, 10]).T) # Reordering for numpyro
truth = jnp.array(task.get_true_parameters(num_observation=1).flatten())

# create simulations
@jax.jit
def get_batch(key, batch_size=args.batch_size):
    model = condition(lotka_volterra, {'z': jnp.array([30.,1.])})
    (log_probs, samples), scores = get_samples_and_scores(model, key, batch_size=batch_size)
    return samples['theta'], samples['y'].reshape([-1,20]), scores

# create model
bijector_layers = [args.bijector_layers_size] * args.bijector_layers_shape

bijector = partial(
  AffineSigmoidCoupling, 
  layers=bijector_layers, 
  n_components=args.n_components, 
  activation=jnp.sin
)

NF = partial(
  ConditionalRealNVP, 
  n_layers=args.nf_layers, 
  bijector_fn=bijector
)

class SmoothNPE:
    def __call__(self, y):
        net = lotka_volterra_y_bijector(y)
        net = jnp.sin(hk.Linear(256, name='comp1')(net))
        net = jnp.sin(hk.Linear(256, name='comp2')(net))
        net = jnp.sin(hk.Linear(256, name='comp3')(net))
        net = hk.Linear(args.n_components, name='comp4')(net)
        nvp = NF(args.dimension)(net)
        return tfd.TransformedDistribution(nvp,
                                           tfb.Chain([tfb.Invert(lotka_volterra_theta_bijector),
                                                      tfb.Scale(10.),
                                                      tfb.Shift(-0.5)]))

nvp_nd = hk.without_apply_rng(hk.transform(lambda theta,y : SmoothNPE()(y).log_prob(theta).squeeze()))

# init parameters
params_nd = nvp_nd.init(next(rng_seq), jnp.ones([1,4]), jnp.ones([1,20]))

# init optimizer
scheduler = optax.piecewise_constant_schedule(init_value=args.initial_learning_rate, 
                                              boundaries_and_scales={5000: 0.1,
                                                                     10000: 0.5,
                                                                     15000: 0.1})
optimizer = optax.adam(learning_rate=scheduler)
opt_state = optimizer.init(params_nd)

# define loss function and model update
def loss_fn(params, weight, mu, batch, score):
  log_prob, out = jax.vmap(
    jax.value_and_grad(lambda theta, x: nvp_nd.apply(params, theta.reshape([1, 4]), x.reshape([1, -1])).squeeze())
  )(mu, batch)
  return -jnp.mean(log_prob) + weight * jnp.mean(jnp.sum((out - score)**2, axis=1))


@jax.jit
def update(params, opt_state, weight, mu, batch, score):
  """Single SGD update step."""
  loss, grads = jax.value_and_grad(loss_fn)(params, weight, mu, batch, score)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  return loss, new_params, new_opt_state

# train
batch_loss = []
for step in range(args.n_steps):
    mu, batch, score = get_batch(next(rng_seq))
    l, params_nd, opt_state = update(params_nd, opt_state, args.score_weight, mu, batch, score)
    if (step % 100) == 0 :
        print(f'Iter {step} / {args.n_steps}:', np.mean(batch_loss[50:]), scheduler(opt_state[1].count))
    batch_loss.append(l)

with open("./outputs/params_nd.pkl", "wb") as fp:
  pickle.dump(params_nd, fp)

try:
  import arviz as az
  import matplotlib.pyplot as plt
  az.style.use("arviz-darkgrid")

  plt.plot(batch_loss)
  plt.title("Batch loss")
  plt.xlabel("Batches")
  plt.ylabel("Loss")
  plt.savefig('./outputs/loss.png')

  if ON_AZURE:
    run.log_image(name='loss', path='./outputs/loss.png', description='batch loss')
except ImportError:
  pass

# evaluate model
nvp_sample_nd = hk.transform(lambda x: SmoothNPE()(x).sample(10000, seed=hk.next_rng_key()))

sample_nd = nvp_sample_nd.apply(
  params_nd, 
  rng=next(rng_seq),
  x=observation.reshape([-1, 20]) * jnp.ones([10000, 20])
)

jnp.save('./outputs/sample_nd.npy', sample_nd)

# compute metric
c2st_metric = c2st(reference_samples, sample_nd, seed=0, n_folds=5)
if ON_AZURE:
  run.log('c2st_metric', float(c2st_metric))
else:
  print(c2st_metric)

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
          "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
          "contourf_kwargs": {"cmap": "Greens"},
      },
      marginals=True,
      marginal_kwargs={'color': 'g', 'label': 'truth'},

  );
  az.plot_pair(
      data={
        k: sample_nd[:,i] 
        for i,k in enumerate(parameters)
      },
      kind="kde",
      var_names=parameters,
      kde_kwargs={
          "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
          "contourf_kwargs": {"cmap": "Blues", "alpha": 0.7},
      },
      marginals=True,
      marginal_kwargs={'color': 'b', 'label': 'predict'}, 
      reference_values=dict(zip(parameters, np.array(truth))),
      reference_values_kwargs={'markersize': 10, 'color': 'r', 'label': 'truth'}, 
      ax=ax
  );
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



  
    # run.log_image(name='contour_plot', path='./outputs/contour_plot.png', description='contour plot of the predicted posterior vs true posterior')