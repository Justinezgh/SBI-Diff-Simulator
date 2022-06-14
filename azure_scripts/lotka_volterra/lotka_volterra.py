# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1Fw_H-gvHwGCwMOSIrc9i5SNCH-ueJAYI
"""
import os
import argparse
import pickle
from functools import partial

from tqdm import tqdm

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
from sbids.tasks import lotka_volterra, get_samples_and_scores
from sbids.models import AffineSigmoidCoupling, ConditionalRealNVP

os.makedirs("./outputs", exist_ok=True)

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--n_simulations", type=int, default=5e5)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--dimension", type=int, default=4)
parser.add_argument("--bijector_layers_size", type=int, default=256)
parser.add_argument("--bijector_layers_shape", type=int, default=3)
parser.add_argument("--nf_layers", type=int, default=3)
parser.add_argument("--n_components", type=int, default=32)
parser.add_argument("--score_weight", type=float, default=0.0)
parser.add_argument("--model_seed", type=int, default=0)
args = parser.parse_args()

if ON_AZURE:
  run.log('batch_size', args.batch_size)
  run.log('n_simulations', args.n_simulations)
  run.log('n_epochs', args.n_epochs)
  run.log('dimension', args.dimension)
  run.log('bijector_layers_size', args.bijector_layers_size)
  run.log('bijector_layers_shape', args.bijector_layers_shape)
  run.log('nf_layers', args.nf_layers)
  run.log('n_components', args.n_components)
  run.log('score_weight', args.score_weight)
  run.log('model_seed', args.model_seed)
else:
  print(args)

# create truth
key = jax.random.PRNGKey(0)
lokta_volterra_10 = partial(lotka_volterra, ts=jnp.linspace(0, 18.9, 5))
lv_cond = condition(lokta_volterra_10, {'z': jnp.array([30.0, 1.0])})
(_, samples_0), _ = get_samples_and_scores(lv_cond, key, batch_size=1)
observation = np.reshape(samples_0['y'], (-1, 10), order='F')
truth_0 = samples_0['theta']

# create data train
key = jax.random.PRNGKey(10)
(_, samples), score = get_samples_and_scores(lv_cond, key, batch_size=args.n_simulations)
batch = np.reshape(samples['y'], (-1, 10), order='F')
mu = samples['theta']

# normalize data
scale_theta = jnp.std(mu, axis=0) / 0.02
shift_theta = jnp.mean(mu/scale_theta, axis=0) - 0.4
transformation_params = tfb.Chain(
    [tfb.Scale(scale_theta), tfb.Shift(shift_theta)])

scale_reg = jnp.std(batch, axis=0) / 0.01
shift_reg = jnp.mean(batch/scale_reg, axis=0) - 0.4
transformation_x = tfb.Chain(
  [tfb.Scale(scale_reg), tfb.Shift(shift_reg)])

normalized_p = transformation_params.inverse(mu)
normalized_reg = transformation_x.inverse(batch)

# create model
bijector_layers = [args.bijector_layers_size] * args.bijector_layers_shape

bijector = partial(AffineSigmoidCoupling, layers=bijector_layers, n_components=args.n_components, activation=jax.nn.silu)
NF = partial(ConditionalRealNVP, n_layers=args.nf_layers, bijector_fn=bijector)

nvp_nd = hk.without_apply_rng(hk.transform(lambda p, x: NF(args.dimension)(x).log_prob(p).squeeze()))
nvp_sample_nd = hk.transform(lambda x: NF(args.dimension)(x).sample(10000, seed=hk.next_rng_key()))

# init parameters
rng_seq = hk.PRNGSequence(args.model_seed)
params_nd = nvp_nd.init(next(rng_seq), 0.4 * jnp.ones([1, 4]), 0.4 * jnp.ones([1, 10]))

# init optimizer
scheduler = optax.exponential_decay(
    init_value=0.001, transition_steps=2000, decay_rate=0.9, end_value=0.00001)
optimizer = optax.chain(
  optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1))
opt_state = optimizer.init(params_nd)

# define loss function and model update
def loss_fn(params, weight, mu, batch, score):
  log_prob, out = jax.vmap(
    jax.value_and_grad(
      lambda theta, x: nvp_nd.apply(params, theta.reshape([1, 4]), x.reshape([1, 10])).squeeze()
    )
  )(mu, batch)
  return -jnp.mean(log_prob) + weight * jnp.mean(jnp.sum((out - score)**2, axis=1))


@jax.jit
def update(params, opt_state, weight, mu, batch, score):
  """Single SGD update step."""
  loss, grads = jax.value_and_grad(loss_fn)(params, weight, mu, batch, score)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return loss, new_params, new_opt_state

# create data stream
batch_size = args.batch_size
n_train = len(batch)
n_batches = n_train // batch_size

def data_stream():
  """
  Creates a data stream with a predefined batch size.
  """
  rng = np.random.RandomState(0)
  while True:
    perm = rng.permutation(n_train)
    for i in range(n_batches):
      batch_idx = perm[i * batch_size: (i + 1)*batch_size]
      yield normalized_reg[batch_idx], normalized_p[batch_idx], score[batch_idx]

# train
batch_loss = []
batch_generator = data_stream()

for epochs in tqdm(range(args.n_epochs)):
  for _ in range(n_batches):
    b,m,s = next(batch_generator)    
    l, params_nd, opt_state = update(params_nd, opt_state, args.score_weight, m,  b, s)
    batch_loss.append(l)

    if jnp.isnan(l)==True:
      print('NAN')
      break
    params_nd_t = params_nd

  if jnp.isnan(l)==True:
    print('NAN')
    break

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

with open("./outputs/params_nd.pkl", "wb") as fp:
  pickle.dump(params_nd, fp)

sample_nd = nvp_sample_nd.apply(
  params_nd, 
  rng=next(rng_seq),
  x=transformation_x.inverse(observation)*jnp.ones([10000, 10])
)

predicted_samples = transformation_params.forward(sample_nd)
jnp.save('./outputs/predicted_samples.npy', predicted_samples)

true_posterior_samples = jnp.load('posterior_z_fixedkey0-4.npy')

# compute metric
c2st_metric = c2st(true_posterior_samples, predicted_samples, seed=0, n_folds=5)
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
        k: true_posterior_samples[:,i]
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

  )
  az.plot_pair(
      data={
        k: predicted_samples[:,i] 
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
      reference_values=dict(zip(parameters, np.array(truth_0))),
      reference_values_kwargs={'markersize': 10, 'color': 'r', 'label': 'truth'}, 
      ax = ax
  )
  plt.savefig("./outputs/contour_plot.png")
  run.log_image(name='contour_plot', path='./outputs/contour_plot.png', description='contour plot of the predicted posterior vs true posterior')
else:
  try:
    from chainconsumer import ChainConsumer

    parameters = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']
    c = ChainConsumer()
    c.add_chain(predicted_samples, parameters=parameters, name="prediction")
    c.add_chain(true_posterior_samples, parameters=parameters, name="truth")
    c.plotter.plot(
      filename="./outputs/contour_plot.png", 
      figsize=[10, 10], 
      truth=np.array(truth_0), 
      extents={
        r'$\alpha$': (0.3, 1),
        r'$\beta$':(0, 0.08),
        r'$\gamma$':(0.8, 2.7),
        r'$\delta$':(0, 0.05),
      }
    )
  except ImportError:
    pass
