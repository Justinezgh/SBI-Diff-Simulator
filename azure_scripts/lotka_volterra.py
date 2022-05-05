# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1Fw_H-gvHwGCwMOSIrc9i5SNCH-ueJAYI
"""
import argparse
import pickle
from functools import partial

import optax
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
from numpyro.handlers import condition

import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

from tqdm import tqdm

#!pip install git+https://github.com/Justinezgh/SBI-Diff-Simulator.git
from sbids.metrics.c2st import c2st
from sbids.tasks import lotka_volterra, get_samples_and_scores
from sbids.bijectors.bijectors import MixtureAffineSigmoidBijector
from sbids.models import AffineSigmoidCoupling, ConditionalRealNVP


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--n_simulations", type=int, default=5e5)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--dimension", type=int, default=4)
parser.add_argument("--layers", type=list, default=[128, 128])
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--n_components", type=int, default=32)
parser.add_argument("--score_weight", type=float, default=0.0)
args = parser.parse_args()

# truth
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

# filter the samples
if (batch > 500).any() == True:
  idx = jnp.where(batch > 500)[0]
  batch = jnp.delete(batch, idx, axis=0)
  mu = jnp.delete(mu, idx, axis=0)
  score = jnp.delete(score, idx, axis=0)

if jnp.isnan(batch).any() == True:
  idx = jnp.where(jnp.isnan(batch))[0]
  batch = jnp.delete(batch, idx, axis=0)
  mu = jnp.delete(mu, idx, axis=0)
  score = jnp.delete(score, idx, axis=0)

# normalize data
scale_theta = jnp.std(mu, axis=0) / 0.02
shift_theta = jnp.mean(mu/scale_theta, axis=0) - 0.4

scale_reg = jnp.std(batch, axis=0) / 0.01
shift_reg = jnp.mean(batch/scale_reg, axis=0) - 0.4

transformation_params = tfb.Chain(
    [tfb.Scale(scale_theta), tfb.Shift(shift_theta)])
transformation_x = tfb.Chain(
  [tfb.Scale(scale_reg), tfb.Shift(shift_reg)])

normalized_p = transformation_params.inverse(mu)
normalized_reg = transformation_x.inverse(batch)

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

bijector = partial(AffineSigmoidCoupling, layers=args.layers, n_components=args.n_components, activation=jax.nn.silu)
NF = partial(ConditionalRealNVP, n_layers=args.n_layers, bijector_fn=bijector)

nvp_nd = hk.without_apply_rng(hk.transform(lambda p, x: NF(args.dimension)(x).log_prob(p).squeeze()))
nvp_sample_nd = hk.transform(lambda x: NF(args.dimension)(x).sample(10000, seed=hk.next_rng_key()))


rng_seq = hk.PRNGSequence(5)
params_nd = nvp_nd.init(
  next(rng_seq), 0.4 * jnp.ones([1, 4]), 0.4 * jnp.ones([1, 10])
)

scheduler = optax.exponential_decay(
    init_value=0.001, transition_steps=2000, decay_rate=0.9, end_value=0.00001)
optimizer = optax.chain(
  optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1))
opt_state = optimizer.init(params_nd)


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

# METRICS

c2st_metric = c2st(true_posterior_samples, predicted_samples, seed=0, n_folds=5)
print(c2st_metric)


# PLOTS

DO_PLOTS = True
try:
  import arviz as az
  import matplotlib.pyplot as plt
  from chainconsumer import ChainConsumer
except ImportError:
  DO_PLOTS = False

if DO_PLOTS:
  az.style.use("arviz-darkgrid")

  plt.plot(batch_loss)
  plt.title("Batch loss")
  plt.xlabel("Batches")
  plt.ylabel("Loss")
  plt.savefig('./outputs/loss.png')
  
  parameters = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']

  c = ChainConsumer()
  c.add_chain(predicted_samples, parameters=parameters, name="prediction")
  c.add_chain(true_posterior_samples, parameters=parameters, name="truth")
  c.plotter.plot(filename="./outputs/contour_plot.png", figsize=[10,10], truth=[0.603503  , 0.03026864, 1.6093055 , 0.01722082])

# TODO: a tester
# fig = c.plotter.plot(figsize=[10,10], truth=truth_0)
