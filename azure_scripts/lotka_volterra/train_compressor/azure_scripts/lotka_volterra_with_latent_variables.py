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
parser.add_argument("--normalization_compressor", type=float, default=15)
parser.add_argument("--normalization_reg", type=float, default=0.04)
parser.add_argument("--comp_nf_layers", type=int, default=3)
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
  run.log('normalization_compressor', args.normalization_compressor)
  run.log('normalization_reg', args.normalization_reg)
  run.log('comp_nf_layers', args.comp_nf_layers)
else:
  print(args)


# create simulations
@jax.jit
def get_batch(key, batch_size=1e5):
    #model = lotka_volterra 
    model = condition(lotka_volterra, {'z': jnp.array([10.,1.])})
    (log_probs, samples), scores = get_samples_and_scores(model, key, batch_size=batch_size)
    return samples['theta'], samples['y'].reshape([-1,20], order='F'), scores


# create model compressor 
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
  n_layers=args.comp_nf_layers, 
  bijector_fn=bijector_compressor)


class Flow_nd_Compressor(hk.Module):
    def __call__(self, y):
        nvp = NF_compressor(4)(y)
        return tfd.TransformedDistribution(nvp,
                                           tfb.Chain([tfb.Invert(lotka_volterra_theta_bijector),
                                                      tfb.Scale(args.normalization_compressor),
                                                      tfb.Shift(-0.5)]))

# compressor
compressor = hk.without_apply_rng(hk.transform(lambda x : Compressor()(x)))
# nf
nf = hk.without_apply_rng(hk.transform(lambda p,x : Flow_nd_Compressor()(x).log_prob(p).squeeze()))

# a_file = open("params_compressor2.pkl", "rb")
# parameters_compressor = pickle.load(a_file)

# compressor
compressor = hk.without_apply_rng(hk.transform(lambda x : Compressor()(x)))
rng_seq = hk.PRNGSequence(12)
params_c = compressor.init(next(rng_seq), 0.5*jnp.ones([1,20]))
# nf
nf = hk.without_apply_rng(hk.transform(lambda p,x : Flow_nd_Compressor()(x).log_prob(p).squeeze()))
rng_seq = hk.PRNGSequence(2)
params_nf = nf.init(next(rng_seq),  0.5*jnp.ones([1,4]), 0.5*jnp.ones([1,4]))

parameters_compressor = hk.data_structures.merge(params_c, params_nf)

learning_rate=0.001
optimizer_c = optax.adam(learning_rate)
opt_state_c = optimizer_c.init(parameters_compressor)

def loss_compressor(params, mu, batch):
  y = compressor.apply(params, batch)
  log_prob = jax.vmap(lambda theta, x: nf.apply(params, theta.reshape([1,4]), x.reshape([1,4])).squeeze())(mu, y)
  return -jnp.mean(log_prob) 

@jax.jit
def update_compressor(params, opt_state, mu, batch):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_compressor)(params, mu, batch)
    updates, new_opt_state = optimizer_c.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
  
    return loss, new_params, new_opt_state

from tqdm import tqdm
losses_c = []
master_seed = hk.PRNGSequence(0)

it = 2000

p, x, _ = get_batch(jax.random.PRNGKey(12))

if jnp.isnan(x).any() == True:
  idx = jnp.where(jnp.isnan(x))[0]
  x = jnp.delete(x, idx, axis=0)
  p = jnp.delete(p, idx, axis=0)

for step in tqdm(range(it)):
  inds = np.random.randint(0, 100000, 256) 
  l, parameters_compressor, opt_state_c = update_compressor(parameters_compressor, opt_state_c, p[inds], x[inds])
  losses_c.append(l)

learning_rate=0.0001
optimizer_c = optax.adam(learning_rate)
opt_state_c = optimizer_c.init(parameters_compressor)

it = 78000
for step in tqdm(range(it)):
  inds = np.random.randint(0, 100000, 256) 
  l, parameters_compressor, opt_state_c = update_compressor(parameters_compressor, opt_state_c,  p[inds], x[inds])
  losses_c.append(l)


import arviz as az
import matplotlib.pyplot as plt
az.style.use("arviz-darkgrid")
plt.clf()
plt.plot(losses_c)
plt.plot(-14*jnp.ones(len(losses_c)))
plt.savefig('./outputs/loss_compressor.png')
if ON_AZURE:
    run.log_image(name='loss compressor', path='./outputs/loss_compressor.png', description=' loss compressor')


reg = compressor.apply(parameters_compressor,x)
scale_reg = (jnp.std(reg, axis =0)/args.normalization_reg)
shift_reg = jnp.mean(reg/scale_reg, axis = 0)-0.5

data = reg/scale_reg - shift_reg
plt.clf()
plt.boxplot([data[...,0],data[...,1],data[...,2],data[...,3]])
plt.savefig('./outputs/normalization.png')
if ON_AZURE:
    run.log_image(name='normalization', path='./outputs/normalization.png', description='normalization')



with open("./outputs/parameters_compressor.pkl", "wb") as fp:
  pickle.dump(parameters_compressor, fp)


