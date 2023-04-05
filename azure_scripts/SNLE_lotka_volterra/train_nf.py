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



# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_simulations", type=int, default=500_000)
parser.add_argument("--score_weight", type=float, default=0.0)
parser.add_argument("--model_seed", type=int, default=42)
parser.add_argument("--n_steps", type=int, default=20_000)
parser.add_argument("--output_file")
parser.add_argument("--input_file")
parser.add_argument("--sample_in", type=str)
parser.add_argument("--sample_out", type=str)
parser.add_argument("--rounds", type=int)
parser.add_argument("--new_thetas")
parser.add_argument("--start_grad",type=int)
parser.add_argument("--seq",type=int)
args = parser.parse_args()



if ON_AZURE:
  run.parent.log('batch_size', args.batch_size)
  run.parent.log('n_steps', args.n_steps)
  run.parent.log('score_weight', args.score_weight)
  run.parent.log('model_seed', args.model_seed)
  run.parent.log('output_file', args.output_file)
  run.parent.log('input_file', args.input_file)
  run.parent.log('sample_in', args.sample_in)
  run.parent.log('sample_out', args.sample_out)
  run.parent.log('rounds', args.rounds)
  run.parent.log('new_thetas', args.new_thetas)
  run.parent.log('start_grad', args.start_grad)
  run.parent.log('seq', args.seq)
  if args.rounds==0:
    run.parent.log('n_simulations', args.n_simulations)
  else :
    run.log('n_simulations', args.n_simulations)
else:
  print(args)

os.makedirs("./outputs", exist_ok=True)


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


## create simulations
if args.rounds == 0: 
  sample = {}
  thetas = None

elif args.seq == 0 : 
  thetas = None
  file_sample = open(args.sample_in, "rb") 
  sample = pickle.load(file_sample)
  file_sample.close()

else: 
  thetas = jnp.load(args.new_thetas + ".npy")
  file_sample = open(args.sample_in, "rb") 
  sample = pickle.load(file_sample)
  file_sample.close()
  print(sample)

@jax.jit
def get_batch(key, batch_size=args.n_simulations):
    model = lotka_volterra
    (log_probs, samples), scores = get_samples_and_scores2(model, key, batch_size=batch_size, score_type='conditional', thetas = thetas)
    return samples['theta'], samples['y'].reshape([-1,20], order='F'), scores

thetas, batch, score = get_batch(jax.random.PRNGKey(args.rounds)) 

# just in case..
if jnp.isnan(score).any() == True:
  idx = jnp.where(jnp.isnan(score))[0]
  batch = jnp.delete(batch, idx, axis=0)
  thetas = jnp.delete(thetas, idx, axis=0)
  score = jnp.delete(score, idx, axis=0)

if args.rounds == 0: 
  sample['thetas'] = thetas
  sample['batch'] = batch
  sample['score'] = score
  sample['round'] = jnp.repeat(args.rounds, len(score))
else :
  sample['thetas'] = np.concatenate((sample['thetas'], thetas))
  sample['batch'] = np.concatenate((sample['batch'], batch))
  sample['score'] = np.concatenate((sample['score'], score))
  sample['round'] = np.concatenate((sample['round'], jnp.repeat(args.rounds, len(score))))

print(sample)

with open(args.sample_out, "wb") as fp:
  pickle.dump(sample, fp)


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


# init parameters
rng_seq = hk.PRNGSequence(args.model_seed)
if args.rounds == 0:
  params_nd = nvp_nd.init(next(rng_seq),  0.5*jnp.ones([1,4]), 0.5*jnp.ones([1,4]))

else:
  file_in = open(args.input_file, "rb") 
  params_nd = pickle.load(file_in)
  file_in.close()

# init optimizer
scheduler = optax.exponential_decay(init_value=0.001, transition_steps=1000, decay_rate=0.9, end_value=0.00001)
optimizer = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1))
opt_state = optimizer.init(params_nd)

# define loss function and model update
def loss_fn(params, weight, mu, batch, score):
  y = compressor.apply(parameters_compressor, batch)
  log_prob, out = jax.vmap(
    jax.value_and_grad(lambda theta, x: nvp_nd.apply(params, theta.reshape([1,4]), x.reshape([1,4])).squeeze())
    )(mu, y)
  # print('out shape: ', out.shape)
  # print('score shape: ', score.shape)
  return -jnp.mean(log_prob) +  jnp.mean(weight * jnp.sum((out - score)**2, axis=1)) 

@jax.jit
def update(params, opt_state, weight, mu, batch, score):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, weight, mu, batch, score)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
  
    return loss, new_params, new_opt_state

# train
batch_loss = []
for step in range(args.n_steps):
    # inds = np.random.randint(0, args.n_simulations * (args.rounds + 1) - 1, args.batch_size) 
    inds = np.random.randint(0, len(sample['thetas']), args.batch_size) 
    
    inds_not_grad = jnp.where(sample['round'][inds] < args.start_grad)
    weight = np.repeat(args.score_weight, len(inds))
    weight[inds_not_grad] = 0

    l, params_nd, opt_state = update(params_nd, opt_state, weight, sample['thetas'][inds], sample['batch'][inds], sample['score'][inds])
    if jnp.isnan(l):
      if ON_AZURE:
        run.cancel()
      else:
        break
    if (step % 100) == 0 and step > 0:
        print(f'Iter {step:5d} ({step/args.n_steps:2.1%}) | average loss = {np.mean(batch_loss[-50:]):2.3f} | learning rate = {scheduler(opt_state[1].count):.5f}')
    batch_loss.append(l)

with open("./outputs/params_nd.pkl", "wb") as fp:
   pickle.dump(params_nd, fp)

with open(args.output_file, "wb") as fp:
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

