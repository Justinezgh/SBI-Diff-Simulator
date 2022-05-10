# -*- coding: utf-8 -*-
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
from sbids.tasks.two_moons import get_two_moons
from sbids.models import AffineSigmoidCoupling, ConditionalRealNVP

os.makedirs("./outputs", exist_ok=True)

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
#parser.add_argument("--n_simulations", type=int, default=5e5)
parser.add_argument("--n_updates", type=int, default=200)
parser.add_argument("--bijector_layers_size", type=int, default=256)
parser.add_argument("--bijector_layers_shape", type=int, default=3)
parser.add_argument("--nf_layers", type=int, default=4)
parser.add_argument("--n_components", type=int, default=16)
parser.add_argument("--score_weight", type=float, default=0.0)
parser.add_argument("--model_seed", type=int, default=0)
args = parser.parse_args()

if ON_AZURE:
  run.log('batch_size', args.batch_size)
  #run.log('n_simulations', args.n_simulations)
  run.log('n_updates', args.n_updates)
  run.log('bijector_layers_size', args.bijector_layers_size)
  run.log('bijector_layers_shape', args.bijector_layers_shape)
  run.log('nf_layers', args.nf_layers)
  run.log('n_components', args.n_components)
  run.log('score_weight', args.score_weight)
  run.log('model_seed', args.model_seed)
else:
  print(args)

  
# create data train
seed = jax.random.PRNGKey(0)
two_moons = get_two_moons(sigma = 0.01, normalized=True)
batch = two_moons.sample(5000, seed=seed)
score = jax.vmap(jax.grad(two_moons.log_prob))(batch)

# normalize data
transformation_params = tfb.Chain([tfb.Scale([0.8,1.5]),tfb.Shift([0.07,-0.16])])

# create model
bijector_layers = [args.bijector_layers_size] * args.bijector_layers_shape

bijector_smooth = partial(AffineSigmoidCoupling, 
                   layers = bijector_layers,
                   n_components = args.n_components, 
                   activation = jnp.sin)

NF_smooth = partial(ConditionalRealNVP, n_layers = args.nf_layers, bijector_fn = bijector_smooth)

model_smooth = hk.without_apply_rng(hk.transform(
    lambda p:tfd.TransformedDistribution(NF_smooth(2)(jnp.ones([1,1])), 
                                           tfb.Invert(transformation_params)).log_prob(p)))


# define loss function and model update
def loss_fn(params, weight, batch, score):
  log_prob, out = jax.vmap(
      jax.value_and_grad(
          lambda x, param: model_smooth.apply(param, 
          x.reshape([1,2])).squeeze()), 
          [0, None])(batch, params) 
  return -jnp.mean(log_prob) + weight * jnp.mean(jnp.sum((out - score)**2, axis=1))


@jax.jit
def update(params, opt_state, weight, batch, score):
  """Single SGD update step."""
  loss, grads = jax.value_and_grad(loss_fn)(params, weight, batch, score)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return loss, new_params, new_opt_state

# train

c2st_save = []
nlp_save = []
for n_simulations in [20,50,100,200,500,1000]:

  # init parameters
  rng_seq = hk.PRNGSequence(args.model_seed)
  params_smooth = model_smooth.init(next(rng_seq), p=jnp.zeros([1,2]))

  # init optimizer
  scheduler = optax.exponential_decay(
      init_value=0.001, transition_steps=2000, decay_rate=0.9, end_value=0.00001)
  optimizer = optax.chain(
    optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1))
  opt_state = optimizer.init(params_smooth)

  batch_loss = []

  for step in tqdm(range(args.n_updates)): 
      
      inds = np.random.randint(0, n_simulations, args.batch_size) 
      l, params_smooth, opt_state = update(params_smooth, opt_state, args.score_weight, batch[inds], score[inds])
      batch_loss.append(l)

      if jnp.isnan(l)==True:
        print('NAN')
        break

  # plot loss      
  try:
    import arviz as az
    import matplotlib.pyplot as plt
    az.style.use("arviz-darkgrid")

    plt.clf()
    plt.plot(batch_loss)
    plt.title("Batch loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.savefig('./outputs/loss%d.png' % (n_simulations))

    if ON_AZURE:
      run.log_image(name='loss', path='./outputs/loss%d.png' % (n_simulations), description='batch loss')
  except ImportError:
    pass

  # save nf parameters
  with open("./outputs/params_nd%d.pkl" % (n_simulations), "wb") as fp:
    pickle.dump(params_smooth, fp)

  # compute metric
  sample_smooth_model = hk.transform(
      lambda : tfd.TransformedDistribution(
          NF_smooth(2)(jnp.ones([10000,1])),
          tfb.Invert(transformation_params)).sample(10000, seed=hk.next_rng_key()))


  predicted_samples = sample_smooth_model.apply(params_smooth, rng = next(rng_seq))
  jnp.save('./outputs/predicted_samples%d.npy' % (n_simulations), predicted_samples)

  true_posterior_samples = two_moons.sample(10000, seed=jax.random.PRNGKey(1000))

  
  c2st_metric = c2st(true_posterior_samples, predicted_samples, seed=0, n_folds=5)
  neg_log_prob = -jnp.mean(
    jax.vmap(
      lambda x: model_smooth.apply(params_smooth, x.reshape([1,2])).squeeze())(true_posterior_samples)
      )
  c2st_save.append(c2st_metric)
  nlp_save.append(neg_log_prob)

  if ON_AZURE:
    run.log('c2st_metric', float(c2st_metric))
    run.log('neg_log_prob', float(neg_log_prob))
  else:
    print(c2st_metric)
    print(neg_log_prob)

  # plot results
  try:
    
    x = jnp.stack(jnp.meshgrid(jnp.linspace(0.,1.,128),
                              jnp.linspace(0.,1.,128)),-1)
    im0 = jax.vmap(
          lambda x: model_smooth.apply(params_smooth, 
          x.reshape([1,2])).squeeze())(x.reshape([-1,2])).reshape([128,128])

    plt.clf()
    plt.contourf(x[...,0],x[...,1],jnp.exp(im0), 100); plt.colorbar()
    plt.savefig("./outputs/contour_plot%d.png" % (n_simulations))

    # (for publication) 
    plt.clf()
    plt.figure(dpi=100)
    plt.contourf(x[...,0],x[...,1],jnp.exp(im0), 100,cmap='Oranges')
    plt.axis('off')
    plt.savefig('./outputs/contour_plotforpubli%d.png' % (n_simulations), 
                transparent=True,
                bbox_inches='tight',
                pad_inches = 0)

    if ON_AZURE:
      run.log_image(name='contourplot', 
                    path='./outputs/contour_plot%d.png' % (n_simulations), 
                    description='contour plot')
      run.log_image(name='contourplotforpubli', 
                    path='./outputs/contour_plotforpubli%d.png' % (n_simulations), 
                    description='contour plot for publi')
  except ImportError:
    pass



jnp.save('./outputs/c2st.npy' , c2st_save)
jnp.save('./outputs/nlp.npy' , nlp_save)

    
      # run.log_image(name='contour_plot', path='./outputs/contour_plot.png', description='contour plot of the predicted posterior vs true posterior')
