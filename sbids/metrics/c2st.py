
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class Classifier_C2ST(hk.Module):

  def __call__(self, x):

    net_x = hk.Linear(8)(x) 
    net_x = jax.nn.leaky_relu(net_x)
    net_x = hk.Linear(16)(net_x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Linear(32)(net_x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Linear(16)(net_x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Linear(8)(net_x) 
    net_x = jax.nn.leaky_relu(net_x) 
    net_x = hk.Linear(1)(net_x) 
    net_x = jax.nn.sigmoid(net_x).squeeze()
    net_x = jnp.clip(net_x, 1e-5, 1.-1e-5)

    return net_x


def c2st(sample1,sample2,num_seed):
  """
  Classifier Two-Sample Test

  Train a binary classifier to evaluate wether sample1 and sample2 
  are from the same distribution. 
  Each sample must be of size +-10000 to work properly.
  Parameters
   ----------
   sample1: First sample.
   sample2 : Second sample.
   num_seed : seed number
   Returns
   -------
     Accuracy.
   
  """

  
  sample1_mean = jnp.mean(sample1, axis = 0)
  sample1_std = jnp.std(sample1, axis = 0)
  sample1 = (sample1 - sample1_mean) / sample1_std
  sample2 = (sample2 - sample1_mean) / sample1_std

  label1 = jnp.zeros(len(sample1)) 
  label2 = jnp.ones(len(sample2))  
  data = jnp.concatenate([jnp.concatenate([sample1,label1.reshape(-1,1)],axis = 1),
                          jnp.concatenate([sample2,label2.reshape(-1,1)],axis = 1)],axis = 0)

  data = jax.random.permutation(jax.random.PRNGKey(num_seed), data, axis=0)
  ntrain = int((len(sample1)+len(sample2))*0.6)
  dtrain, dte = data[:ntrain], data[ntrain:]

  sample_dim = sample1.shape[1]
  ztrain = dtrain[...,:sample_dim]
  ltrain = dtrain[...,sample_dim:].squeeze()
  zte = dte[...,:sample_dim]
  lte = dte[...,sample_dim:].squeeze()

  batch_size = 560
  num_batches = ntrain // batch_size 

  def data_stream():
      """
      Creates a data stream with a predifined batch size.
      """
      rng = np.random.RandomState(0)
      while True:
        perm = rng.permutation(ntrain)
        for i in range(num_batches):
          batch_idx = perm[i * batch_size: (i + 1)*batch_size]
          yield ztrain[batch_idx], ltrain[batch_idx]

  batches = data_stream()


  c2stClassifier = hk.without_apply_rng(hk.transform(lambda x : Classifier_C2ST()(x)))
  rng_seq = hk.PRNGSequence(num_seed)
  params_c2stClassifier = c2stClassifier.init(next(rng_seq), 0.5*jnp.ones(sample_dim))
  scheduler = optax.exponential_decay(init_value=0.001, transition_steps=2000, decay_rate=0.9, end_value=0.00001)
  optimizer_c2stClassifier = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1))
  opt_state_c2stClassifier = optimizer_c2stClassifier.init(params_c2stClassifier)


  @jax.jit
  def logloss(params,batch,label):
      """Loss"""
      prob = jax.vmap(lambda x : c2stClassifier.apply(params,x.reshape(4)))(batch)
      test = (1-label)*jnp.log(1-prob)
      return -jnp.mean(label*jnp.log(prob)+(1-label)*jnp.log(1-prob))

  @jax.jit
  def update_c2stClassifier(params, opt_state, batch, label):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(logloss)(params, batch, label)
    updates, new_opt_state = optimizer_c2stClassifier.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # train classifier
  losses = []
  num_epochs = 2000
  for epoch in tqdm(range(num_epochs)): 
    for _ in range(num_batches):
      z,l = next(batches)    
      loss, params_c2stClassifier, opt_state_c2stClassifier = update_c2stClassifier(params_c2stClassifier, opt_state_c2stClassifier, z,l)
      losses.append(loss)
    

  # compute metric 
  prob = jax.vmap(lambda x : c2stClassifier.apply(params_c2stClassifier,x.reshape(sample_dim)))(zte)
  prob = prob.at[prob <= 0.5].set(0)
  prob = prob.at[prob > 0.5].set(1)
  t = jnp.mean(prob==lte)

  return losses, t