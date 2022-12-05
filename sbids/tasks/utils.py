from numpyro.handlers import seed, trace, condition
import jax

def get_samples_and_scores(model, key, batch_size=64, score_type='density', thetas=None):
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

        logp = 0 
        for i in range(len(model_trace) - 1): 
          key, val = list(model_trace.items())[i]
          
          if not (key == 'theta' and score_type == 'conditional'):
            logp += val['fn'].log_prob(val['value']).sum()
        
        logp += model_trace['y']['fn'].log_prob(jax.lax.stop_gradient(model_trace['y']['value'])).sum()

        sample = {'theta': model_trace['theta']['value'],
                  'y': model_trace['y']['value']}


        return logp, sample
    
    # Split the key by batch
    keys = jax.random.split(key, batch_size)

    # Sample theta from the model
    if thetas is None:
        thetas = jax.vmap(lambda k: trace(seed(model, k)).get_trace()['theta']['value'])(keys)

    return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)