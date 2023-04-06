import jax
import jax.numpy as jnp
import numpy as np
import optax
from sbids.tasks.utils import get_samples_and_scores
from sbids.tasks.two_moons import get_two_moons
from tqdm import tqdm
from functools import partial


class train_model():

    def loss_sbi(
        self,
        model_params,
        theta,
        x,
        score,
        nll_weight,
        score_weight
    ):

        log_prob, out = jax.vmap(
            jax.value_and_grad(
                self.model_logp(model_params)
            )
        )(theta, x)

        nll = -jnp.mean(log_prob)
        mse = jnp.mean(jnp.sum((out - score)**2, axis=1))

        return nll_weight * nll + score_weight * mse

    def loss_sbi_with_compressor(
        self,
        model_params,
        theta,
        x,
        score,
        nll_weight,
        score_weight
    ):

        y = self.compressor.apply(self.params_compressor, x)
        loss = self.loss_sbi(
            model_params,
            theta,
            y,
            score,
            nll_weight,
            score_weight
        )

        return loss

    def __init__(
        self,
        model_logp,
        optimizer,
        task_name,
        numpyro_task=None,
        compressor=None,
        params_compressor=None,
    ):

        self.model_logp = model_logp
        self.optimizer = optimizer
        self.task_name = task_name
        self.numpyro_task = numpyro_task
        self.compressor = compressor
        self.params_compressor = params_compressor

        if self.compressor is None:
            self.loss = self.loss_sbi
        else:
            self.loss = self.loss_sbi_with_compressor

    def get_batch(self, key, nb_simulations):

        if self.task_name == 'two_moons':
            two_moons = get_two_moons(sigma=0.01)
            x = two_moons.sample(nb_simulations, seed=key)
            score = jax.vmap(jax.grad(two_moons.log_prob))(x)
            theta = jnp.ones(x.shape) * 0.01

        else:
            (_, sample), score = get_samples_and_scores(
                self.numpyro_task,
                key,
                batch_size=nb_simulations
            )

            theta = sample['theta']
            x = sample['y']
            score = score

            if self.task_name == 'lotka_volterra':
                x = x.reshape([-1, 20], order='F')

        if jnp.isnan(score).any() is True:
            idx = jnp.where(jnp.isnan(score))[0]
            x = jnp.delete(x, idx, axis=0)
            theta = jnp.delete(theta, idx, axis=0)
            score = jnp.delete(score, idx, axis=0)

        return (theta, x, score)

    def data_stream(self, key, batch_size, dataset):

        theta = dataset[0]
        x = dataset[1]
        score = dataset[2]

        dataset_size = len(score)

        rng = np.random.RandomState(key)

        while True:
            perm = rng.permutation(dataset_size)
            for i in range(dataset_size // batch_size):
                batch_idx = perm[i * batch_size: (i + 1) * batch_size]
                yield theta[batch_idx], x[batch_idx], score[batch_idx]

    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        model_params,
        opt_state,
        theta,
        x,
        score,
        nll_weight,
        score_weight
    ):

        loss, grads = jax.value_and_grad(self.loss)(
            model_params,
            theta,
            x,
            score,
            nll_weight,
            score_weight
        )
        updates, new_opt_state = self.optimizer.update(
            grads,
            opt_state,
            model_params
        )
        new_params = optax.apply_updates(model_params, updates)

        return loss, new_params, new_opt_state

    def train(
        self,
        model_params,
        epoch,
        batch_size,
        nb_simulations,
        nll_weight,
        score_weight,
        jax_key,
        numpy_key
    ):

        dataset = self.get_batch(jax_key, nb_simulations)
        batch_loss = []
        opt_state = self.optimizer.init(model_params)

        batch_generator = self.data_stream(numpy_key, batch_size, dataset)

        for epoch in tqdm(range(epoch)):
            for step in range(nb_simulations // batch_size):

                theta, x, score = next(batch_generator)

                l, model_params, opt_state = self.update(
                    model_params,
                    opt_state,
                    theta,
                    x,
                    score,
                    nll_weight,
                    score_weight
                )

                batch_loss.append(l)

                if jnp.isnan(l):
                    print('NaN in loss')
                    break

        return batch_loss, opt_state, model_params
