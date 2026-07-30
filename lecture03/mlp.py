"""
Lecture 03: Hi, PyTrees!

Demonstration: Implement and train a multi-layer perceptron on XOR data with
minibatch SGD.

Learning objectives:

* more jax.numpy and jax.random
* introducing "PyTrees"
* introducing jax.tree and jax.tree_util
"""

import dataclasses
import time
from functools import partial
from typing import Self

import einops
import jax
import jax.numpy as jnp
import matthewplotlib as mp
import tyro
from jaxtyping import Array, Bool, Float, PRNGKeyArray

# # #
# MODEL CODE

# type Model = tuple[
#     Float[Array, "h 2"],
#     Float[Array, "h"],
#     Float[Array, "1 h"],
#     Float[Array, "1"],
# ]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Model:
    w1: Float[Array, "h 2"]
    b1: Float[Array, "h"]
    w2: Float[Array, "1 h"]
    b2: Float[Array, "1"]


@jax.jit
@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
def forward(
    mod: Model,
    x: Float[Array, "2"],
) -> Float[Array, ""]:
    w1, b1, w2, b2 = mod.w1, mod.b1, mod.w2, mod.b2
    x = x @ w1.T + b1
    x = jax.nn.relu(x)
    x = x @ w2.T + b2
    return x.reshape()


def init_model(
    key: PRNGKeyArray,
    num_hidden: int,
) -> Model:
    key_w1, key_w2 = jax.random.split(key, 2)
    normal = partial(jax.random.truncated_normal, lower=-2, upper=+2, dtype=jnp.float32)
    zeros = partial(jnp.zeros, dtype=jnp.float32)
    w1 = normal(key_w1, shape=(num_hidden, 2)) / jnp.sqrt(num_hidden)
    b1 = zeros(num_hidden)
    w2 = normal(key_w2, shape=(1, num_hidden)) / jnp.sqrt(num_hidden)
    b2 = zeros(1)
    return Model(w1=w1, b1=b1, w2=w2, b2=b2)


# # #
# TRAINING CODE


@jax.jit
def loss(
    mod: Model,
    xs: Float[Array, "b 2"],
    ys: Float[Array, "b"],
) -> Float[Array, ""]:
    logits = forward(mod, xs)
    ce = jnp.logaddexp(0, logits) - ys * logits
    return jnp.mean(ce)


def main(
    num_points: int = 1024,
    num_steps: int = 512,
    learning_rate: float = 0.1,
    num_hidden: int = 16,
    minibatch_size: int = 64,
    seed: int = 42,
):
    # TODO
    key = jax.random.key(seed)

    key_data, key = jax.random.split(key)
    xs = jax.random.multivariate_normal(
        key=key_data,
        mean=jnp.zeros(2),
        cov=jnp.eye(2),
        shape=(num_points,),
    )
    cs = einops.repeat(jnp.arange(4), "n -> (n k)", k=num_points // 4)
    xs = (
        0.5 * xs
        + jnp.array([
            (-1, -1),
            (+1, +1),
            (-1, +1),
            (+1, -1),
        ])[cs]
    )
    ys = cs // 2

    print(vis_data(xs, ys))

    key_mod, key = jax.random.split(key, 2)
    mod = init_model(key_mod, num_hidden)

    print(vis_model(mod, xs, ys, -1))
    plot = None
    key_train, key = jax.random.split(key, 2)
    for t in range(num_steps):
        # sample minibatch
        key_minibatch, key_train = jax.random.split(key_train)
        minibatch_idx = jax.random.choice(
            key_minibatch,
            num_points,
            shape=(minibatch_size,),
            replace=False,
        )
        xs_minibatch = xs[minibatch_idx]
        ys_minibatch = ys[minibatch_idx]

        l, g = jax.value_and_grad(loss)(mod, xs_minibatch, ys_minibatch)

        mod = jax.tree.map(lambda x, y: x - learning_rate * y, mod, g)

        plot = vis_model(mod, xs, ys, t)
        print(f"{-plot}{plot}")
        time.sleep(0.02)

    if plot is not None:
        plot.saveimg("result.png")


# # #
# VISUALISATION


def vis_data(
    xs: Float[Array, "n 2"],
    ys: Bool[Array, "n"],
) -> mp.plot:
    return mp.axes(
        mp.scatter(
            (xs[:, 0], xs[:, 1], mp.cyber(ys)),
            xrange=(-3, +3),
            yrange=(-3, +3),
            width=40,
            height=20,
        ),
        title="ground truth labels",
        xlabel="x0",
        ylabel="x1",
    )


def vis_model(
    w: Model,
    xs: Float[Array, "n 2"],
    ys: Bool[Array, "n"],
    step: int,
) -> mp.plot:
    # compute predictions
    ys_pred = jax.nn.sigmoid(forward(w, xs))
    # ys_pred = jax.nn.sigmoid(w.forward(xs)[:, 0])

    # plot
    return mp.axes(
        mp.dstack2(
            mp.function2(
                lambda xs: jax.nn.sigmoid(forward(w, xs)),
                # F=lambda xs: jax.nn.sigmoid(w.forward(xs)[:,0]),
                xrange=(-3, 3),
                yrange=(-3, 3),
                width=40,
                height=20,
                zrange=(0.0, 1.0),
                colormap=lambda z: 0.5 * (mp.cyber(z) / 255),
                endpoints=True,
            ),
            mp.scatter(
                (xs[:, 0], xs[:, 1], mp.cyber(ys)),
                xrange=(-3, +3),
                yrange=(-3, +3),
                width=40,
                height=20,
            ),
        ),
        title=f"model predictions @ step {step + 1:3d}",
        xlabel="x0",
        ylabel="x1",
    )


if __name__ == "__main__":
    tyro.cli(main)
