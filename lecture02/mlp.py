"""
Lecture 02: Hi, procedural random number generation!

Demonstration: Implement and train a classical perceptron with classical
stochastic gradient descent.

Learning objectives:

* introducing jax.random
* more practice with jax arrays, jax.grad
"""

import time
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import matthewplotlib as mp
import numpy as np
import tyro
from jaxtyping import Array, Bool, Float

# # #
# ENTRY POINT


def main(
    num_points: int = 256,
    learning_rate: float = 0.2,
    seed: int = 42,
):
    key = jax.random.key(seed)

    key_pos, key = jax.random.split(key, 2)

    xs_pos = jax.random.multivariate_normal(
        key_pos,
        mean=jnp.ones(2),
        cov=0.25 * jnp.eye(2),
        shape=(num_points // 2,),
    )
    ys_pos = jnp.ones(num_points // 2, dtype=jnp.bool)

    key_neg, key = jax.random.split(key, 2)

    xs_neg = jax.random.multivariate_normal(
        key_neg,
        mean=-jnp.ones(2),
        cov=0.25 * jnp.eye(2),
        shape=(num_points // 2,),
    )
    ys_neg = jnp.zeros(num_points // 2, dtype=jnp.bool)

    xs = jnp.concat((xs_pos, xs_neg), axis=0)
    ys = jnp.concat((ys_pos, ys_neg), axis=0)

    print(vis_data(xs, ys))

    key_shuffle, key = jax.random.split(key, 2)

    pi = jax.random.permutation(key_shuffle, xs.shape[0])
    xs = xs[pi]
    ys = ys[pi]

    key_model_init, key = jax.random.split(key, 2)

    w = jax.random.truncated_normal(key_model_init, lower=-2.0, upper=+2.0, shape=(25,))

    print(vis_model(w, xs, step=1))

    for t, (x, y) in enumerate(zip(xs, ys), 1):
        l, g = jax.value_and_grad(loss)(w, x, y)
        w = w - learning_rate * g

        plot = vis_model(w, xs, step=t)
        print(f"{-plot}{plot}")
        time.sleep(0.02)


def loss(
    w: Float[Array, "25"],
    x: Float[Array, "2"],
    y: Bool[Array, ""],
) -> Float[Array, ""]:
    logit = forward_single_sample(w, x)
    ce = jnp.logaddexp(0, logit) - y * logit
    return ce


def forward_single_sample(
    w: Float[Array, "25"],
    x: Float[Array, "2"],
) -> Float[Array, ""]:
    w1 = w[0 : 2 * 8].reshape(8, 2)
    w2 = w[2 * 8 : 2 * 8 + 8 * 1].reshape(1, 8)
    b2 = w[24].reshape(1)

    x1 = w1 @ x
    x2 = jnp.where(x1 > 0, x1, 0)
    x3 = w2 @ x2
    x4 = x3 + b2

    return x4.reshape()


forward = jax.vmap(
    forward_single_sample,
    in_axes=[None, 0],
    out_axes=0,
)


# # #
# VISUALISATION CODE


def vis_data(
    xs: Float[Array, "n 2"],
    ys: Bool[Array, "n"],
) -> mp.plot:
    return mp.axes(
        mp.scatter(
            (xs[:, 0], xs[:, 1], mp.cyber(ys.astype(float))),
            xrange=(-3, +3),
            yrange=(-3, +3),
            width=40,
            height=20,
        ),
        title="ground truth labels",
        xlabel="x0",
        ylabel="x1",
    )


@jax.jit
def soft_classify(
    w: Float[Array, "25"],
    xs: Float[Array, "n 2"],
) -> Float[Array, "n"]:
    return jax.nn.sigmoid(forward(w, xs))


@partial(jax.jit, static_argnames=("n_grid", "threshold"))
def infer_decision_boundary(
    w: Float[Array, "25"],
    n_grid: int = 100,
    threshold: float = 0.05,
) -> tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n"]]:
    """Mlp generate boundary regions instead of a formulable line."""
    xgrids, ygrids = jnp.meshgrid(
        jnp.linspace(-3, +3, num=n_grid),
        jnp.linspace(-3, +3, num=n_grid),
    )
    xgrids, ygrids = xgrids.ravel(), ygrids.ravel()
    grid_pts = jnp.stack([xgrids, ygrids], axis=-1)
    grid_preds = soft_classify(w, grid_pts)
    boundary = jnp.abs(grid_preds - 0.5) < threshold
    return xgrids, ygrids, boundary


def vis_model(w, xs, step):
    ys_pred = soft_classify(w, xs)
    xgrids, ygrids, boundary = infer_decision_boundary(w)
    bx = xgrids[boundary]
    by = ygrids[boundary]

    series: list[tuple[tp.Any, tp.Any, tp.Any]] = [
        (xs[:, 0], xs[:, 1], mp.cyber(ys_pred)),
    ]
    if bx.size > 0:
        series.append((bx, by, "white"))

    return mp.axes(
        mp.scatter(
            *series,
            xrange=(-3, +3),
            yrange=(-3, +3),
            width=40,
            height=20,
        ),
        title=f"model predictions @ step {step:3d}",
        xlabel="x0",
        ylabel="x1",
    )


if __name__ == "__main__":
    tyro.cli(main)
