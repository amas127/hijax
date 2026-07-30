"""
Lecture 1: Hi, automatic differentiation!

Demonstration: Train a teacher--student linear regression model with gradient
descent.

Learning objectives:

* more jax.numpy
* introducing functional model API
* introducing jax.grad
"""

import time

import jax
import jax.numpy as jnp
import matthewplotlib as mp
import tyro
from jaxtyping import Array, Float, Scalar


def main(
    num_steps: int = 400,
    learning_rate: float = 0.01,
):
    # Define teacher
    w_teacher = jnp.array([0.5, -1.0], dtype=jnp.float32)

    # Define student
    w_student = jnp.array([-1.0, 3.0], dtype=jnp.float32)

    print(vis(w_student, w_teacher, step=0, loss=jnp.inf))

    for step in range(1, num_steps + 1):
        # l = loss(w_student, w_teacher)
        l, g = jax.value_and_grad(loss)(w_student, w_teacher)
        w_student -= g * learning_rate

        plot = vis(w_student, w_teacher, step, l)
        print(f"\x1b[{plot.height}A{plot}")
        time.sleep(0.02)


def loss(
    w_student: Float[Array, "2"],
    w_teacher: Float[Array, "2"],
) -> Scalar:
    x = jnp.linspace(-4, 4, 80)
    y_student = forward(w_student, x)
    y_teacher = forward(w_teacher, x)
    errors = y_teacher - y_student
    mse: Scalar = jnp.mean(errors**2)
    return mse


def forward(
    w: Float[Array, "2"],
    x: Float[Array, "b"],
) -> Float[Array, "b"]:
    a, b = w
    return a * x + b


def vis(
    w_student: Float[Array, "2"],
    w_teacher: Float[Array, "2"],
    step: int,
    loss: float | Scalar,
) -> mp.plot:
    x = jnp.linspace(-4, 4, 80)
    return mp.axes(
        mp.scatter(
            mp.xaxis(-4, 4, 80),
            mp.yaxis(-4, 4, 80),
            (x, forward(w_teacher, x), "cyan"),
            (x, forward(w_student, x), "magenta"),
            height=20,
            width=40,
            xrange=(-4, 4),
            yrange=(-4, 4),
        ),
        title=f"step {step:03d} | loss {loss:6.3f}",
        ylabel="y",
        xlabel="x",
    )


if __name__ == "__main__":
    tyro.cli(main)
