# worse convergence
"""
Lecture 04: Hi, automatic vectorisation!

Demonstration: Implement and train a CNN on MNIST with minibatch SGD.

Learning objectives:

* more practice with PyTrees
* introducing jax.vmap
"""

import dataclasses
import functools
from typing import Self

import einops
import jax
import jax.numpy as jnp
import jaxtyping
import matthewplotlib as mp
import numpy as np

# import time # unused
import tyro
from jaxtyping import Array, Float, Int, PRNGKeyArray

# # #
# Training loop


def normalise(x: Float[Array, "b h w"]) -> Float[Array, "b h w"]:
    return 1.275 * x / 255 - 0.1


def pad(x: Float[Array, "b h w"]) -> Float[Array, "b h+2 w+2"]:
    return jnp.pad(
        x,
        pad_width=((0, 0), (2, 2), (2, 2)),
        mode="constant",
        constant_values=0,
    )


def get_parameters_count(model: jaxtyping.PyTree):
    return jax.tree.reduce(lambda cu, leaf: cu + leaf.size, model, initializer=0)


def main(
    learning_rate: float = 0.05,
    batch_size: int = 512,
    num_steps: int = 256,
    steps_per_visualisation: int = 4,
    seed: int = 42,
):
    key = jax.random.key(seed=seed)

    print("load the training data...")
    datafile = jnp.load("../data/mnist.npz")
    x_train = jnp.array(datafile["x_train"])
    y_train = jnp.array(datafile["y_train"])
    x_test = jnp.array(datafile["x_test"])
    y_test = jnp.array(datafile["y_test"])

    x_train = normalise(pad(x_train))
    x_test = normalise(pad(x_test))

    # Show data chunks
    print(x_train.shape, x_train.dtype)
    print(y_train.shape, y_train.dtype)

    # Show image in TUI
    print(mp.image(x_train[0].astype(jnp.float32)), y_train[0])

    print("initialising the model...")
    key_model_init, key = jax.random.split(key)
    model = SimpLeNet.init(key_model_init)

    print(f"Model parameters count: {get_parameters_count(model)}")

    print("training...")
    losses = []
    accuracies = []
    for step in range(num_steps):
        # sample in a small batch
        key_batch, key = jax.random.split(key)
        batch_idx = jax.random.choice(
            key=key_batch,
            a=x_train.shape[0],
            shape=(batch_size,),
            replace=False,
        )
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        # forward & backward
        loss, grad = jax.value_and_grad(cross_entropy)(model, x_batch, y_batch)

        # update gradient
        model = jax.tree.map(lambda w, g: w - learning_rate * g, model, grad)

        # show metrics
        losses.append((step, loss.item()))

        test_acc = get_accuracy(model, x_test[:1000], y_test[:1000])
        accuracies.append((step, test_acc.item()))

        # vis
        plot = None

        if step % steps_per_visualisation == 0:
            digit_plot = vis_digits(
                digits=x_test[np.asarray((1, 6, 7, 8))],
                labels=y_test[np.asarray((1, 6, 7, 8))],
                model=model,
            )
            metrics_plot = vis_metrics(
                losses=losses,
                accuracies=accuracies,
                total_num_steps=num_steps,
            )
            plot = digit_plot / metrics_plot

            if step == 0:
                print(plot)
            else:
                print(f"{-plot}{plot}")

        if plot:
            plot.saveimg("result.png")


# # #
# Architecture


normal = functools.partial(jax.random.truncated_normal, lower=-2, upper=+2)


def scaled_tanh(x: Array) -> Array:
    return 1.7159 * jnp.tanh(0.6667 * x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SimpleConv2d:
    kernel: Float[Array, "k k i o"]
    bias: Float[Array, "o 1 1"]

    def __call__(
        self,
        x: Float[Array, "i h+k-1 w+k-1"],
    ) -> Float[Array, "o h w"]:
        x = x[None]
        x = jax.lax.conv_general_dilated(
            lhs=x.astype(jnp.float32),
            rhs=self.kernel.astype(jnp.float32),
            window_strides=(1, 1),
            padding=((0, 0), (0, 0)),
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        ).astype(jnp.float32)
        # x = x[0] + self.bias
        x = x[0]
        return x

    @staticmethod
    def init(
        key: PRNGKeyArray,
        ksize: int,
        din: int,
        dout: int,
    ):
        kernel = normal(
            key=key,
            shape=(ksize, ksize, din, dout),
            dtype=jnp.float32,
        )
        kernel = kernel * jnp.sqrt(2 / (din + dout))  # xavier_normal
        bias = jnp.zeros(shape=(dout, 1, 1), dtype=jnp.float32)
        return SimpleConv2d(kernel=kernel, bias=bias)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Subsample2x2:
    weight: Float[Array, "c 1 1"]
    bias: Float[Array, "c 1 1"]

    def __call__(
        self,
        x: Float[Array, "c h w"],
    ) -> Float[Array, "c h//2 w//2"]:
        x = einops.reduce(x, "c (h 2) (w 2) -> c h w", "sum")
        return x * self.weight + self.bias

    @staticmethod
    def init(
        key: PRNGKeyArray,
        din: int,
    ):
        weight = jnp.ones(shape=(din, 1, 1), dtype=jnp.float32) * 0.25
        bias = jnp.zeros(shape=(din, 1, 1), dtype=jnp.float32)
        return Subsample2x2(weight=weight, bias=bias)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AffineTransform:
    weight: Float[Array, "i o"]
    bias: Float[Array, "o"]

    def __call__(
        self,
        x: Float[Array, "i"],
    ) -> Float[Array, "o"]:
        return x @ self.weight + self.bias

    @staticmethod
    def init(
        key: PRNGKeyArray,
        din: int,
        dout: int,
    ):
        weight = normal(
            key=key,
            shape=(din, dout),
            dtype=jnp.float32,
        )
        weight = weight * jnp.sqrt(2 / (din + dout))  # xavier_normal
        bias = jnp.zeros(shape=dout, dtype=jnp.float32)
        return AffineTransform(weight=weight, bias=bias)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SimpLeNet:
    c1: SimpleConv2d
    s2: Subsample2x2
    c3: SimpleConv2d
    s4: Subsample2x2
    c5: SimpleConv2d
    f6: AffineTransform
    out: AffineTransform

    def __call__(self, x: Float[Array, "32 32"]) -> Float[Array, "10"]:
        x = x[None]
        x = scaled_tanh(self.c1(x))
        x = scaled_tanh(self.s2(x))
        x = scaled_tanh(self.c3(x))
        x = scaled_tanh(self.s4(x))
        x = scaled_tanh(self.c5(x))
        x = jnp.ravel(x)
        x = scaled_tanh(self.f6(x))
        x = self.out(x)
        return jax.nn.softmax(x)

    def batch_inference(self, x: Float[Array, "b 32 32"]) -> Float[Array, "b 10"]:
        return jax.vmap(self.__call__)(x)

    @staticmethod
    def init(key: PRNGKeyArray):
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        c1 = SimpleConv2d.init(k1, ksize=5, din=1, dout=6)
        s2 = Subsample2x2.init(k2, din=6)
        c3 = SimpleConv2d.init(k3, ksize=5, din=6, dout=16)
        s4 = Subsample2x2.init(k4, din=16)
        c5 = SimpleConv2d.init(k5, ksize=5, din=16, dout=120)
        f6 = AffineTransform.init(k6, din=120, dout=84)
        out = AffineTransform.init(k7, din=84, dout=10)
        return SimpLeNet(c1=c1, s2=s2, c3=c3, s4=s4, c5=c5, f6=f6, out=out)


# # #
# Metrics


@jax.jit
def cross_entropy(
    model: SimpLeNet,
    xs: Float[Array, "b h w"],
    ys: jaxtyping.UInt8[Array, "b"],
) -> Float[Array, ""]:
    probs = model.batch_inference(xs)
    probs = jnp.take_along_axis(probs, ys[:, None], axis=1).squeeze(1)
    losses = -jnp.log(probs)
    return jnp.mean(losses, axis=0)


@jax.jit
def get_accuracy(
    model: SimpLeNet,
    xs: Float[Array, "b h w"],
    ys: jaxtyping.UInt8[Array, "b"],
) -> Float[Array, ""]:
    probs = model.batch_inference(xs)
    preds = jnp.argmax(probs, axis=1, keepdims=False)
    accuracies = preds == ys
    return jnp.mean(jnp.where(accuracies, 1, 0), axis=0)


# # #
# Visualisation


def vis_digits(
    digits: Float[Array, "n h w"],
    labels: Int[Array, "n"],
    model: SimpLeNet,
) -> mp.plot:
    # shrink and normalise images
    digs = einops.reduce(
        (digits + 0.1) / 1.275,
        "b (h 2) (w 2) -> b h w",
        "mean",
    )
    width = digs.shape[-1]

    # classify digits and mark correct or incorrect
    pred_probs = model.batch_inference(digits)
    pred_labels = pred_probs.argmax(axis=-1)
    corrects = labels == pred_labels
    cmaps = [mp.cyans if correct else mp.magentas for correct in corrects]

    # build the visualisation
    array = mp.wrap(
        *[
            mp.text("p( digit | image )")
            / mp.columns(
                probs,
                height=6,
                vrange=1,
                column_width=1,
                column_spacing=1,
                colors=[mp.cyber(i == label) for i in range(10)],
            )
            / mp.text(" ".join(str(d) for d in range(10)))
            + mp.text("image") / mp.image(dig, colormap=cmap)
            for dig, label, probs, cmap in zip(digs, labels, pred_probs, cmaps)
        ],
        cols=2,
    )
    return array


def vis_metrics(
    losses: list[tuple[int, float]],
    accuracies: list[tuple[int, float]],
    total_num_steps: int,
) -> mp.plot:
    losses = np.asarray(losses)
    accuracies = np.asarray(accuracies)
    loss_plot = mp.axes(
        mp.scatter(
            (losses, "magenta"),
            xrange=(0, total_num_steps - 1),
            yrange=(0, max(l for s, l in losses)),
            width=28,
            height=9,
        ),
        title=f"cross entropy {losses[-1][1]:.3f}",
        xlabel="train steps",
    )
    acc_plot = mp.axes(
        mp.scatter(
            (accuracies, "cyan"),
            xrange=(0, total_num_steps - 1),
            yrange=(0, 1),
            width=28,
            height=9,
        ),
        title=f"test accuracy {accuracies[-1][1]:.2%}",
        xlabel="train steps",
    )
    return loss_plot + acc_plot


# # #
# Entry point

if __name__ == "__main__":
    tyro.cli(main)
