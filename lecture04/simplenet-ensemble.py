import dataclasses
import typing as tp
from os import PathLike

import einops
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import matthewplotlib as mp
import nnlogging2
import numpy as np
import tyro
from jaxtyping import Array, Float

# --- Logging --- #

logger = nnlogging2.get_metric_logger(__name__)
logger.metric_attrs = ("lastval", "median")
formatter = nnlogging2.ContextedMetricFormatter(
    "%(levelname)-17s %(context_ext)s %(message)s%(metrics_ext)s"
)
handler = nnlogging2.AttyStreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


# --- Transformations ---  #


def normalise(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return 1.275 * x / 255.0 - 0.1


def addc(x: Float[Array, "*d"]) -> Float[Array, "1 *d"]:
    return x[None]


def pad(x: Float[Array, "h w"]) -> Float[Array, "h+4 w+4"]:
    return jnp.pad(x, ((2, 2), (2, 2)), mode="constant", constant_values=0)


def transform(x: Float[Array, "28 28"]) -> Float[Array, "1 32 32"]:
    x = pad(x)
    x = addc(x)
    x = normalise(x)
    return x


vtransform = jax.vmap(transform)


# --- Helpers --- #


def get_num_trainable_parameters(model: jtp.PyTree):

    def _get_leaf_sz(leaf: tp.Any) -> int:
        if isinstance(leaf, jax.Array):
            return leaf.size
        return 0

    return jax.tree.reduce(
        lambda cu, leaf: cu + _get_leaf_sz(leaf),
        tree=model,
        initializer=0,
    )


def get_padded_array(x: tp.Any, size: int, *, fill_value=0, dtype: jnp.dtype) -> Array:
    return jnp.pad(jnp.asarray(x, dtype=dtype), (0, size), constant_values=fill_value)[
        :size
    ]


# --- Model --- #

PRECISION = jnp.bfloat16


def scaled_tanh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    x = x.astype(PRECISION)
    return 1.7159 * jnp.tanh(0.6667 * x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Conv2d:
    kernel: Float[Array, "o i k k"]
    bias: Float[Array, "o 1 1"]

    def __call__(self, x: Float[Array, "i h+k-1 w+k-1"]) -> Float[Array, "o h w"]:
        x = x[None].astype(PRECISION)
        kernel = self.kernel.astype(PRECISION)
        bias = self.bias.astype(PRECISION)

        x = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=(1, 1),
            padding=((0, 0), (0, 0)),
            # dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        x = x[0] + bias
        return x

    @classmethod
    def init(
        cls,
        key: jtp.PRNGKeyArray,
        ksize: int,
        din: int,
        dout: int,
    ):
        kernel_initializer = jax.nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_geo_avg",
            distribution="truncated_normal",
            in_axis=1,
            out_axis=0,
        )
        kernel = kernel_initializer(
            key=key,
            shape=(dout, din, ksize, ksize),
            dtype=jnp.float32,
        )
        bias = jnp.zeros((dout, 1, 1), dtype=jnp.float32)
        return Conv2d(kernel=kernel, bias=bias)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Pool2d:
    weight: Float[Array, "c 1 1"]
    bias: Float[Array, "c 1 1"]

    def __call__(
        self,
        x: Float[Array, "c 2*h 2*w"],
    ) -> Float[Array, "c h w"]:
        x = x.astype(PRECISION)
        weight = self.weight.astype(PRECISION)
        bias = self.bias.astype(PRECISION)

        x = einops.reduce(x, "c (h 2) (w 2) -> c h w", "mean")
        x = x * weight + bias
        return x

    @classmethod
    def init(cls, din: int):
        weight = jnp.ones((din, 1, 1), dtype=jnp.float32)
        bias = jnp.zeros((din, 1, 1), dtype=jnp.float32)
        return Pool2d(weight=weight, bias=bias)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Linear:
    weight: Float[Array, "i o"]
    bias: Float[Array, "o"]

    def __call__(
        self,
        x: Float[Array, "i"],
    ) -> Float[Array, "o"]:
        x = x.astype(PRECISION)
        weight = self.weight.astype(PRECISION)
        bias = self.bias.astype(PRECISION)

        x = x @ weight + bias
        return x

    @classmethod
    def init(
        cls,
        key: jtp.PRNGKeyArray,
        din: int,
        dout: int,
    ):
        weight_initializer = jax.nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_geo_avg",
            distribution="truncated_normal",
            in_axis=0,
            out_axis=1,
        )
        weight = weight_initializer(key=key, shape=(din, dout), dtype=jnp.float32)
        bias = jnp.zeros(dout, dtype=jnp.float32)
        return Linear(weight=weight, bias=bias)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SimpLeNet:
    c1: Conv2d
    s2: Pool2d
    c3: Conv2d
    s4: Pool2d
    c5: Conv2d
    f6: Linear
    out: Linear

    def __call__(self, x: Float[Array, "1 32 32"]) -> Float[Array, "10"]:
        x = x.astype(PRECISION)

        x = scaled_tanh(self.c1(x))
        x = scaled_tanh(self.s2(x))
        x = scaled_tanh(self.c3(x))
        x = scaled_tanh(self.s4(x))
        x = scaled_tanh(self.c5(x))
        x = jnp.ravel(x)
        x = scaled_tanh(self.f6(x))
        x = self.out(x)
        return x

    def batch_inference(self, x: Float[Array, "b 1 32 32"]) -> Float[Array, "b 10"]:
        return jax.vmap(self.__call__)(x)

    @classmethod
    def init(cls, key: jtp.PRNGKeyArray):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        c1 = Conv2d.init(k1, ksize=5, din=1, dout=6)
        s2 = Pool2d.init(din=6)
        c3 = Conv2d.init(k2, ksize=5, din=6, dout=16)
        s4 = Pool2d.init(din=16)
        c5 = Conv2d.init(k3, ksize=5, din=16, dout=120)
        f6 = Linear.init(k4, din=120, dout=84)
        out = Linear.init(k5, din=84, dout=10)
        return SimpLeNet(c1=c1, s2=s2, c3=c3, s4=s4, c5=c5, f6=f6, out=out)

    @property
    def num_trainable_parameters(self) -> int:
        return get_num_trainable_parameters(self)

    @property
    def num_nontrainable_parameters(self) -> int:
        return 0


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SimpLeNetEnsemble:
    # nets: atp.Sequence[SimpLeNet]
    stacked_net: SimpLeNet

    def __call__(
        self,
        x: Float[Array, "1 32 32"],
        net_enables: jtp.Bool[Array, "num_nets"],
    ) -> Float[Array, "10"]:
        # outs: Float[Array, "num_nets 10"] = jnp.asarray([net(x) for net in self.nets])
        outs: Float[Array, "num_nets 10"] = jax.vmap(lambda net: net(x))(
            self.stacked_net
        )
        net_enables = jax.lax.stop_gradient(net_enables).astype(outs.dtype)[..., None]
        active_count = jnp.maximum(jnp.sum(net_enables, dtype=jnp.float32), 1.0)
        out = jnp.sum(outs * net_enables, axis=0) / active_count
        return out

    def batch_inference(
        self,
        x: Float[Array, "b 1 32 32"],
        net_enables: jtp.Bool[Array, "num_nets"],
    ) -> Float[Array, "b 10"]:
        return jax.vmap(self.__call__, in_axes=(0, None))(x, net_enables)

    @classmethod
    def init(
        cls,
        key: jtp.PRNGKeyArray,
        num_nets: int = 1,
    ):
        assert num_nets > 0

        # nets = []
        # for _ in range(num_nets):
        #     key_net, key = jax.random.split(key)
        #     nets.append(SimpLeNet.init(key_net))
        # return SimpLeNetEnsemble(nets=tuple(nets))

        keys = jax.random.split(key, num_nets)
        stacked_net = jax.vmap(SimpLeNet.init)(keys)
        return SimpLeNetEnsemble(stacked_net=stacked_net)

    @property
    def num_trainable_parameters(self) -> int:
        return get_num_trainable_parameters(self)

    @property
    def num_nontrainable_parameters(self) -> int:
        return 0


# --- Data --- #


def get_mnist(fpath: str | PathLike[str]):
    data = jnp.load(fpath)
    x_train = jnp.asarray(data["x_train"], dtype=jnp.uint8)
    x_test = jnp.asarray(data["x_test"], dtype=jnp.uint8)
    y_train = jnp.asarray(data["y_train"], dtype=jnp.int32)
    y_test = jnp.asarray(data["y_test"], dtype=jnp.int32)

    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


# @functools.partial(
#     jax.jit,
#     static_argnames=("batch_size", "num_nets_total", "num_nets_per_train"),
# )
def sample_batch(
    key: jtp.PRNGKeyArray,
    xs: jtp.Shaped[Array, "b ..."],
    ys: jtp.Shaped[Array, "b ..."],
    batch_size: int,
    num_nets_total: int,
    num_nets_per_train: int,
):
    key_indices, key_net_choices = jax.random.split(key)
    indices = jax.random.choice(
        key_indices,
        a=xs.shape[0],
        shape=(batch_size,),
        replace=False,
    )
    nets_enables: jtp.Bool[Array, "num_nets"] = jax.random.uniform(
        key_net_choices,
        shape=(num_nets_total,),
    ) < (num_nets_per_train / num_nets_total)
    return xs[indices], ys[indices], nets_enables


def get_data_generator(
    key: jtp.PRNGKeyArray,
    xs: jtp.Shaped[Array, "b ..."],
    ys: jtp.Shaped[Array, "b ..."],
    batch_size: int,
    num_nets_total: int,
    num_nets_per_train: int,
):
    assert xs.shape[0] == ys.shape[0]

    num_samples = xs.shape[0]
    assert num_samples >= batch_size

    while True:
        key_sample, key = jax.random.split(key)
        x_batch, y_batch, nets_enabled = sample_batch(
            key_sample,
            xs,
            ys,
            batch_size,
            num_nets_total,
            num_nets_per_train,
        )
        yield x_batch, y_batch, nets_enabled


# --- Training --- #


def loss_fn(
    logits: Float[Array, "b 10"],
    labels: jtp.Int32[Array, "b"],
) -> Float[Array, ""]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -log_probs[jnp.arange(labels.shape[0]), labels]
    return jnp.mean(nll)


def compute_loss(
    model: SimpLeNetEnsemble,
    xs: Float[Array, "b 32 32"],
    ys: Float[Array, "b"],
    nets_enables: jtp.Bool[Array, "num_nets"],
) -> tuple[Float[Array, ""], Float[Array, "b 10"]]:
    logits = model.batch_inference(xs, nets_enables)
    loss = loss_fn(logits, ys)
    return loss, logits


def get_avg_accuracy(
    logits: Float[Array, "b 10"],
    labels: Float[Array, "b"],
) -> Float[Array, ""]:
    accuracies = jnp.argmax(logits, axis=-1, keepdims=False) == labels
    accuracy_avg = jnp.mean(accuracies, axis=0)
    return accuracy_avg


@jax.jit
def train_step(
    model: SimpLeNetEnsemble,
    xs: Float[Array, "b 28 28"],
    ys: Float[Array, "b"],
    nets_enables: jtp.Bool[Array, "num_nets"],
    lr: Float[Array, ""],
) -> tuple[Float[Array, ""], SimpLeNetEnsemble]:
    xs = vtransform(jax.lax.stop_gradient(xs)).astype(PRECISION)
    model_casted = jax.tree.map(lambda l: l.astype(PRECISION), model)

    (loss, logits), grad = jax.value_and_grad(compute_loss, has_aux=True)(
        model_casted, xs, ys, nets_enables
    )
    grad_f32 = jax.tree.map(lambda g: g.astype(jnp.float32), grad)
    model = jax.tree.map(lambda w, g: w - lr * g, model, grad_f32)
    return loss, model


@jax.jit
def evaluation(
    model: SimpLeNetEnsemble,
    xs: Float[Array, "10000 28 28"],
    ys: Float[Array, "10000"],
    nets_enables: jtp.Bool[Array, "num_nets"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    xs = vtransform(jax.lax.stop_gradient(xs)).astype(PRECISION)
    model = jax.tree.map(lambda l: l.astype(PRECISION), jax.lax.stop_gradient(model))

    loss, logits = compute_loss(model, xs, ys, nets_enables)
    accuracy = get_avg_accuracy(logits, ys)
    return loss, accuracy


# --- Visualization --- #


def vis_digits(
    digits: Float[Array, "n h w"],
    labels: jtp.Int[Array, "n"],
    model: SimpLeNetEnsemble,
    net_enables: jtp.Bool[Array, "num_nets"],
) -> mp.plot:
    # shrink and normalise images
    digs = einops.reduce(
        (digits + 0.1) / 1.275,
        "b (h 2) (w 2) -> b h w",
        "mean",
    )
    width = digs.shape[-1]

    # classify digits and mark correct or incorrect (apply vtransform & softmax)
    pred_logits = model.batch_inference(vtransform(digits), net_enables)
    pred_probs = jax.nn.softmax(pred_logits, axis=-1)
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
    losses_np = np.asarray(losses)
    accuracies_np = np.asarray(accuracies)
    loss_plot = mp.axes(
        mp.scatter(
            (losses_np, "magenta"),
            xrange=(0, total_num_steps - 1),
            yrange=(0, max(l for s, l in losses_np)),
            width=28,
            height=9,
        ),
        title=f"cross entropy {losses_np[-1][1]:.3f}",
        xlabel="train steps",
    )
    acc_plot = mp.axes(
        mp.scatter(
            (accuracies_np, "cyan"),
            xrange=(0, total_num_steps - 1),
            yrange=(0, 1),
            width=28,
            height=9,
        ),
        title=f"test accuracy {accuracies_np[-1][1]:.2%}",
        xlabel="train steps",
    )
    return loss_plot + acc_plot


# --- Entry --- #


def main(
    num_nets_total: int = 10,
    num_nets_per_train: int = 3,
    lr: float = 5e-2,
    bs: int = 512,
    num_steps: int = 256,
    steps_per_eval: int = 4,
    seed: int = 42,
    log_level: tp.Literal["DEBUG", "INFO", "ERROR"] = "INFO",
) -> None:
    logger.setLevel(log_level)

    key = jax.random.key(seed)

    num_steps = int(num_steps * num_nets_total / num_nets_per_train)
    lr = lr * num_nets_per_train

    key_model, key = jax.random.split(key)
    model = SimpLeNetEnsemble.init(key_model, num_nets=num_nets_total)
    logger.info("Trainable parameter num: %s", model.num_trainable_parameters)
    logger.info("Non-trainable parameter num: %s", model.num_nontrainable_parameters)

    x0 = jnp.zeros((1, 32, 32), dtype=jnp.float32)
    enables0 = get_padded_array([True], size=num_nets_total, dtype=jnp.bool)
    model(x0, enables0)
    enables1 = get_padded_array(
        [True] * num_nets_total,
        size=num_nets_total,
        dtype=jnp.bool,
    )
    model(x0, enables1)
    logger.info("Model sanity checks passed!")

    data = get_mnist("/home/amas/Desktop/Projects/hijax/data/mnist.npz")
    key_loader_train, key = jax.random.split(key)
    train_dataloader = get_data_generator(
        key_loader_train,
        xs=data["x_train"],
        ys=data["y_train"],
        batch_size=bs,
        num_nets_total=num_nets_total,
        num_nets_per_train=num_nets_per_train,
    )

    x0, y0, nets_enables = next(train_dataloader)
    assert x0.shape == (bs, 28, 28) and x0.dtype == jnp.uint8
    assert y0.shape == (bs,) and y0.dtype == jnp.int32
    logger.info("Data loader sanity checks passed!")

    train_losses = []
    test_losses_net0, test_losses_allnet = [], []
    test_accuracies_net0, test_accuracies_allnet = [], []

    for step in range(num_steps):
        x_batch, y_batch, nets_enables = next(train_dataloader)
        train_loss, model = train_step(model, x_batch, y_batch, nets_enables, lr)
        train_losses.append(train_loss.item())
        logger_contexted = logger.contexted({"step": f"S{step + 1}/{num_steps}"})

        logger_contexted.update(train_loss=train_loss.item())

        if step == (num_steps - 1) or step % steps_per_eval == 0:
            net0_enable = get_padded_array([True], num_nets_total, dtype=jnp.bool)
            test_loss_net0, test_accuracy_net0 = evaluation(
                model, data["x_test"], data["y_test"], net0_enable
            )
            all_net_enable = jnp.ones(num_nets_total, dtype=jnp.bool)
            test_loss_allnet, test_accuracy_allnet = evaluation(
                model, data["x_test"], data["y_test"], all_net_enable
            )

            # Store step-value tuples so vis_metrics can unpack (step, metric)
            test_losses_net0.append((step, test_loss_net0.item()))
            test_losses_allnet.append((step, test_loss_allnet.item()))
            test_accuracies_net0.append((step, test_accuracy_net0.item()))
            test_accuracies_allnet.append((step, test_accuracy_allnet.item()))

            logger_contexted.update(
                test_loss_net0=test_loss_net0.item(),
                test_loss_allnet=test_loss_allnet.item(),
                test_accuracy_net0=test_accuracy_net0.item() * 100,
                test_accuracy_allnet=test_accuracy_allnet.item() * 100,
            )
            logger_contexted.contexted({"stage": "test"}, cover=True).log_metric(
                level="INFO",
                metrics=[
                    ("train_loss", "[%(metricname)s: %(lastval).4f (%(median).4f)]"),
                    ("test_loss_net0", "[%(metricname)s: %(lastval).4f]"),
                    ("test_accuracy_net0", "[%(metricname)s: %(lastval).2f]"),
                    ("test_loss_allnet", "[%(metricname)s: %(lastval).4f]"),
                    ("test_accuracy_allnet", "[%(metricname)s: %(lastval).2f]"),
                ],
            )

            # Visualization moved inside the evaluation condition
            logger.info("Net 0 result:")
            digit_plot_net0 = vis_digits(
                digits=data["x_test"][jnp.asarray((1, 6, 7, 8))],
                labels=data["y_test"][jnp.asarray((1, 6, 7, 8))],
                model=model,
                net_enables=get_padded_array([True], num_nets_total, dtype=jnp.bool),
            )
            metrics_plot_net0 = vis_metrics(
                losses=test_losses_net0,
                accuracies=test_accuracies_net0,
                total_num_steps=num_steps,
            )
            plot_net0 = digit_plot_net0 / metrics_plot_net0

            logger.info("Ensembling result:")
            digit_plot_allnet = vis_digits(
                digits=data["x_test"][np.asarray((1, 6, 7, 8))],
                labels=data["y_test"][np.asarray((1, 6, 7, 8))],
                model=model,
                net_enables=jnp.ones(num_nets_total, dtype=jnp.bool),
            )
            metrics_plot_allnet = vis_metrics(
                losses=test_losses_allnet,
                accuracies=test_accuracies_allnet,
                total_num_steps=num_steps,
            )
            plot_allnet = digit_plot_allnet / metrics_plot_allnet

            plot = (
                mp.text("Net 0") / plot_net0
                + mp.text(f"Ensembling ({num_nets_total} nets)") / plot_allnet
            )

            if log_level == "ERROR" and step > 0:  # Higher than info
                print(f"{-plot}{plot}")
            else:
                print(plot)

            plot.saveimg("ensembling-result.png")

        else:
            logger_contexted.contexted({"stage": "train"}, cover=True).log_metric(
                level="DEBUG",
                metrics=[
                    ("train_loss", "[%(metricname)s: %(lastval).4f (%(median).4f)]")
                ],
            )


if __name__ == "__main__":
    tyro.cli(main)
