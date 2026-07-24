import tyro
import jaxtyping as jtp
import imageio.v3 as iio
import jax.numpy as jnp
import jax


def get_init_pos(
    init_shape: tuple[int, ...],
    canva_shape: tuple[int, ...],
) -> tuple[slice, slice]:
    iw, ih = init_shape
    cw, ch = canva_shape
    init_lt = ((cw - iw) // 2, (ch - ih) // 2)
    init_rb = (init_lt[0] + iw, init_lt[1] + ih)
    return slice(init_lt[0], init_rb[0]), slice(init_lt[1], init_rb[1])


@jax.jit
def simulate_single_step(
    state: jtp.Bool[jtp.Array, "w h"],
) -> jtp.Bool[jtp.Array, "w h"]:
    state_int = state.astype(jnp.uint8)
    neighbors: jtp.UInt8[jtp.Array, "w h"] = (
        jnp.roll(state_int, (1, 1), axis=(0, 1))
        + jnp.roll(state_int, (1, 0), axis=(0, 1))
        + jnp.roll(state_int, (1, -1), axis=(0, 1))
        + jnp.roll(state_int, (0, 1), axis=(0, 1))
        + jnp.roll(state_int, (0, -1), axis=(0, 1))
        + jnp.roll(state_int, (-1, 1), axis=(0, 1))
        + jnp.roll(state_int, (-1, 0), axis=(0, 1))
        + jnp.roll(state_int, (-1, -1), axis=(0, 1))
    )
    return (neighbors == 3) | (state & (neighbors == 2))


def simulate(
    initial: jtp.Bool[jtp.Array, "w h"],
    num_steps: int,
) -> jtp.Bool[jtp.Array, "t w h"]:
    def scan_step(carry, _):
        next_carry = simulate_single_step(carry)
        return next_carry, next_carry

    _, states = jax.lax.scan(scan_step, initial, length=num_steps - 1)
    return jnp.concat([initial[None], states])


def save_gif(states: jtp.Bool[jtp.Array, "t w h"], duration: int):
    iio.imwrite(
        "output.gif", (states.astype(jnp.uint8) * 255), duration=duration, loop=0
    )


def main(
    canvas_shape: tuple[int, int] = (80, 80),
    num_steps: int = 6000,
    duration: int = 20,
):
    initial_py = [
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
    ]
    initial = jnp.asarray(initial_py, dtype=jnp.bool)
    init_pos = get_init_pos(initial.shape, canvas_shape)
    canvas: jtp.Bool[jtp.Array, "w h"] = jnp.zeros(canvas_shape, dtype=jnp.bool)
    canvas = canvas.at[init_pos].set(initial)
    states = simulate(canvas, num_steps)
    save_gif(states, duration)


if __name__ == "__main__":
    tyro.cli(main)
