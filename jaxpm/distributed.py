from typing import Any, Callable, Hashable

Specs = Any
AxisName = Hashable

try:
    import jaxdecomp
    distributed = True
except ImportError:
    print("jaxdecomp not installed. Distributed functions will not work.")
    distributed = False

from functools import partial

import jax
import jax.numpy as jnp
from jax._src import mesh as mesh_lib
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

# NOTE
# This should not be used as a decorator
# Must be used inside a function only
# Example
# BAD
# @autoshmap
# def foo():
#     pass
# GOOD
# def foo():
#     return autoshmap(foo_impl)()


def autoshmap(f: Callable,
              in_specs: Specs,
              out_specs: Specs,
              check_rep: bool = True,
              auto: frozenset[AxisName] = frozenset()):
    """Helper function to wrap the provided function in a shard map if
    the code is being executed in a mesh context."""
    mesh = mesh_lib.thread_resources.env.physical_mesh
    if mesh.empty:
        return f
    else:
        return shard_map(f, mesh, in_specs, out_specs, check_rep, auto)


def fft3d(x):
    if distributed and not (mesh_lib.thread_resources.env.physical_mesh.empty):
        return jaxdecomp.pfft3d(x.astype(jnp.complex64))
    else:
        return jnp.fft.fftn(x.astype(jnp.complex64))


def ifft3d(x):
    if distributed and not (mesh_lib.thread_resources.env.physical_mesh.empty):
        return jaxdecomp.pifft3d(x).real
    else:
        return jnp.fft.ifftn(x).real


def get_halo_size(halo_size):
    mesh = mesh_lib.thread_resources.env.physical_mesh
    if mesh.empty:
        zero_ext = (0, 0, 0)
        zero_tuple = (0, 0)
        return (zero_tuple, zero_tuple, zero_tuple), zero_ext
    else:
        pdims = mesh.devices.shape
    halo_x = (0, 0) if pdims[0] == 1 else (halo_size, halo_size)
    halo_y = (0, 0) if pdims[1] == 1 else (halo_size, halo_size)

    halo_x_ext = 0 if pdims[0] == 1 else halo_size // 2
    halo_y_ext = 0 if pdims[1] == 1 else halo_size // 2
    return ((halo_x, halo_y, (0, 0)), (halo_x_ext, halo_y_ext, 0))


def halo_exchange(x, halo_extents, halo_periods=(True, True, True)):
    mesh = mesh_lib.thread_resources.env.physical_mesh
    if distributed and not (mesh.empty) and (halo_extents[0] > 0
                                             or halo_extents[1] > 0):
        return jaxdecomp.halo_exchange(x, halo_extents, halo_periods)
    else:
        return x


def slice_unpad_impl(x, pad_width):

    halo_x, _ = pad_width[0]
    halo_y, _ = pad_width[0]

    # Apply corrections along x
    x = x.at[halo_x:halo_x + halo_x // 2].add(x[:halo_x // 2])
    x = x.at[-(halo_x + halo_x // 2):-halo_x].add(x[-halo_x // 2:])
    # Apply corrections along y
    x = x.at[:, halo_y:halo_y + halo_y // 2].add(x[:, :halo_y // 2])
    x = x.at[:, -(halo_y + halo_y // 2):-halo_y].add(x[:, -halo_y // 2:])

    return x[halo_x:-halo_x, halo_y:-halo_y, :]


def slice_pad(x, pad_width):
    mesh = mesh_lib.thread_resources.env.physical_mesh
    if distributed and not (mesh.empty) and (pad_width[0][0] > 0
                                             or pad_width[1][0] > 0):
        return autoshmap((partial(jnp.pad, pad_width=pad_width)),
                         in_specs=(P('x', 'y')),
                         out_specs=P('x', 'y'))(x)
    else:
        return x


def slice_unpad(x, pad_width):
    mesh = mesh_lib.thread_resources.env.physical_mesh
    if distributed and not (mesh.empty) and (pad_width[0][0] > 0
                                             or pad_width[1][0] > 0):
        return autoshmap(partial(slice_unpad_impl, pad_width=pad_width),
                         in_specs=(P('x', 'y')),
                         out_specs=P('x', 'y'))(x)
    else:
        return x


def get_local_shape(mesh_shape):
    """ Helper function to get the local size of a mesh given the global size.
  """
    if mesh_lib.thread_resources.env.physical_mesh.empty:
        return mesh_shape
    else:
        pdims = mesh_lib.thread_resources.env.physical_mesh.devices.shape
        return [
            mesh_shape[0] // pdims[0], mesh_shape[1] // pdims[1], mesh_shape[2]
        ]
    


def normal_field(mesh_shape, seed=None):
    """Generate a Gaussian random field with the given power spectrum."""
    if distributed and not (mesh_lib.thread_resources.env.physical_mesh.empty):
        local_mesh_shape = get_local_shape(mesh_shape)
        if seed is None:
            key = None
        else:
            size = jax.process_count()
            rank = jax.process_index()
            key = jax.random.split(seed, size)[rank]
        return autoshmap(
            partial(jax.random.normal, shape=local_mesh_shape, dtype='float32'),
            in_specs=P(None),
            out_specs=P('x', 'y'))(key)  # yapf: disable
    else:
        return jax.random.normal(shape=mesh_shape, key=seed)
