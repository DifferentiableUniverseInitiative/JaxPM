from typing import Any, Callable, Hashable

Specs = Any
AxisName = Hashable

from functools import partial

import jax
import jax.numpy as jnp
import jaxdecomp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import AbstractMesh, Mesh
from jax.sharding import PartitionSpec as P


def autoshmap(
    f: Callable,
    gpu_mesh: Mesh | AbstractMesh | None,
    in_specs: Specs,
    out_specs: Specs,
    check_rep: bool = False,
    auto: frozenset[AxisName] = frozenset()) -> Callable:
    """Helper function to wrap the provided function in a shard map if
    the code is being executed in a mesh context."""
    if gpu_mesh is None or gpu_mesh.empty:
        return f
    else:
        return shard_map(f, gpu_mesh, in_specs, out_specs, check_rep, auto)


def fft3d(x):
    return jaxdecomp.pfft3d(x)


def ifft3d(x):
    return jaxdecomp.pifft3d(x).real


def get_halo_size(halo_size, sharding):
    gpu_mesh = sharding.mesh if sharding is not None else None
    if gpu_mesh is None or gpu_mesh.empty:
        zero_ext = (0, 0)
        zero_tuple = (0, 0)
        return (zero_tuple, zero_tuple, zero_tuple), zero_ext
    else:
        pdims = gpu_mesh.devices.shape
    halo_x = (0, 0) if pdims[0] == 1 else (halo_size, halo_size)
    halo_y = (0, 0) if pdims[1] == 1 else (halo_size, halo_size)

    halo_x_ext = 0 if pdims[0] == 1 else halo_size // 2
    halo_y_ext = 0 if pdims[1] == 1 else halo_size // 2
    return ((halo_x, halo_y, (0, 0)), (halo_x_ext, halo_y_ext))


def halo_exchange(x, halo_extents, halo_periods=(True, True)):
    if (halo_extents[0] > 0 or halo_extents[1] > 0):
        return jaxdecomp.halo_exchange(x, halo_extents, halo_periods)
    else:
        return x


def slice_unpad_impl(x, pad_width):

    halo_x, _ = pad_width[0]
    halo_y, _ = pad_width[1]
    # Apply corrections along x
    x = x.at[halo_x:halo_x + halo_x // 2].add(x[:halo_x // 2])
    x = x.at[-(halo_x + halo_x // 2):-halo_x].add(x[-halo_x // 2:])
    # Apply corrections along y
    x = x.at[:, halo_y:halo_y + halo_y // 2].add(x[:, :halo_y // 2])
    x = x.at[:, -(halo_y + halo_y // 2):-halo_y].add(x[:, -halo_y // 2:])

    unpad_slice = [slice(None)] * 3
    if halo_x > 0:
        unpad_slice[0] = slice(halo_x, -halo_x)
    if halo_y > 0:
        unpad_slice[1] = slice(halo_y, -halo_y)

    return x[tuple(unpad_slice)]


def slice_pad_impl(x, pad_width):
    return jax.tree.map(lambda x: jnp.pad(x, pad_width), x)


def slice_pad(x, pad_width, sharding):
    gpu_mesh = sharding.mesh if sharding is not None else None
    if gpu_mesh is not None and not (gpu_mesh.empty) and (
            pad_width[0][0] > 0 or pad_width[1][0] > 0):
        assert sharding is not None
        spec = sharding.spec
        return shard_map((partial(slice_pad_impl, pad_width=pad_width)),
                         mesh=gpu_mesh,
                         in_specs=spec,
                         out_specs=spec)(x)
    else:
        return x


def slice_unpad(x, pad_width, sharding):
    mesh = sharding.mesh if sharding is not None else None
    if mesh is not None and not (mesh.empty) and (pad_width[0][0] > 0
                                                  or pad_width[1][0] > 0):
        assert sharding is not None
        spec = sharding.spec
        return shard_map(partial(slice_unpad_impl, pad_width=pad_width),
                         mesh=mesh,
                         in_specs=spec,
                         out_specs=spec)(x)
    else:
        return x


def get_local_shape(mesh_shape, sharding=None):
    """ Helper function to get the local size of a mesh given the global size.
  """
    gpu_mesh = sharding.mesh if sharding is not None else None
    if gpu_mesh is None or gpu_mesh.empty:
        return mesh_shape
    else:
        pdims = gpu_mesh.devices.shape
        return [
            mesh_shape[0] // pdims[0], mesh_shape[1] // pdims[1],
            *mesh_shape[2:]
        ]


def _axis_names(spec):
    if len(spec) == 1:
        x_axis, = spec
        y_axis = None
        single_axis = True
    elif len(spec) == 2:
        x_axis, y_axis = spec
        if y_axis == None:
            single_axis = True
        elif x_axis == None:
            x_axis = y_axis
            single_axis = True
        else:
            single_axis = False
    else:
        raise ValueError("Only 1 or 2 axis sharding is supported")
    return x_axis, y_axis, single_axis


def uniform_particles(mesh_shape, sharding=None):

    gpu_mesh = sharding.mesh if sharding is not None else None
    if gpu_mesh is not None and not (gpu_mesh.empty):
        local_mesh_shape = get_local_shape(mesh_shape, sharding)
        spec = sharding.spec
        x_axis, y_axis, single_axis = _axis_names(spec)

        def particles():
            x_indx = lax.axis_index(x_axis)
            y_indx = 0 if single_axis else lax.axis_index(y_axis)

            x = jnp.arange(local_mesh_shape[0]) + x_indx * local_mesh_shape[0]
            y = jnp.arange(local_mesh_shape[1]) + y_indx * local_mesh_shape[1]
            z = jnp.arange(local_mesh_shape[2])
            return jnp.stack(jnp.meshgrid(x, y, z, indexing='ij'), axis=-1)

        return shard_map(particles, mesh=gpu_mesh, in_specs=(),
                         out_specs=spec)()
    else:
        return jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape],
                                      indexing='ij'),
                         axis=-1)


def normal_field(mesh_shape, seed, sharding=None):
    """Generate a Gaussian random field with the given power spectrum."""
    gpu_mesh = sharding.mesh if sharding is not None else None
    if gpu_mesh is not None and not (gpu_mesh.empty):
        local_mesh_shape = get_local_shape(mesh_shape, sharding)

        size = jax.device_count()
        # rank = jax.process_index()
        # process_index is multi_host only
        # to make the code work both in multi host and single controller we can do this trick
        keys = jax.random.split(seed, size)
        spec = sharding.spec
        x_axis, y_axis, single_axis = _axis_names(spec)

        def normal(keys, shape, dtype):
            idx = lax.axis_index(x_axis)
            if not single_axis:
                y_index = lax.axis_index(y_axis)
                x_size = lax.psum(1, axis_name=x_axis)
                idx += y_index * x_size

            return jax.random.normal(key=keys[idx], shape=shape, dtype=dtype)

        return shard_map(
            partial(normal, shape=local_mesh_shape, dtype='float32'),
            mesh=gpu_mesh,
            in_specs=P(None),
            out_specs=spec)(keys)  # yapf: disable
    else:
        return jax.random.normal(shape=mesh_shape, key=seed)
