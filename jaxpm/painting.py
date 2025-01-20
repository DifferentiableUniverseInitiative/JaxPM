from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpm.distributed import (autoshmap, fft3d, get_halo_size, halo_exchange,
                               ifft3d, slice_pad, slice_unpad)
from jaxpm.kernels import cic_compensation, fftk
from jaxpm.painting_utils import gather, scatter


def _cic_paint_impl(grid_mesh, positions, weight=None):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    displacement field: [nx, ny, nz, 3]
    """

    positions = positions.reshape([-1, 3])
    positions = jax.tree.map(lambda p: jnp.expand_dims(p, 1), positions)
    floor = jax.tree.map(jnp.floor, positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                             [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jax.tree.map(jnp.abs, (positions - neighboor_coords))
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    if weight is not None:
        if jax.tree.all(jax.tree.map(jnp.isscalar, weight)):
            kernel = jax.tree.map(
                lambda k, w: jnp.multiply(jnp.expand_dims(w, axis=-1), k),
                kernel, weight)
        else:
            kernel = jax.tree.map(
                lambda k, w: jnp.multiply(w.reshape(*positions.shape[:-1]), k),
                kernel, weight)

    neighboor_coords = jax.tree.map(
        lambda nc: jnp.mod(
            nc.reshape([-1, 8, 3]).astype('int32'), jnp.array(grid_mesh.shape)
        ), neighboor_coords)

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(),
                                            inserted_window_dims=(0, 1, 2),
                                            scatter_dims_to_operand_dims=(0, 1,
                                                                          2))
    mesh = jax.tree.map(
        lambda g, nc, k: lax.scatter_add(g, nc, k.reshape([-1, 8]), dnums),
        grid_mesh, neighboor_coords, kernel)

    return mesh


@partial(jax.jit, static_argnums=(3, 4))
def cic_paint(grid_mesh, positions, weight=None, halo_size=0, sharding=None):

    positions_structure = jax.tree.structure(positions)
    grid_mesh = jax.tree.unflatten(positions_structure,
                                   jax.tree.leaves(grid_mesh))
    positions = positions.reshape((*grid_mesh.shape, 3))

    halo_size, halo_extents = get_halo_size(halo_size, sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding)

    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    grid_mesh = autoshmap(_cic_paint_impl,
                          gpu_mesh=gpu_mesh,
                          in_specs=(spec, spec, P()),
                          out_specs=spec)(grid_mesh, positions, weight)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    grid_mesh = slice_unpad(grid_mesh, halo_size, sharding)

    return grid_mesh


def _cic_read_impl(grid_mesh, positions):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [nx,ny,nz, 3]
    """
    # Save original shape for reshaping output later
    original_shape = positions.shape
    # Reshape positions to a flat list of 3D coordinates
    positions = positions.reshape([-1, 3])
    # Expand dimensions to calculate neighbor coordinates
    positions = jax.tree.map(lambda p: jnp.expand_dims(p, 1), positions)
    # Floor the positions to get the base grid cell for each particle
    floor = jax.tree.map(jnp.floor, positions)
    # Define connections to calculate all neighbor coordinates
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                             [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]])
    # Calculate the 8 neighboring coordinates
    neighboor_coords = floor + connection
    # Calculate kernel weights based on distance from each neighboring coordinate
    kernel = 1. - jax.tree.map(jnp.abs, positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    # Modulo operation to wrap around edges if necessary
    neighboor_coords = jax.tree.map(
        lambda nc: jnp.mod(nc.astype('int32'), jnp.array(grid_mesh.shape)),
        neighboor_coords)

    # Ensure grid_mesh shape is as expected
    # Retrieve values from grid_mesh at each neighboring coordinate and multiply by kernel
    grid_mesh = jax.tree.map(
        lambda g, nc, k: g[nc[..., 0], nc[..., 1], nc[..., 2]] * k, grid_mesh,
        neighboor_coords, kernel)
    return grid_mesh.sum(axis=-1).reshape(original_shape[:-1]) # yapf: disable


@partial(jax.jit, static_argnums=(2, 3))
def cic_read(grid_mesh, positions, halo_size=0, sharding=None):

    original_shape = positions.shape
    positions = positions.reshape((*grid_mesh.shape, 3))

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding=sharding)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()

    displacement = autoshmap(_cic_read_impl,
                             gpu_mesh=gpu_mesh,
                             in_specs=(spec, spec),
                             out_specs=spec)(grid_mesh, positions)

    return displacement.reshape(original_shape[:-1])


def cic_paint_2d(mesh, positions, weight):
    """ Paints positions onto a 2d mesh
    mesh: [nx, ny]
    positions: [npart, 2]
    weight: [npart]
    """
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[0, 0], [1., 0], [0., 1], [1., 1]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1]
    if weight is not None:
        kernel = kernel * weight[..., jnp.newaxis]

    neighboor_coords = jnp.mod(
        neighboor_coords.reshape([-1, 4, 2]).astype('int32'),
        jnp.array(mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(),
                                            inserted_window_dims=(0, 1),
                                            scatter_dims_to_operand_dims=(0,
                                                                          1))
    mesh = lax.scatter_add(mesh, neighboor_coords, kernel.reshape([-1, 4]),
                           dnums)
    return mesh


def _cic_paint_dx_impl(displacements, halo_size, weight=1., chunk_size=2**24):

    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]

    original_shape = displacements.shape
    particle_mesh = jax.tree.map(
        lambda x: jnp.zeros(x.shape[:-1], dtype=displacements.dtype),
        displacements)
    if not jnp.isscalar(weight):
        if weight.shape != original_shape[:-1]:
            raise ValueError("Weight shape must match particle shape")
        else:
            weight = weight.flatten()
    # Padding is forced to be zero in a single gpu run

    a, b, c = jax.tree.map(
        lambda x: jnp.stack(jnp.meshgrid(jnp.arange(x.shape[0]),
                                         jnp.arange(x.shape[1]),
                                         jnp.arange(x.shape[2]),
                                         indexing='ij'),
                            axis=0), particle_mesh)

    particle_mesh = jax.tree.map(lambda x: jnp.pad(x, halo_size),
                                 particle_mesh)
    pmid = jax.tree.map(
        lambda a, b, c: jnp.stack([a + halo_x, b + halo_y, c], axis=-1), a, b,
        c)
    return scatter(pmid.reshape([-1, 3]),
                   displacements.reshape([-1, 3]),
                   particle_mesh,
                   chunk_size=2**24,
                   val=weight)


@partial(jax.jit, static_argnums=(1, 2, 4))
def cic_paint_dx(displacements,
                 halo_size=0,
                 sharding=None,
                 weight=1.0,
                 chunk_size=2**24):

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)

    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    grid_mesh = autoshmap(partial(_cic_paint_dx_impl,
                                  halo_size=halo_size,
                                  weight=weight,
                                  chunk_size=chunk_size),
                          gpu_mesh=gpu_mesh,
                          in_specs=spec,
                          out_specs=spec)(displacements)

    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    grid_mesh = slice_unpad(grid_mesh, halo_size, sharding)
    return grid_mesh


def _cic_read_dx_impl(grid_mesh, disp, halo_size):

    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]

    original_shape = [
        dim - 2 * halo[0] for dim, halo in zip(grid_mesh.shape, halo_size)
    ]
    a, b, c = jnp.meshgrid(jnp.arange(original_shape[0]),
                           jnp.arange(original_shape[1]),
                           jnp.arange(original_shape[2]),
                           indexing='ij')
    a, b, c = jax.tree.map(
        lambda x: jnp.stack(jnp.meshgrid(jnp.arange(original_shape[0]),
                                         jnp.arange(original_shape[1]),
                                         jnp.arange(original_shape[2]),
                                         indexing='ij'),
                            axis=0), grid_mesh)

    pmid = jax.tree.map(
        lambda a, b, c: jnp.stack([a + halo_x, b + halo_y, c], axis=-1), a, b,
        c)
    pmid = pmid.reshape([-1, 3])
    disp = disp.reshape([-1, 3])

    return gather(pmid, disp, grid_mesh).reshape(original_shape)


@partial(jax.jit, static_argnums=(2, 3))
def cic_read_dx(grid_mesh, disp, halo_size=0, sharding=None):

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding=sharding)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    displacements = autoshmap(partial(_cic_read_dx_impl, halo_size=halo_size),
                              gpu_mesh=gpu_mesh,
                              in_specs=(spec),
                              out_specs=spec)(grid_mesh, disp)

    return displacements


def compensate_cic(field):
    """
    Compensate for CiC painting
    Args:
      field: input 3D cic-painted field
    Returns:
      compensated_field
    """
    delta_k = fft3d(field)

    kvec = fftk(delta_k)
    delta_k = cic_compensation(kvec) * delta_k
    return ifft3d(delta_k)
