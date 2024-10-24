from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jaxpm.distributed import (autoshmap, get_halo_size, halo_exchange,
                               slice_pad, slice_unpad, fft3d, ifft3d)
from jaxpm.kernels import cic_compensation, fftk
from jaxpm.painting_utils import gather, scatter


def cic_paint_impl(grid_mesh, displacement, weight=None):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    displacement field: [nx, ny, nz, 3]
    """
    part_shape = displacement.shape
    positions = jnp.stack(jnp.meshgrid(jnp.arange(part_shape[0]),
                                       jnp.arange(part_shape[1]),
                                       jnp.arange(part_shape[2]),
                                       indexing='ij'),
                          axis=-1) + displacement
    positions = positions.reshape([-1, 3])
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                             [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    if weight is not None:
        kernel = jnp.multiply(jnp.expand_dims(weight, axis=-1), kernel)

    neighboor_coords = jnp.mod(
        neighboor_coords.reshape([-1, 8, 3]).astype('int32'),
        jnp.array(grid_mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(),
                                            inserted_window_dims=(0, 1, 2),
                                            scatter_dims_to_operand_dims=(0, 1,
                                                                          2))
    mesh = lax.scatter_add(grid_mesh, neighboor_coords,
                           kernel.reshape([-1, 8]), dnums)
    return mesh


@partial(jax.jit, static_argnums=(2, 3, 4))
def cic_paint(grid_mesh, positions, halo_size=0, weight=None, sharding=None):

    positions = positions.reshape((*grid_mesh.shape, 3))

    halo_size, halo_extents = get_halo_size(halo_size, sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding)

    gpu_mesh = sharding.mesh if sharding is not None else None
    spec = sharding.spec if sharding is not None else P()
    grid_mesh = autoshmap(cic_paint_impl,
                          gpu_mesh=gpu_mesh,
                          in_specs=(spec, spec, P()),
                          out_specs=spec)(grid_mesh, positions, weight)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    grid_mesh = slice_unpad(grid_mesh, halo_size, sharding)

    print(f"shape of grid_mesh: {grid_mesh.shape}")
    return grid_mesh


def cic_read_impl(mesh, displacement):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    displacement: [nx,ny,nz, 3]
    """
    # Compute the position of the particles on a regular grid
    part_shape = displacement.shape
    positions = jnp.stack(jnp.meshgrid(jnp.arange(part_shape[0]),
                                       jnp.arange(part_shape[1]),
                                       jnp.arange(part_shape[2]),
                                       indexing='ij'),
                          axis=-1) + displacement
    positions = positions.reshape([-1, 3])
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                             [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords = jnp.mod(neighboor_coords.astype('int32'),
                               jnp.array(mesh.shape))

    return (mesh[neighboor_coords[..., 0], neighboor_coords[..., 1],
                 neighboor_coords[..., 3]] * kernel).sum(axis=-1).reshape(
                     displacement.shape[:-1])


@partial(jax.jit, static_argnums=(2, 3))
def cic_read(grid_mesh, positions, halo_size=0, sharding=None):

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding=sharding)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    gpu_mesh = sharding.mesh if sharding is not None else None
    spec = sharding.spec if sharding is not None else P()
    displacement = autoshmap(cic_read_impl,
                             gpu_mesh=gpu_mesh,
                             in_specs=(spec, spec),
                             out_specs=spec)(grid_mesh, positions)
    print(f"shape of displacement: {displacement.shape}")

    return displacement


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


def cic_paint_dx_impl(displacements, halo_size):

    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]

    original_shape = displacements.shape
    particle_mesh = jnp.zeros(original_shape[:-1], dtype='float32')

    # Padding is forced to be zero in a single gpu run

    a, b, c = jnp.meshgrid(jnp.arange(particle_mesh.shape[0]),
                           jnp.arange(particle_mesh.shape[1]),
                           jnp.arange(particle_mesh.shape[2]),
                           indexing='ij')

    particle_mesh = jnp.pad(particle_mesh, halo_size)
    pmid = jnp.stack([a + halo_x, b + halo_y, c], axis=-1)
    pmid = pmid.reshape([-1, 3])
    return scatter(pmid, displacements.reshape([-1, 3]), particle_mesh)


@partial(jax.jit, static_argnums=(1, 2))
def cic_paint_dx(displacements, halo_size=0, sharding=None):

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)

    gpu_mesh = sharding.mesh if sharding is not None else None
    spec = sharding.spec if sharding is not None else P()
    grid_mesh = autoshmap(partial(cic_paint_dx_impl, halo_size=halo_size),
                          gpu_mesh=gpu_mesh,
                          in_specs=spec,
                          out_specs=spec)(displacements)

    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    grid_mesh = slice_unpad(grid_mesh, halo_size, sharding)
    return grid_mesh


def cic_read_dx_impl(grid_mesh, halo_size):

    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]

    original_shape = [
        dim - 2 * halo[0] for dim, halo in zip(grid_mesh.shape, halo_size)
    ]
    a, b, c = jnp.meshgrid(jnp.arange(original_shape[0]),
                           jnp.arange(original_shape[1]),
                           jnp.arange(original_shape[2]),
                           indexing='ij')

    pmid = jnp.stack([a + halo_x, b + halo_y, c], axis=-1)

    pmid = pmid.reshape([-1, 3])

    return gather(pmid, jnp.zeros_like(pmid),
                  grid_mesh).reshape(original_shape)


@partial(jax.jit, static_argnums=(1, 2))
def cic_read_dx(grid_mesh, halo_size=0, sharding=None):
    # return mesh
    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding=sharding)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    gpu_mesh = sharding.mesh if sharding is not None else None
    spec = sharding.spec if sharding is not None else P()
    displacements = autoshmap(partial(cic_read_dx_impl, halo_size=halo_size),
                              gpu_mesh=gpu_mesh,
                              in_specs=(spec),
                              out_specs=spec)(grid_mesh)

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
