from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jaxpm.distributed import (autoshmap, get_halo_size, halo_exchange,
                               slice_pad, slice_unpad)
from jaxpm.kernels import cic_compensation, fftk
from jaxpm.painting_utils import gather, scatter


def cic_paint_impl(mesh, displacement, weight=None):
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
        jnp.array(mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(),
                                            inserted_window_dims=(0, 1, 2),
                                            scatter_dims_to_operand_dims=(0, 1,
                                                                          2))
    mesh = lax.scatter_add(mesh, neighboor_coords, kernel.reshape([-1, 8]),
                           dnums)
    return mesh


@partial(jax.jit, static_argnums=(2, ))
def cic_paint(mesh, positions, halo_size=0, weight=None):

    halo_padding, halo_extents = get_halo_size(halo_size)
    mesh = slice_pad(mesh, halo_padding)
    mesh = autoshmap(cic_paint_impl,
                     in_specs=(P('x', 'y'), P('x', 'y'), P()),
                     out_specs=P('x', 'y'))(mesh, positions, weight)
    mesh = halo_exchange(mesh,
                         halo_extents=halo_size // 2,
                         halo_periods=True)
    mesh = slice_unpad(mesh, halo_padding)
    return mesh


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


@partial(jax.jit, static_argnums=(2, ))
def cic_read(mesh, displacement, halo_size=0):

    halo_padding, halo_extents = get_halo_size(halo_size)
    mesh = slice_pad(mesh, halo_padding)
    mesh = halo_exchange(mesh,
                         halo_extents=halo_size//2,
                         halo_periods=True)
    displacement = autoshmap(cic_read_impl,
                             in_specs=(P('x', 'y'), P('x', 'y')),
                             out_specs=P('x', 'y'))(mesh, displacement)

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


@partial(jax.jit, static_argnums=(1, ))
def cic_paint_dx(displacements, halo_size=0):
<<<<<<< HEAD

    halo_size, halo_extents = get_halo_size(halo_size)

    mesh = autoshmap(partial(cic_paint_dx_impl, halo_size=halo_size),
=======
    
    halo_padding, halo_extents = get_halo_size(halo_size)
                
    mesh = autoshmap(partial(cic_paint_dx_impl, halo_size=halo_padding),
>>>>>>> glab/ASKabalan/jaxdecomp_proto
                     in_specs=(P('x', 'y')),
                     out_specs=P('x', 'y'))(displacements)

    mesh = halo_exchange(mesh,
                         halo_extents=halo_size//2,
                         halo_periods=True)
    mesh = slice_unpad(mesh, halo_padding)
    return mesh


def cic_read_dx_impl(mesh, halo_size):

    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]

    original_shape = [
        dim - 2 * halo[0] for dim, halo in zip(mesh.shape, halo_size)
    ]
    a, b, c = jnp.meshgrid(jnp.arange(original_shape[0]),
                           jnp.arange(original_shape[1]),
                           jnp.arange(original_shape[2]),
                           indexing='ij')

    pmid = jnp.stack([a + halo_x, b + halo_y, c], axis=-1)

    pmid = pmid.reshape([-1, 3])

    return gather(pmid, jnp.zeros_like(pmid), mesh).reshape(original_shape)


@partial(jax.jit, static_argnums=(1, ))
def cic_read_dx(mesh, halo_size=0):
    # return mesh
    halo_padding, halo_extents = get_halo_size(halo_size)
    mesh = slice_pad(mesh, halo_padding)
    mesh = halo_exchange(mesh,
<<<<<<< HEAD
                         halo_extents=halo_extents,
                         halo_periods=(True, True, True))
    displacements = autoshmap(partial(cic_read_dx_impl, halo_size=halo_size),
=======
                         halo_extents=halo_size//2,
                         halo_periods=True)
    displacements = autoshmap(partial(cic_read_dx_impl ,  halo_size=halo_padding),
>>>>>>> glab/ASKabalan/jaxdecomp_proto
                              in_specs=(P('x', 'y')),
                              out_specs=P('x', 'y'))(mesh)

    return displacements


def compensate_cic(field):
    """
  Compensate for CiC painting
  Args:
    field: input 3D cic-painted field
  Returns:
    compensated_field
  """
    nc = field.shape
    kvec = fftk(nc)

    delta_k = jnp.fft.rfftn(field)
    delta_k = cic_compensation(kvec) * delta_k
    return jnp.fft.irfftn(delta_k)
