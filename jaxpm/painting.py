import jax
import jax.numpy as jnp
import jax.lax as lax

from jaxpm.ops import halo_reduce
from jaxpm.kernels import fftk, cic_compensation


def cic_paint(mesh, positions, halo_size=0, token=None, comms=None):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [npart, 3]
    """
    if comms is not None:
        # Add some padding for the halo exchange
        mesh = jnp.pad(mesh, [[halo_size, halo_size],
                              [halo_size, halo_size],
                              [0, 0]])
        positions += jnp.array([halo_size, halo_size, 0]).reshape([-1, 3])

    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0],
                             [0., 0, 1], [1., 1, 0], [1., 0, 1],
                             [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords = jnp.mod(neighboor_coords.reshape(
        [-1, 8, 3]).astype('int32'), jnp.array(mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2))
    mesh = lax.scatter_add(mesh,
                           neighboor_coords,
                           kernel.reshape([-1, 8]),
                           dnums)

    if comms == None:
        return mesh
    else:
        mesh, token = halo_reduce(mesh, halo_size, token, comms)
        return mesh[halo_size:-halo_size, halo_size:-halo_size]


def cic_read(mesh, positions, halo_size=0, token=None, comms=None):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [npart, 3]
    """

    if comms is not None:
        # Add some padding and perfom hao exchange to retrieve
        # neighboring regions
        mesh = jnp.pad(mesh, [[halo_size, halo_size],
                              [halo_size, halo_size],
                              [0, 0]])
        mesh, token = halo_reduce(mesh, halo_size, token, comms)
        positions += jnp.array([halo_size, halo_size, 0]).reshape([-1, 3])

    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0],
                             [0., 0, 1], [1., 1, 0], [1., 0, 1],
                             [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords = jnp.mod(
        neighboor_coords.astype('int32'), jnp.array(mesh.shape))

    res = (mesh[neighboor_coords[..., 0],
                neighboor_coords[..., 1],
                neighboor_coords[..., 3]]*kernel).sum(axis=-1)

    if comms is not None:
        return res
    else:
        return res, token


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

    neighboor_coords = jnp.mod(neighboor_coords.reshape(
        [-1, 4, 2]).astype('int32'), jnp.array(mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1))
    mesh = lax.scatter_add(mesh,
                           neighboor_coords,
                           kernel.reshape([-1, 4]),
                           dnums)
    return mesh


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
