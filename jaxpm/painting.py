import jax
import jax.numpy as jnp
import jax.lax as lax

from jaxpm.ops import halo_reduce
from jaxpm.kernels import fftk, cic_compensation
import jaxdecomp
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P,NamedSharding
from jax.experimental.shard_map import shard_map


@partial(jax.jit,static_argnums=(1))
def add_halo(positions , halo_size):
    positions += jnp.array([halo_size, halo_size, 0]).reshape([-1, 3])
    return positions



def cic_paint(gpu_mesh,nbody_mesh, positions, halo_size=0, sharding_info=None):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [npart, 3]
    """
    print(f" positions {positions.shape}")
    if sharding_info is not None:

        @partial(shard_map, mesh=gpu_mesh, in_specs=P('z', 'y'),
         out_specs=P('z', 'y'))
        def sharded_pad(arr):
            padded =  jnp.pad(arr,pad_width=((halo_size, halo_size), (halo_size, halo_size), (0, 0)))
            return padded
        
        # Add some padding for the halo exchange
        with gpu_mesh:
            nbody_mesh = sharded_pad(nbody_mesh)
            positions = add_halo(positions , halo_size)

    with gpu_mesh:
        positions = jnp.expand_dims(positions, 1)
        floor = jnp.floor(positions)

    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0],
                             [0., 0, 1], [1., 1, 0], [1., 0, 1],
                             [0., 1, 1], [1., 1, 1]]])

    with gpu_mesh:
        neighboor_coords = floor + connection
        kernel = 1. - jnp.abs(positions - neighboor_coords)
        kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

        neighboor_coords = jnp.mod(neighboor_coords.reshape(
            [-1, 8, 3]).astype('int32'), jnp.array(nbody_mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2))
    
    with gpu_mesh:
        nbody_mesh = lax.scatter_add(nbody_mesh,
                            neighboor_coords,
                            kernel.reshape([-1, 8]),
                            dnums)

    if sharding_info == None:
        return nbody_mesh
    else:
        with gpu_mesh : 
            nbody_mesh = halo_reduce(nbody_mesh, sharding_info)
            nbody_mesh =  nbody_mesh[halo_size:-halo_size, halo_size:-halo_size]

        return nbody_mesh

@jax.jit
def reduce_and_sum(mesh,neighboor_coords,kernel):
    return (mesh[neighboor_coords[..., 0],
                 neighboor_coords[..., 1],
                 neighboor_coords[..., 3]]*kernel).sum(axis=-1)



def cic_read(gpu_mesh , mesh, positions, halo_size=0, sharding_info=None):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [npart, 3]
    """
    @partial(shard_map, mesh=gpu_mesh, in_specs=(P('z', 'y'),P()),
        out_specs=P('z', 'y'))
    def sharded_pad(arr , padding_width):
        return  jnp.pad(arr,pad_width=padding_width)

    if sharding_info is not None:
        # Add some padding and perfom hao exchange to retrieve
        # neighboring regions
        
        # mesh = halo_reduce(mesh, sharding_info)
        with gpu_mesh:
            padding_width = jnp.array([(halo_size, halo_size), (halo_size, halo_size), (0, 0)])
            #mesh = sharded_pad(mesh,padding_width)
            mesh = jaxdecomp.halo_exchange(mesh,
                            halo_extents=sharding_info.halo_extents,
                                halo_periods=(True,True,True))
            positions = add_halo(positions , halo_size)

    with gpu_mesh:
        positions = jnp.expand_dims(positions, 1)
        floor = jnp.floor(positions)

    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0],
                             [0., 0, 1], [1., 1, 0], [1., 0, 1],
                             [0., 1, 1], [1., 1, 1]]])

    with gpu_mesh:
        neighboor_coords = floor + connection
        kernel = 1. - jnp.abs(positions - neighboor_coords)
        kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

        neighboor_coords = jnp.mod(
            neighboor_coords.astype('int32'), jnp.array(mesh.shape))

        reduced = reduce_and_sum(mesh,neighboor_coords,kernel)

    return reduced


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
