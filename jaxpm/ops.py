# Module for custom ops, typically mpi4jax
import jax
import jax.numpy as jnp
import jaxdecomp
from dataclasses import dataclass
from typing import Tuple
from functools import partial
from jax import jit
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P,NamedSharding
from jax.experimental.shard_map import shard_map
@dataclass
class ShardingInfo:
    """Class for keeping track of the distribution strategy"""
    global_shape: Tuple[int, int, int] 
    pdims: Tuple[int, int]
    halo_extents: Tuple[int, int, int]
    rank: int = 0


def fft3d(arr, sharding_info=None):
    """ Computes forward FFT, note that the output is transposed
    """
    if sharding_info is None:
        arr = jnp.fft.fftn(arr).transpose([1, 2, 0])
    else:
        arr = jaxdecomp.pfft3d(arr)
    return arr

def ifft3d(arr, sharding_info=None):
    if sharding_info is None:
        arr = jnp.fft.ifftn(arr.transpose([2, 0, 1]))
    else:
        arr = jaxdecomp.pifft3d(arr)
    return arr



def halo_reduce(arr, halo_size , gpu_mesh):

    with gpu_mesh:
        arr = jaxdecomp.halo_exchange(arr,
                                    halo_extents=(halo_size//2, halo_size//2, 0),
                                    halo_periods=(True,True,True))

    @partial(shard_map, mesh=gpu_mesh, in_specs=P('z', 'y'),out_specs=P('z', 'y'))
    def apply_correction_x(arr):
        arr = arr.at[halo_size:halo_size + halo_size//2].add(arr[ :halo_size//2])
        arr = arr.at[-halo_size - halo_size//2:-halo_size].add(arr[-halo_size//2:])

        return arr

    @partial(shard_map, mesh=gpu_mesh, in_specs=P('z', 'y'),out_specs=P('z', 'y'))
    def apply_correction_y(arr):
        arr = arr.at[:, halo_size:halo_size + halo_size//2].add(arr[:, :halo_size//2][:, :])
        arr = arr.at[:, -halo_size - halo_size//2:-halo_size].add(arr[:, -halo_size//2:][:, :])

        return arr

    @partial(shard_map, mesh=gpu_mesh, in_specs=P('z', 'y'),out_specs=P('z', 'y'))
    def un_pad(arr):
        return arr[halo_size:-halo_size, halo_size:-halo_size]
    

    # Apply correction along x
    arr = apply_correction_x(arr)
    # Apply correction along y
    arr = apply_correction_y(arr)

    arr = un_pad(arr)

    
    return arr


def meshgrid3d(shape, sharding_info=None):
    if sharding_info is not None:
        coords = [jnp.arange(sharding_info.global_shape[0]//sharding_info.pdims[1]),
                  jnp.arange(sharding_info.global_shape[1]//sharding_info.pdims[0]), jnp.arange(sharding_info.global_shape[2])]
    else:
        coords = [jnp.arange(s) for s in shape[2:]]

    return jnp.stack(jnp.meshgrid(*coords), axis=-1).reshape([-1, 3])

def zeros(mesh , shape, sharding_info=None):
    """ Initialize an array of given global shape
    partitionned if need be accross dimensions.
    """
    if sharding_info is None:
        return jnp.zeros(shape)

    zeros_slice = jnp.zeros([sharding_info.global_shape[0]//sharding_info.pdims[1], \
        sharding_info.global_shape[1]//sharding_info.pdims[0]]+list(sharding_info.global_shape[2:]))

    gspmd_zeros = multihost_utils.host_local_array_to_global_array(zeros_slice ,mesh, P('z' , 'y'))
    return gspmd_zeros


def normal(key, shape, sharding_info=None):
    """ Generates a normal variable for the given
    global shape.
    """
    if sharding_info is None:
        return jax.random.normal(key, shape)

    return jax.random.normal(key,
                            [sharding_info.global_shape[0]//sharding_info.pdims[1], sharding_info.global_shape[1]//sharding_info.pdims[0], sharding_info.global_shape[2]])
