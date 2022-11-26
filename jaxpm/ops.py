# Module for custom ops, typically mpi4jax
import jax
import jax.numpy as jnp
import mpi4jax
import jaxdecomp
from dataclasses import dataclass
from typing import Tuple

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
        arr = jaxdecomp.pfft3d(arr, 
                     pdims=sharding_info.pdims,
                     global_shape=sharding_info.global_shape)
    return arr


def ifft3d(arr, sharding_info=None):
    if sharding_info is None:
        arr = jnp.fft.ifftn(arr.transpose([2, 0, 1]))
    else:
        arr = jaxdecomp.pifft3d(arr, 
                     pdims=sharding_info.pdims,
                     global_shape=sharding_info.global_shape)
    return arr


def halo_reduce(arr, sharding_info=None):
    if sharding_info is None:
        return arr
    halo_size = sharding_info.halo_extents[0]
    global_shape = sharding_info.global_shape
    arr = jaxdecomp.halo_exchange(arr,
                                halo_extents=(halo_size//2, halo_size//2, 0),
                                halo_periods=(True,True,True),
                                pdims=sharding_info.pdims,
                                global_shape=(global_shape[0]+2*halo_size, 
                                              global_shape[1]+halo_size,
                                              global_shape[2]))

    # Apply correction along x
    arr = arr.at[halo_size:halo_size + halo_size//2].add(arr[ :halo_size//2])
    arr = arr.at[-halo_size - halo_size//2:-halo_size].add(arr[-halo_size//2:])

    # Apply correction along y
    arr = arr.at[:, halo_size:halo_size + halo_size//2].add(arr[:, :halo_size//2][:, :])
    arr = arr.at[:, -halo_size - halo_size//2:-halo_size].add(arr[:, -halo_size//2:][:, :])
    
    return arr


def meshgrid3d(shape, sharding_info=None):
    if sharding_info is not None:
        coords = [jnp.arange(sharding_info.global_shape[0]//sharding_info.pdims[1]),
                  jnp.arange(sharding_info.global_shape[1]//sharding_info.pdims[0]), jnp.arange(sharding_info.global_shape[2])]
    else:
        coords = [jnp.arange(s) for s in shape[2:]]

    return jnp.stack(jnp.meshgrid(*coords), axis=-1).reshape([-1, 3])


def zeros(shape, sharding_info=None):
    """ Initialize an array of given global shape
    partitionned if need be accross dimensions.
    """
    if sharding_info is None:
        return jnp.zeros(shape)

    return jnp.zeros([sharding_info.global_shape[0]//sharding_info.pdims[1], sharding_info.global_shape[1]//sharding_info.pdims[0]]+list(sharding_info.global_shape[2:]))


def normal(key, shape, sharding_info=None):
    """ Generates a normal variable for the given
    global shape.
    """
    if sharding_info is None:
        return jax.random.normal(key, shape)

    return jax.random.normal(key,
                            [sharding_info.global_shape[0]//sharding_info.pdims[1], sharding_info.global_shape[1]//sharding_info.pdims[0], sharding_info.global_shape[2]])
